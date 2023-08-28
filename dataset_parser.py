from __future__ import annotations

import os
from loguru import logger
from detection_improvement import DetectionImprovement
import cv2
import numpy as np
import shutil

from utils import sort_corners


class DatasetParser:
    def __init__(self, datasets_folder: str, output_folder: str, discarded_txt: str = "discarded.txt"):
        assert os.path.isdir(datasets_folder), f"Dataset folder {datasets_folder} does not exist"
        if not os.path.isdir(output_folder):
            logger.warning(f"Output folder {output_folder} does not exist, creating it")
            os.makedirs(output_folder)
        assert os.path.isdir(output_folder), f"Output folder {output_folder} does not exist"

        self.datasets_folder = datasets_folder
        self.output_folder = output_folder
        self.discarded_txt = discarded_txt
        self.discarded_elements = self._get_discarded_elements(discarded_txt=discarded_txt)

        self.check_consistency()

        self.parser = DetectionImprovement()
        self.inspected_images = 0


    def _get_discarded_elements(self, discarded_txt: str) -> set:
        """
        Returns a set of the discarded elements. These are saved in a txt file (discarded.txt) where each line is
        a .jpg file that was discarded during the parsing process.
        :param discarded_txt: path to the discarded.txt file
        :return: set of discarded elements
        """
        discarded_elements = set()
        if os.path.isfile(discarded_txt):
            with open(discarded_txt, "r") as f:
                for line in f.readlines():
                    discarded_elements.add(line.strip())
        return discarded_elements

    def _write_discarded_elements(self):
        """
        Writes the discarded elements to a txt file (discarded.txt) where each line is a .jpg file that was discarded
        during the parsing process. In order to avoid data loss, they are written in a discarded.temp.txt file and
        then renamed to discarded.txt at the end of the process, replacing the old discarded.txt file.
        """
        temp_file = self.discarded_txt.replace('.txt', '.temp.txt')
        with open(temp_file, 'w') as f:
            for element in self.discarded_elements:
                f.write(f"{element}\n")
        os.remove(self.discarded_txt)
        os.rename(temp_file, self.discarded_txt)


    def parse_dataset(self):
        """
        Parses all datasets in the datasets folder. The datasets folder must have the following structure:
        datasets_folder:
            - <dataset_name>:
                - train:
                    images:
                    labels:
                - validation:
                ...
        """
        for dataset_name in os.listdir(self.datasets_folder):
            dataset_path = os.path.join(self.datasets_folder, dataset_name)
            self.__parse_sub_dataset(dataset_path)
        self._write_discarded_elements()

    def __parse_sub_dataset(self, dataset_path: str):
        """
        Parses a dataset. The dataset must have the following structure:
        dataset_path:
            - <set_name>:
                - images:
                - labels:
        :param dataset_path: path to the dataset
        """
        assert os.path.isdir(dataset_path), f"Dataset path {dataset_path} does not exist"
        for set_name in os.listdir(dataset_path):
            assert set_name in ('train', 'valid', 'validation', 'test'), f"Set name {set_name} is not valid"
            images_path = os.path.join(dataset_path, set_name, 'images')
            labels_path = os.path.join(dataset_path, set_name, 'labels')
            assert os.path.isdir(images_path), f"Images folder {images_path} does not exist"
            assert os.path.isdir(labels_path), f"Labels folder {labels_path} does not exist"
            for image_name in os.listdir(images_path):
                image_name_without_extension = os.path.splitext(image_name)[0]
                image_path = os.path.join(images_path, image_name)
                label_path = os.path.join(labels_path, f"{image_name_without_extension}.txt")
                if not self.__image_is_already_processed(entry_name=image_name_without_extension, set_name=set_name):
                    self._process_image(image_path=image_path, label_path=label_path, set_name=set_name)
                    self.inspected_images += 1
                    if self.inspected_images % 100 == 0:
                        logger.info(f"Inspected {self.inspected_images} images")
                        self._write_discarded_elements()

    def __image_is_already_processed(self, entry_name: str, set_name:str) -> bool:
        """
        Checks if an image is already processed.
        :param entry_name: name of the image/label (no extension)
        :param set_name: name of the set
        :return: True if the image is already processed, False otherwise
        """
        if set_name not in os.listdir(self.output_folder):
            return False
        if entry_name in self.discarded_elements:
            return True
        visualization_path = os.path.join(self.output_folder, set_name, 'visualization')
        return os.path.isfile(os.path.join(visualization_path, f"{entry_name}.jpg"))

    def _process_image(self, image_path: str, label_path: str, set_name: str) -> bool:
        """
        Process an image and its label. Generates the output files and saves them in the output folder.
        Also updates the discarded_elements set.
        :param image_path: path to the image
        :param label_path: path to the label
        :return: True if the image was processed successfully, False otherwise
        """

        assert os.path.isfile(image_path), f"Image {image_path} does not exist"
        assert os.path.isfile(label_path), f"Label {label_path} does not exist"
        assert label_path.endswith('.txt'), f"Label {label_path} is not a .txt file"
        assert image_path.endswith('.jpg'), f"Image {image_path} is not a .jpg file"

        image_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
        label_name_without_extension = os.path.splitext(os.path.basename(label_path))[0]

        if image_name_without_extension in self.discarded_elements:
            logger.warning(f"Image {image_name_without_extension} was discarded in a previous run, skipping it")
            return False

        assert image_name_without_extension == label_name_without_extension, \
            f"Image {image_name_without_extension} and label {label_name_without_extension} do not match"

        # Read the image and label
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        try:
            is_bbox, polygons = self.read_label(label_path=label_path)
            polygons = list(polygons)
        except AssertionError as e:
            logger.warning(f"Image {image_name_without_extension} discarded because of an invalid label: {e}")
            self.discarded_elements.add(image_name_without_extension)
            return False

        to_discard, is_empty = False, len(polygons) == 0
        if is_empty:
            is_empty = self.parser.check_is_empty(image=image)
            polygons = [None]
        # Images with no polygons must save an empty file
        if not is_empty:
            reconstructed_polygons = []
            while len(polygons) > 0:
                polygon = polygons.pop(0)
                # If any polygon is None, discard the whole image
                new_polygon, are_more = self.parser.check_polygon(image=image, current_polygon=polygon,
                                                                  is_bbox=is_bbox and polygon is not None,
                                                                  remaining_polygons=len(polygons))
                if are_more:
                    polygons.append(None)
                if new_polygon is None:
                    to_discard = True
                    break
                new_polygon = sort_corners(corners=new_polygon)
                reconstructed_polygons.append(new_polygon)
            if to_discard:
                logger.warning(f"Image {image_name_without_extension} discarded because of an invalid polygon")
                self.discarded_elements.add(image_name_without_extension)
                if len(self.discarded_elements) % 2 == 0:
                    self._write_discarded_elements()
                return False
            polygons = np.array(reconstructed_polygons, dtype=np.float32)

        # Save the image and label
        self.write_output(image_path=image_path, polygons=polygons, entry_name=image_name_without_extension,
                          set_name=set_name)
        return True


    def read_label(self, label_path: str, only_keep_these_class_ids: None | tuple = None) -> tuple[
        bool, tuple[np.ndarray]]:
        """
        Reads a label file and returns a list of coordinates.
        The label file must be a .txt file where each line is in the following format:
        <class> <cx> <cy> <width> <height> (for bboxes) or <class> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>... (for polygons)
        always separated by spaces and always in normalized coordinates (0. to 1.).
        Polygons always have more than 4 coordinates (2 points), so if the label file has 5 coordinates or less, it is
        considered a bbox.
        :param label_path: path to the label file
        :param only_keep_these_class_ids: if not None, only the labels with these class ids will be kept.
        :return: a tuple with a bool and a tuple of numpy arrays of coordinates, each in the shape (n, 2) where n is the
        number of points of that polygon. Each point is a (x, y) coordinate. The bool indicates if the label is a bbox (True) or a polygon (False).
        """
        assert os.path.isfile(label_path), f"Label {label_path} does not exist"
        assert label_path.endswith('.txt'), f"Label {label_path} is not a .txt file"

        polygons = []
        is_bbox = None
        with open(label_path, 'r') as f:
            for line in f.readlines():
                values = line.strip().split(' ')
                class_id = int(values[0])
                coords = np.array(values[1:], dtype=float)
                polygon_coords = []
                if only_keep_these_class_ids is None or class_id in only_keep_these_class_ids:
                    if len(coords) == 4:
                        if is_bbox is not None: assert is_bbox, f"Label {label_path} has both bboxes and polygons"
                        # It's a bbox
                        is_bbox = True
                        x, y, w, h = coords
                        # Convert bbox to polygon points, clockwise starting from top-left
                        polygon_coords = [
                            [(x - w / 2), (y - h / 2)],  # top-left point
                            [(x + w / 2), (y - h / 2)],  # top-right point
                            [(x + w / 2), (y + h / 2)],  # bottom-right point
                            [(x - w / 2), (y + h / 2)]  # bottom-left point
                        ]
                    else:
                        if is_bbox is not None:
                            raise AssertionError(f"Label {label_path} has both bboxes and polygons")
                        # It's a polygon
                        is_bbox = False
                        polygon_coords = coords.reshape(-1, 2)
                polygons.append(np.array(polygon_coords))

        return is_bbox, tuple(polygons)

    def overlay_polygons_on_image(self, image_path: str, polygons: np.ndarray) -> np.ndarray:
        """
        Read the image from image_path and overlay the polygons.
        :param image_path: path to the image
        :param polygons: array of polygons
        :return: image with overlayed polygons in RGB format
        """
        assert os.path.isfile(image_path), f"Image {image_path} does not exist"
        assert len(polygons) == 0 or polygons.shape[1:] == (4, 2), f"Polygons must be in the format (n, 4, 2)"
        assert len(polygons) == 0 or (np.max(polygons) >= 0. and np.max(
            polygons) <= 1.), f"Polygons must be in normalized coordinates (0. to 1.)"

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        for polygon in polygons:
            # Escalate polygon coordinates to image size and overlay each polygon
            points = np.array(polygon * np.array([width, height]), np.int32).reshape((-1, 1, 2))
            image = cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        return image

    def write_output(self, image_path: str, polygons: np.ndarray, entry_name: str, set_name: str = 'validation'):
        """
        Save the output in the output folder. Image, label, and visualization are saved in output_forlder/set_name/images,
        output_folder/set_name/labels and output_folder/set_name/visualization respectively.
        """
        # Create set directories if they don't exist
        images_dir = os.path.join(self.output_folder, set_name, 'images')
        labels_dir = os.path.join(self.output_folder, set_name, 'labels')
        visualization_dir = os.path.join(self.output_folder, set_name, 'visualization')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)

        # Generate image with overlayed polygons
        overlay_image = self.overlay_polygons_on_image(image_path=image_path, polygons=polygons)

        # Save the image to the new location
        output_image_path = os.path.join(images_dir, f"{entry_name}.jpg")
        cv2.imwrite(output_image_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # Write the polygons to a txt file
        output_label_path = os.path.join(labels_dir, f"{entry_name}.txt")
        with open(output_label_path, 'w') as f:
            for polygon in polygons:
                polygon = polygon.flatten()  # Flatten the array to a single line
                class_id = 0  # QR is always class 0
                f.write(f"{class_id} " + " ".join(
                    map(str, polygon)) + "\n")  # Write the class_id and the polygon points to the file

        # Save the visualization to the new location
        output_visualization_path = os.path.join(visualization_dir, f"{entry_name}.jpg")
        cv2.imwrite(output_visualization_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # If there is a multiple of 100 images in the set, make a security copy of the set
        if len(os.listdir(images_dir)) % 100 == 0:
            shutil.copytree(self.output_folder, self.output_folder + '_backup_temp')
            if os.path.isdir(self.output_folder + '_backup'):
                shutil.rmtree(self.output_folder + '_backup')
            shutil.move(self.output_folder + '_backup_temp', self.output_folder + '_backup')

    def check_consistency(self):
        """
        Reads the image, labels, and visualization folder. If finds any image that is in any of these folders
        but not in any of the others, deletes it.
        Before deleting shows a message asking for confirmation.
        """
        set_names = ['train', 'validation', 'test']

        for set_name in set_names:
            images_dir = os.path.join(self.output_folder, set_name, 'images')
            labels_dir = os.path.join(self.output_folder, set_name, 'labels')
            visualization_dir = os.path.join(self.output_folder, set_name, 'visualization')

            # Check if the directories exist before processing
            if os.path.exists(images_dir) and os.path.exists(labels_dir) and os.path.exists(visualization_dir):

                # Extracting filenames without extensions for each directory
                image_files = {os.path.splitext(filename)[0] for filename in os.listdir(images_dir)}
                label_files = {os.path.splitext(filename)[0] for filename in os.listdir(labels_dir)}
                visualization_files = {os.path.splitext(filename)[0] for filename in os.listdir(visualization_dir)}

                # Find files that are not common to all three directories
                all_files = image_files | label_files | visualization_files
                inconsistent_files = [filename for filename in all_files if
                                      filename not in image_files or filename not in label_files or filename not in visualization_files]

                if inconsistent_files:
                    print(f"Inconsistent files found in {set_name}:")
                    for filename in inconsistent_files:
                        print(filename)

                    # Ask user for confirmation
                    choice = input("Do you want to delete these inconsistent files? (y/n): ").lower()
                    if choice.lower() in ('yes', 'y'):
                        for filename in inconsistent_files:
                            if filename in image_files:
                                os.remove(os.path.join(images_dir, filename + '.jpg'))
                            if filename in label_files:
                                os.remove(os.path.join(labels_dir, filename + '.txt'))
                            if filename in visualization_files:
                                os.remove(os.path.join(visualization_dir, filename + '.jpg'))

                        print(f"Deleted inconsistent files for {set_name}.")
                    else:
                        print("Skipped deleting for now.")
