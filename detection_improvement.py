from __future__ import annotations


import cv2
from qrdet import PADDED_QUAD_XYN, POLYGON_XYN, QUAD_XYN
from scipy.spatial import ConvexHull
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from loguru import logger

from utils import sort_corners, select_polygon_corners
from qreader import QReader

class DetectionImprovement:

    def __init__(self):
        self.qreader = QReader()

    @staticmethod
    def get_polygon_from_opencv(image, current_polygon: np.ndarray):
        # Calculate center of the current_polygon
        if current_polygon is not None:
            center_current_polygon = np.mean(current_polygon, axis=0)
        else:
            center_current_polygon = np.array((0.5, 0.5), dtype=np.float32)

        # Initialize min_distance and closest_bbox
        min_distance, closest_bbox = float('inf'), None

        # Try to detect with OpenCV
        qrDecoder = cv2.QRCodeDetector()
        _, bbox = qrDecoder.detect(img=image)
        if bbox is not None:
            for box in bbox:
                # OpenCV gives bad detections sometimes. So check that no point is duplicated
                is_bbox_correct = box.shape[0] == np.unique(box, axis=0).shape[0]
                if not is_bbox_correct:
                    continue
                # Normalize it
                box = box / (image.shape[1], image.shape[0])
                center_box = np.mean(box, axis=0)
                # Calculate Euclidean distance
                distance = np.linalg.norm(center_box - center_current_polygon)
                if distance < min_distance:
                    min_distance, closest_bbox = distance, box

        if closest_bbox is not None:
            assert closest_bbox.shape == (4, 2), "The bbox must have 4 corners"
            closest_bbox = np.clip(closest_bbox, a_min=0., a_max=1.)
            return closest_bbox

        # If OpenCV detection fails, fall back to Pyzbar
        barcodes = decode(image, symbols=[ZBarSymbol.QRCODE])
        if not barcodes:
            return None

        # Calculate center and find closest polygon
        min_distance, closest_bbox = float('inf'), None
        for barcode in barcodes:
            # Pyzbar returns the polygon points in a slightly different format, adjust it to match OpenCV's format
            bbox = np.array([barcode.polygon], np.float32)
            # Normalize it
            bbox = bbox / (image.shape[1], image.shape[0])
            center_bbox = np.mean(bbox, axis=0)

            # Calculate Euclidean distance
            distance = np.linalg.norm(center_bbox - center_current_polygon)
            if distance < min_distance:
                min_distance, closest_bbox = distance, bbox
        assert len(closest_bbox) == 1, "There should be only one QR code"
        closest_bbox = closest_bbox[0]
        closest_bbox = np.clip(closest_bbox, a_min=0., a_max=1.)
        assert closest_bbox.shape == (4, 2), "The bbox must have 4 corners"
        return closest_bbox

    def check_polygon(self, image: np.ndarray, current_polygon: np.ndarray, is_bbox: bool,
                      remaining_polygons: int = 0) -> None | np.ndarray:
        """
        Checks if the polygon is valid. If it is not, it tries to improve it.
        For this process, first, build the 4 corners romboid and show it to the user, to make it discard or
        accept it. If it is accepted, return the corners. If not, try to get the corners from the QR code with OpenCV,
        and, if it is possible, show it to the user again. If it is accepted, return the corners. If not, return None.
        :param image: np.ndarray. Image to be processed
        :param current_polygon: np.ndarray. Polygon to be checked, in the format [[x1, y1], [x2, y2], ...]
        :param is_bbox: bool. Indicates if the polygon is a bbox (True) or a polygon (False). If it is a bbox, it won't
        try to get the corners from polygon and will go directly to OpenCV
        :return: None|np.ndarray. If the polygon is valid, return the corners. If not, return None.
        """
        last_correct_corners = None
        corners = current_polygon
        if current_polygon is not None:
            last_correct_corners = corners
            accepted, are_more = self.ask_user(image=image, corners=corners, title="From Label",
                                                  remaining_polygons=remaining_polygons)
            if accepted:
                return corners, are_more
        else:
            corners = None

        # First, try OpenCV. Doesn't always work, but when it does, it is very accurate
        corners = self.get_polygon_from_opencv(image, current_polygon=corners)

        if corners is not None:
            last_correct_corners = corners
            accepted, are_more = self.ask_user(image=image, corners=corners, title="From OpenCV",
                                               remaining_polygons=remaining_polygons)
            if accepted:
                return corners, are_more

        # Then try with QRDet
        decodes, dets = self.qreader.detect_and_decode(image=image, return_detections=True)
        if len(dets) > 0:
            for i, (det, decode) in enumerate(zip(dets, decodes)):
                print("Decoding:", decode)
                for corners, title in zip((det[POLYGON_XYN], det[QUAD_XYN], det[PADDED_QUAD_XYN]),
                                          ("QRDet (Full Polygon)", "QRDet (Quadrilateral)", "QRDet (Expanded Quadrilateral)")):
                    if decode is None:
                        title += " [Not decoded]"
                    if len(dets) > 1:
                        title += f" [Checking {i+1}/{len(dets)}]"
                    accepted, are_more = self.ask_user(image=image, corners=corners, title=title,
                                                       remaining_polygons=remaining_polygons)
                    if accepted:
                        return corners, are_more

        # If nothing worked, ask the user to select the corners
        corners = select_polygon_corners(image=image, prev_corners=last_correct_corners)
        if corners is not None:
            accepted, are_more = self.ask_user(image=image, corners=corners, title="Your selection",
                                               remaining_polygons=remaining_polygons)
            if accepted:
                return corners, are_more

        return None, False


    def check_is_empty(self, image: np.ndarray)-> bool:
        """
        Shows the image, that does not contain any polygon and asks to the user if it is empty or not.
        :param image: np.ndarray. Image to be shown
        :return: bool. True if the image is empty, False otherwise
        """
        cv2.imshow("Is empty?", image)
        key = cv2.waitKey(0)
        if key not in (ord('y'), ord('n')):
            raise ValueError("The key must be 'y' or 'n'")
        empty = key == ord('y')
        # Destroy all windows
        cv2.destroyAllWindows()
        return empty

    @staticmethod
    def _build_romboid_from_polygon(current_polygon: np.ndarray) -> np.ndarray:
        """
        Builds a 4 corners romboid from the polygon.
        :param current_polygon: np.ndarray. Polygon to be checked, in the format [[x1, y1], [x2, y2], ...]
        :return: np.ndarray. Corners of the romboid
        """

        assert current_polygon.shape[0] >= 4, "The polygon must have at least 4 points"

        # Remove duplicate points if exist
        current_polygon = np.unique(current_polygon, axis=0)

        # Calculate the Convex Hull of the points
        hull = ConvexHull(current_polygon)

        # Get the points forming the Convex Hull
        hull_points = current_polygon[hull.vertices, :]

        # Calculate centroid of the hull
        centroid = np.mean(hull_points, axis=0)

        # Calculate the angles and distances from centroid to the hull points
        angles = np.arctan2(hull_points[:, 1] - centroid[1], hull_points[:, 0] - centroid[0])
        distances = np.sum((hull_points - centroid) ** 2, axis=1)

        # Combine angles and distances into a 2D array and sort by both columns
        points = np.column_stack((angles, distances))
        sorted_points = points[np.lexsort(points.T[::-1])]

        # Get indices of the 4 corners (every n/4-th point) and sort them
        corner_indices = np.sort([i * len(sorted_points) // 4 for i in range(4)])

        # Select the corners from the hull points
        corners = hull_points[corner_indices, :]

        assert corners.shape == (4, 2), f"The corners must have 4 points. Current shape: {corners.shape}"

        return corners

    @staticmethod
    def ask_user(image: np.ndarray, corners: np.ndarray, title: str = "", remaining_polygons: int = 0) -> bool:
        """
        Asks the user if the polygon is valid by showing the corners on the image.
        Awaits a key press to determine the user's response.

        :param image: np.ndarray. Image to be displayed
        :param corners: np.ndarray. Corners of the polygon to be displayed
        :param title: str. Title of the image window
        :param remaining_polygons: int. Number of remaining polygons to be checked (just for the title)
        :return: bool. True if the polygon is accepted by the user, False otherwise
        """

        if remaining_polygons > 0:
            title += f" ({remaining_polygons} remaining)"
        if np.min(corners) < 0. or np.max(corners) > 1.:
            logger.warning("Don't wanna save a polygon with corners outside the image")
            return False, False
        # Sort and scale corners
        #corners = sort_corners(corners=corners)
        corners = (corners * (image.shape[1], image.shape[0])).astype(np.int32)

        # Plot corners on the image
        image_with_polygon = cv2.polylines(image.copy(), [corners], True, (255, 0, 0), thickness=2)

        # Display the image
        cv2.imshow(title, cv2.cvtColor(image_with_polygon, cv2.COLOR_RGB2BGR))
        print("Press 'Y' to accept or 'N' to reject on the displayed image window. 'M' to accept and continue.")

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        are_more = key in (ord('m'), ord('M'))
        accepted = key in (ord('y'), ord('Y')) or are_more
        return accepted, are_more


    @staticmethod
    def plot(image, data):
        for d in data:
            coords = np.array(d[1:]).reshape(-1, 2) * np.array([image.shape[1], image.shape[0]])
            image = cv2.polylines(image, [np.int32(coords)], True, (255, 0, 0), thickness=2)
        cv2.imshow('Image', image)
        cv2.waitKey(1)