import os
import shutil

from dataset_parser import DatasetParser
from detection_improvement import DetectionImprovement
import os


def main():
    dataset_parser = DatasetParser(datasets_folder="input_datasets", output_folder="parsed_dataset", discarded_txt=os.path.join("parsed_dataset", "discarded.txt"))
    dataset_parser.parse_dataset()

if __name__ == "__main__":
    main()
