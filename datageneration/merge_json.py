import os
import argparse

import utils
from config import Config


class MergeDataset:
    """Merge per-task generated JSON files into a single dataset file.

    The 4D LiDAR understanding stage is trained on the simple and complex
    tasks combined, so the per-task outputs of ``generate_dataset.py`` are
    merged into one file before preprocessing.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.input_dir = cfg.GENERATED_DATASET_DIR
        self.save_path = os.path.join(cfg.MERGED_DATASET_DIR, "merged_dataset.json")

    def merge(self):
        json_paths = []
        for task in self.cfg.TASKS:
            task_dir = os.path.join(self.input_dir, task)
            if os.path.isdir(task_dir):
                json_paths.extend(utils.get_json_filenames(task_dir))

        print(f"Found {len(json_paths)} JSON files to merge.")
        merged_data = utils.merge_json_files(json_paths)
        utils.save_json(merged_data, self.save_path)
        print(f"Saved {len(merged_data)} samples to {self.save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge per-task generated datasets into one file")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    MergeDataset(cfg).merge()


if __name__ == "__main__":
    main()
