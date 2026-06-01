import os
import argparse

import utils
from config import Config

# Modality placeholder expected by the MLLM trainer (mllm DEFAULT_IMAGE_TOKEN);
# the LiDAR features for each scene are injected in place of this token.
MODALITY_TOKEN = "<video>"


class PreprocessDataset:
    """Convert the merged dataset into the conversation format used for training.

    Each generated sample holds a single question/answer pair with its
    scene/sequence identifiers. Training expects records of the form
    ``{"id", "scene_id", "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}``,
    where ``scene_id`` is used to load the scene features and the modality token
    is prepended to the human turn.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.input_path = os.path.join(cfg.MERGED_DATASET_DIR, "merged_dataset.json")
        self.save_path = os.path.join(cfg.PREPROCESSED_DATASET_DIR, "preprocessed_dataset.json")

    def to_training_format(self, sample):
        question, answer = sample["conversations"][0], sample["conversations"][1]
        return {
            "id": sample["sequence_id"],
            "scene_id": sample["scene_id"],
            "conversations": [
                {"from": "human", "value": f"{MODALITY_TOKEN}\n{question['value']}"},
                {"from": "gpt", "value": answer["value"]},
            ],
        }

    def preprocess(self):
        samples = utils.load_json(self.input_path)
        preprocessed = [self.to_training_format(s) for s in samples]
        utils.save_json(preprocessed, self.save_path)
        print(f"Saved {len(preprocessed)} samples to {self.save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the merged dataset into the training conversation format")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_args(args)
    PreprocessDataset(cfg).preprocess()


if __name__ == "__main__":
    main()
