"""Generate per-frame captions for the 3D LiDAR understanding stage.

Auxiliary / reference script. In the paper, the 3D (Stage-1) training uses the
external LiDAR-LLM-Nu-Caption dataset; this single-frame captioning utility is
provided only as a reference for how per-frame captions can be produced from
the nuScenes multi-view images.
"""

import os
import sys
import argparse

from tqdm import tqdm
from PIL import Image
from openai import OpenAI

# Auxiliary script under tools/; make the parent package modules importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from config import Config

SYSTEM_PROMPT = "You are a helpful assistant that creates captions for scenes consisting of point clouds."
USER_PROMPT = (
    "These images show the front and back view of the scene. The first image shows the "
    "front view, and the second image shows the back view. Please describe the scene "
    "briefly in 1 sentence."
)


class Stage1Caption:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataroot = cfg.NUSCENES_ROOT
        self.client = OpenAI(api_key=cfg.API_KEY)

    def prepare_frames(self):
        sequences = utils.load_json(self.cfg.SEQUENCE_METADATA_PATH)

        seen, frames = set(), []
        for sequence in sequences:
            for frame in sequence["frames"]:
                token = frame["TOKEN_LIDAR_TOP"]
                if token in seen:
                    continue
                seen.add(token)
                frames.append({
                    "frame_id": frame["frame_id"],
                    "PATH_LIDAR_TOP": frame["PATH_LIDAR_TOP"],
                    "PATH_CAM_FRONT": frame["PATH_CAM_FRONT"],
                    "PATH_CAM_BACK": frame["PATH_CAM_BACK"],
                })

        utils.save_json(frames, self.cfg.STAGE1_FRAME_PATH)
        return frames

    def generate_caption(self, front_path, back_path):
        images = [Image.open(front_path), Image.open(back_path)]
        images_base64 = utils.encode_images_to_base64(images)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [USER_PROMPT, *map(lambda x: {"image": x}, images_base64)]},
        ]
        result = self.client.chat.completions.create(
            model=self.cfg.STAGE1_GPT_MODEL,
            max_tokens=self.cfg.MAX_TOKENS,
            messages=messages,
        )
        return result.choices[0].message.content

    def generate(self):
        frames = self.prepare_frames()

        captioned = []
        for frame in tqdm(frames, desc="Stage-1 captioning"):
            front_path = os.path.join(self.dataroot, frame["PATH_CAM_FRONT"])
            back_path = os.path.join(self.dataroot, frame["PATH_CAM_BACK"])
            frame["caption"] = self.generate_caption(front_path, back_path)
            captioned.append(frame)

        utils.save_json(captioned, self.cfg.STAGE1_CAPTION_PATH)
        print(f"Saved {len(captioned)} captions to {self.cfg.STAGE1_CAPTION_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-frame captions for the 3D understanding stage")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key", dest="API_KEY")
    parser.add_argument("--nuscenes_root", type=str, default=Config.NUSCENES_ROOT, help="Path to nuScenes root directory", dest="NUSCENES_ROOT")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_args(args)
    Stage1Caption(cfg).generate()


if __name__ == "__main__":
    main()
