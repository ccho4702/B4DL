"""Build sequence/scene metadata from the raw nuScenes dataset.

Auxiliary / reference script. The metadata used in the paper is released on
the Hugging Face repository, so this is provided only to document how the
metadata was constructed (sequence sampling, identifiers, train/val split).
Regenerating will produce the same schema but new random identifiers.
"""

import argparse

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

import utils
from config import Config


class CreateMetadata:
    def __init__(self, nusc, cfg: Config):
        self.nusc = nusc
        self.cfg = cfg
        self.camera_views = cfg.CAMERA_VIEWS
        self.lidar_view = cfg.LIDAR_VIEW
        self.num_sets = cfg.GENERATE_N_SETS      # sequences sampled per scene
        self.frame_len = cfg.LOAD_N_FRAMES       # frames per sequence
        self.frame_interval = cfg.FRAME_INTERVAL # keyframe stride within a sequence
        self.scene_split = self._build_scene_split()

    def _build_scene_split(self):
        splits = create_splits_scenes()
        name_to_split = {}
        for split in ["train", "val"]:
            for scene_name in splits[split]:
                name_to_split[scene_name] = split
        return name_to_split

    def get_sample_tokens(self, scene):
        sample_tokens = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            sample_tokens.append(sample_token)
            sample_token = sample["next"]
        return sample_tokens

    def get_frame_data(self, sample):
        frame_data = {}
        for view in self.camera_views:
            token = sample["data"][view]
            frame_data["PATH_" + view] = self.nusc.get("sample_data", token)["filename"]
            frame_data["TOKEN_" + view] = token

        lidar_token = sample["data"][self.lidar_view]
        frame_data["PATH_" + self.lidar_view] = self.nusc.get("sample_data", lidar_token)["filename"]
        frame_data["TOKEN_" + self.lidar_view] = lidar_token
        return frame_data

    def generate_frame_indices(self, total_len):
        span = self.frame_len * self.frame_interval
        starts = np.linspace(0, max(total_len - span, 0), self.num_sets, dtype=int)
        return [list(range(start, start + span, self.frame_interval)) for start in starts]

    def build_sequence_metadata(self):
        sequences = []
        for scene in self.nusc.scene:
            split = self.scene_split.get(scene["name"])
            if split is None:
                continue

            sample_tokens = self.get_sample_tokens(scene)
            for indices in self.generate_frame_indices(len(sample_tokens)):
                frames = []
                for frame_id, idx in enumerate(indices):
                    sample = self.nusc.get("sample", sample_tokens[idx])
                    frame_data = self.get_frame_data(sample)
                    frame_data["frame_id"] = str(frame_id)
                    frame_data["sample_token"] = sample_tokens[idx]
                    frames.append(frame_data)

                sequences.append({
                    "scene_token": scene["token"],
                    "sequence_id": str(utils.generate_unique_id(1)[0]),
                    "split": split,
                    "frames": frames,
                    "indices": indices,
                })

        utils.save_json(sequences, self.cfg.SEQUENCE_METADATA_PATH)
        print(f"Saved {len(sequences)} sequences to {self.cfg.SEQUENCE_METADATA_PATH}")

    def build_scene_metadata(self):
        scenes = []
        for scene in self.nusc.scene:
            split = self.scene_split.get(scene["name"])
            if split is None:
                continue

            scenes.append({
                "scene_token": scene["token"],
                "scene_id": str(utils.generate_unique_id(1)[0]),
                "num_frames": scene["nbr_samples"],
                "split": split,
            })

        utils.save_json(scenes, self.cfg.SCENE_METADATA_PATH)
        print(f"Saved {len(scenes)} scenes to {self.cfg.SCENE_METADATA_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build sequence/scene metadata from nuScenes")
    parser.add_argument("--nuscenes_root", type=str, default=Config.NUSCENES_ROOT, help="Path to nuScenes root directory", dest="NUSCENES_ROOT")
    parser.add_argument("--nuscenes_version", type=str, default=Config.NUSCENES_VERSION, help="nuScenes version", dest="NUSCENES_VERSION")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_ROOT, verbose=True)
    builder = CreateMetadata(nusc, cfg)
    builder.build_sequence_metadata()
    builder.build_scene_metadata()


if __name__ == "__main__":
    main()
