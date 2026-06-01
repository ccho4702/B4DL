from dataclasses import dataclass

@dataclass
class Config:
    DESCRIPTION_GPT_MODEL = "gpt-4o" # "gpt-4o" / "gpt-4o-mini" / "o1-preview"
    GENERATE_GPT_MODEL = "gpt-4o"
    STAGE1_GPT_MODEL = "gpt-4o"
    STAGE2_GPT_MODEL = "gpt-4o"

    MAX_TOKENS = 2000

    API_KEY = ""

    CAMERA_VIEWS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    LIDAR_VIEW = "LIDAR_TOP"

    TASK = ""
    TASKS = ["existence", "binary", "time_grounding", "description", "temporal_understanding", "comprehensive"]
    SIMPLE_TASKS=["existence", "binary", "time_grounding"]
    COMPLEX_TASKS=["description", "temporal_understanding", "comprehensive"]  

    FRAME_INTERVAL = 2
    LOAD_N_FRAMES = 5
    DYNAMIC_FRAME_LEN = [3,4,5,6,7,8,9,10]
    GENERATE_N_SETS = 6

    START_INDEX = 0
    END_INDEX = 1000
    SAVE_TERM = 10

    NUSCENES_ROOT = "./data/nuscenes"  # override with --nuscenes_root
    NUSCENES_VERSION = "v1.0-trainval"  # or "v1.0-mini"

    DATAROOT = "./data"

    METADATA_DIR = DATAROOT + "/metadata"
    SEQUENCE_METADATA_PATH = METADATA_DIR + "/sequence_metadata.json"
    SCENE_METADATA_PATH = METADATA_DIR + "/scene_metadata.json"

    # Stage-1 (3D LiDAR understanding) per-frame caption data (auxiliary / reference)
    STAGE1_FRAME_PATH = METADATA_DIR + "/stage1_frames.json"
    STAGE1_CAPTION_PATH = METADATA_DIR + "/stage1_captions.json"

    # Data-generation pipeline outputs
    GENERATED_DESCRIPTION_DIR = DATAROOT + "/generated_description"
    GENERATED_DATASET_DIR = DATAROOT + "/generated_dataset"
    MERGED_DATASET_DIR = DATAROOT + "/merged_dataset"
    PREPROCESSED_DATASET_DIR = DATAROOT + "/preprocessed_dataset"

    @classmethod
    def from_args(cls, args):
        """Build a config from parsed CLI args, then resolve derived paths."""
        cfg = cls()
        for key, value in vars(args).items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg.resolve_paths()

    def resolve_paths(self):
        """Recompute DATAROOT-derived paths after ``DATAROOT`` is overridden."""
        self.METADATA_DIR = self.DATAROOT + "/metadata"
        self.SEQUENCE_METADATA_PATH = self.METADATA_DIR + "/sequence_metadata.json"
        self.SCENE_METADATA_PATH = self.METADATA_DIR + "/scene_metadata.json"
        self.STAGE1_FRAME_PATH = self.METADATA_DIR + "/stage1_frames.json"
        self.STAGE1_CAPTION_PATH = self.METADATA_DIR + "/stage1_captions.json"
        self.GENERATED_DESCRIPTION_DIR = self.DATAROOT + "/generated_description"
        self.GENERATED_DATASET_DIR = self.DATAROOT + "/generated_dataset"
        self.MERGED_DATASET_DIR = self.DATAROOT + "/merged_dataset"
        self.PREPROCESSED_DATASET_DIR = self.DATAROOT + "/preprocessed_dataset"
        return self

