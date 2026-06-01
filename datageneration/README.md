# Data Generation Pipeline

This module turns raw nuScenes data into the instruction-following QA dataset
used to train the 4D LiDAR understanding stage.

**Requirements**
- The [nuScenes](https://www.nuscenes.org/) dataset.
- An OpenAI API key: `export OPENAI_API_KEY=...` (passed via `--api_key`).
- Set your local nuScenes path with `--nuscenes_root` (no paths are hard-coded).

```bash
cd datageneration
pip install -r requirements.txt
```

## Pipeline

### 1. (Optional) Build metadata — `create_metadata.py`
Builds `sequence_metadata.json` / `scene_metadata.json` from nuScenes.
The metadata used in the paper is already released on
[Hugging Face](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL), so this
step is only needed to regenerate it from scratch.

```bash
python3 create_metadata.py --nuscenes_root /path/to/nuscenes
```

### 2. 4D LiDAR Context Extraction — `generate_description.py`
Generates front/back scene descriptions for each sequence from the multi-view images.

```bash
bash scripts/generate_description.sh
```

### 3. Context-to-QA Transformation — `generate_dataset.py`
Generates QA pairs for each task (`existence`, `binary`, `description`,
`temporal_understanding`, `comprehensive`), then merges and preprocesses them
into the training conversation format.

```bash
bash scripts/generate_dataset.sh
```

This runs, for every task:

```bash
python3 generate_dataset.py --api_key "$OPENAI_API_KEY" --dataroot ./data --task <task>
python3 merge_json.py --dataroot ./data         # merge per-task outputs
python3 preprocess_dataset.py --dataroot ./data # -> {"id", "conversations"} format
```

## Auxiliary / reference scripts
- `create_metadata.py` — metadata generator (metadata is already released; reference only).
- `generate_stage1_caption.py` — per-frame captioning for the 3D (Stage-1) data. The paper
  uses the external LiDAR-LLM-Nu-Caption dataset for Stage-1; this is provided only as a reference.
