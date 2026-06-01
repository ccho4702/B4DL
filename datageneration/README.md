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
python3 tools/create_metadata.py --nuscenes_root /path/to/nuscenes
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

## Stage-1 (3D) data
Stage-1 uses the external [LiDAR-LLM-Nu-Caption](https://huggingface.co/datasets/Senqiao/LiDAR-LLM-Nu-Caption)
dataset (B4DL does not redistribute it). Convert it to the training format with:

```bash
python3 tools/build_stage1_from_lidarllm.py --output ./stage1_train.json
```

This downloads the dataset, converts each QA pair to the conversation format, and
keeps only the samples whose scene is in the B4DL training split (the same 699
scenes used by stage2/stage3 — no test leakage), using a bundled mapping table
under `tools/assets/` (no nuScenes required). Pass `--input` to use a local copy.

## Auxiliary / reference scripts (`tools/`)
- `tools/create_metadata.py` — metadata generator (metadata is already released; reference only).
- `tools/generate_stage1_caption.py` — alternative per-frame captioning utility (reference only;
  the paper's Stage-1 uses LiDAR-LLM-Nu-Caption above, not this).
