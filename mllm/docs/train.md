# Training

The B4DL model is trained on top of a frozen Vicuna-7B v1.5 LLM in two stages:

1. **3D LiDAR understanding** (`stage1`) — trains the LiDAR projection layer so the
   LLM can interpret static point-cloud features. Uses the LiDAR-LLM-Nu-Caption dataset.
2. **4D LiDAR understanding** (`stage2`) — trains a LoRA adapter on the B4DL training
   set (the simple and complex tasks combined) for spatio-temporal reasoning.

### Prerequisites
- **LLM weights**: Vicuna-7B v1.5 under `./base_model/` ([download](https://huggingface.co/lmsys/vicuna-7b-v1.5)).
- **LiDAR features**: pre-extracted with `encoders/lidarclip/extract_pc_features.py`,
  saved as one `{scene_id}.npy` per scene (see [data.md](data.md)).
- **Training data**: conversation-format JSON (see [data.md](data.md)), produced by the
  `datageneration` pipeline or downloaded from the
  [B4DL dataset](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL).

### Run

```shell
bash run_stages.sh \
    --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
    --s1_feat ./lidarclip/stage1_features \
    --s2_data ./b4dl_dataset/stage2.json \
    --s2_feat ./lidarclip/stage2_features \
    --model_name_or_path ./base_model/vicuna-v1-5-7b
```

`--s2_data` should point to the B4DL 4D training set (the simple-task and
complex-task files combined). You can also run each stage directly:

```shell
bash scripts/stage1.sh
bash scripts/stage2.sh
```
