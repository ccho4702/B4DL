# Data Format

Training data is a JSON list of single-turn samples. Each sample carries the
`scene_id` used to load the LiDAR features and a `conversations` pair whose human
turn begins with the modality token `<video>` (replaced by the scene's LiDAR
features at train time):

```json
{
    "id": "101229297",
    "scene_id": "008395103",
    "conversations": [
        {"from": "human", "value": "<video>\nWas a pedestrian present in front of the ego vehicle between frame 30 and frame 38?"},
        {"from": "gpt", "value": "Yes."}
    ]
}
```

- **id**: sequence identifier.
- **scene_id**: used to load features from `{feat_folder}/{scene_id}.npy` (`<N, 768>` float16).
- **conversations**: `[human, gpt]`; the human turn starts with `<video>`.

### Where the data comes from
- Download the B4DL dataset from
  [Hugging Face](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL), or
- Generate it with the [`datageneration`](../../datageneration/README.md) pipeline,
  whose `preprocess_dataset.py` emits this format.

LiDAR features are pre-extracted per scene with
`encoders/lidarclip/extract_pc_features.py`.
