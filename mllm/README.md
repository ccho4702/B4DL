# Training

The B4DL model is trained in two stages on top of a frozen Vicuna-7B v1.5 LLM:
`s1` (3D LiDAR understanding) and `s2` (4D LiDAR understanding, trained on the
B4DL simple + complex tasks combined).

> `stage2.json` / `stage3.json` are the released dataset's simple / complex task
> splits (not training-stage numbers); concatenate them for the 4D stage.

Before running, download the [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main)
weights and place them under `./base_model/`.

```shell
bash run_stages.sh \
     --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./lidarclip/stage1_features \
     --s2_data ./b4dl_dataset/b4dl_4d_train.json \
     --s2_feat ./lidarclip/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
```

See [docs/train.md](docs/train.md) for the full procedure and
[docs/data.md](docs/data.md) for the data format.
