# Evaluation

Evaluate a trained B4DL model on the B4DL test sets. Each test file is a JSON of
test samples with pre-extracted LiDAR features; the model's answers are written to
a log file and then scored. Make sure you can run inference first (see
[offline_demo.md](offline_demo.md)).

- Use the B4DL test splits (one file per task) from the
  [B4DL dataset](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL) as `--data_path`.
- Provide the pre-extracted LiDAR features via `--feat_folder` (one `.npy` per id).

Run inference and log the responses:

```bash
python vtimellm/eval/eval.py \
     --data_path /path/to/test/<task>.json \
     --feat_folder /path/to/features \
     --log_path /path/to/log \
     --model_base /path/to/vicuna-7b-v1.5
```

Then compute the metrics from the log (captioning metrics need `pycocoevalcap` and Java):

```bash
python vtimellm/eval/metric.py \
     --data_path /path/to/test/<task>.json \
     --log_path /path/to/log
```
