# B4DL
Official PyTorch implementation of the paper "B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal
Understanding".

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-🤗-yellow)](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)
[![arXiv](https://img.shields.io/badge/arXiv-2508.05269-b31b1b.svg)](https://www.arxiv.org/abs/2508.05269)
[![Paper](https://img.shields.io/badge/Paper-arXiv-blue.svg)](https://arxiv.org/pdf/2508.05269)


<img src="assets/main_figure.jpg">

---
## Overview

B4DL is a benchmark and an MLLM for 4D LiDAR spatio-temporal understanding. The
repo is organized into three parts:

| Folder | Purpose |
|--------|---------|
| [`datageneration/`](datageneration/README.md) | Build the B4DL QA dataset from nuScenes (+ tools for the Stage-1 data and metadata) |
| [`encoders/lidarclip/`](encoders/lidarclip/README.md) | Extract LiDAR features for the model |
| [`mllm/`](mllm/README.md) | Train and run the B4DL model (Vicuna-7B based) |

The model is trained in **two stages**: **Stage 1 (3D)** learns static point-cloud
features, **Stage 2 (4D)** learns spatio-temporal reasoning. So you prepare **two
datasets** (Stage-1 and Stage-2/4D) plus their **LiDAR features**, then train.

---
## Installation

```bash
git clone https://github.com/ccho4702/B4DL.git
cd B4DL
pip install -r datageneration/requirements.txt   # data generation
pip install -r mllm/requirements.txt             # training / inference
```

Download the base [Vicuna-7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) weights into
`mllm/base_model/`, and the CLIP `ViT-L/14` weights used by the encoder.

---
## Step 1 — Prepare data

### (a) Stage-2 / 4D data — the B4DL benchmark (simple + complex tasks)

**Option 1 — download** the released dataset from
[🤗 ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL):
`train/stage2.json` (simple tasks), `train/stage3.json` (complex tasks), `test/*.json`, and `metadata/`.

**Option 2 — regenerate** from nuScenes (needs an OpenAI key):
```bash
cd datageneration
export OPENAI_API_KEY=...
bash scripts/generate_description.sh   # 4D context extraction (set --nuscenes_root inside)
bash scripts/generate_dataset.sh       # per-task QA -> merge -> preprocess
# -> data/preprocessed_dataset/preprocessed_dataset.json   (4D training file, simple+complex combined)
```
See [`datageneration/README.md`](datageneration/README.md) for details.

### (b) Stage-1 / 3D data — external LiDAR-LLM-Nu-Caption

B4DL does not redistribute it; build it from the public dataset:
```bash
python3 datageneration/tools/build_stage1_from_lidarllm.py --output data/stage1_train.json
```
This downloads [LiDAR-LLM-Nu-Caption](https://huggingface.co/datasets/Senqiao/LiDAR-LLM-Nu-Caption),
converts it to the training format, and keeps only the samples whose scene is in the **same 699
training scenes as Stage-2** (no test leakage), using a bundled mapping table (no nuScenes required).

### (c) Extract LiDAR features

```bash
cd encoders/lidarclip
python3 extract_pc_features.py    # writes one .npy per scene (Stage-2) / per frame (Stage-1)
```
See [`encoders/lidarclip/README.md`](encoders/lidarclip/README.md).

---
## Step 2 — Train

```bash
cd mllm
bash run_stages.sh \
     --s1_data  ../data/stage1_train.json \
     --s1_feat  ./lidarclip/stage1_features \
     --s2_data  ../datageneration/data/preprocessed_dataset/preprocessed_dataset.json \
     --s2_feat  ./lidarclip/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
```
- `s1` = 3D understanding stage, `s2` = 4D understanding stage.
- The Stage-2 data is the simple-task and complex-task data **combined** into one file.

See [`mllm/docs/train.md`](mllm/docs/train.md) and [`mllm/docs/data.md`](mllm/docs/data.md).

---
## Step 3 — Inference / Evaluation

```bash
python3 mllm/vtimellm/inference.py --model_base ./mllm/base_model/vicuna-v1-5-7b   # demo
python3 mllm/vtimellm/eval/eval.py --data_path <test.json> --feat_folder <features> \
        --model_base ./mllm/base_model/vicuna-v1-5-7b --log_path <log>             # evaluation
```
See [`mllm/docs/`](mllm/docs/).

---
## Notes
- File names `stage2.json` / `stage3.json` are the dataset's **simple / complex task splits**,
  not training-stage numbers. Both feed the single 4D training stage (`s2`).
- Stage-1 (3D) uses the **external** LiDAR-LLM-Nu-Caption dataset (build it with the script above).

---

## Demo

<table align="center">
  <tr>
    <td colspan="2" align="center"><b>Example of Generated Dataset</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/dataset_lidar.gif" alt="Dataset (LiDAR)" width="350">
    </td>
    <td align="center">
      <img src="assets/dataset_cam.gif" alt="Dataset (Camera)" width="350">
    </td>
  </tr>
  
  <tr>
    <td colspan="2" align="center">
      <img src="assets/dataset_text.png" alt="Dataset (Text)" width="700">
    </td>
  </tr>
  
  <tr>
    <td colspan="2" align="center"><b>Example of Inference</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/inference_lidar.gif" alt="Inference (LiDAR)" width="350">
    </td>
    <td align="center">
      <img src="assets/inference_cam.gif" alt="Inference (Camera)" width="350">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="assets/inference_output_text.png" alt="Inference (Text)" width="700">
    </td>
  </tr>
</table>

---

## Acknowledgements
This work was partly supported by the Institute of Information &
Communications Technology Planning & Evaluation(IITP) grant
funded by the Korea government(MSIT) (No.RS-2024-00439020,
Developing Sustainable, Real-Time Generative AI for Multimodal
Interaction, SW Starlab) and partly supported by the Institute of
Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS2025-02283048, Developing the Next-Generation General AI with
Reliability, Ethics, and Adaptability)

If you use B4DL in your research, please cite:
```bibtex
@inproceedings{choi2025b4dl,
  title={B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding},
  author={Choi, Changho and Shin, Youngwoo and Han, Gyojin and Lee, Dong-Jae and Kim, Junmo},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={3399--3407},
  year={2025}
}
```

## License
<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a> 

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License</a>.
