# Running B4DL inference offline
Please follow the instructions below to run B4DL inference on your local GPU machine.

Note: the demo requires approximately 18 GB of GPU memory.

### Set up the repository

```shell
conda create --name=b4dl python=3.10
conda activate b4dl

git clone https://github.com/ccho4702/B4DL.git
cd B4DL/mllm
pip install -r requirements.txt
```

### Download weights

* Place the B4DL checkpoints and the CLIP model into the 'checkpoints' directory.
* Download the base [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) weights.

### Run the inference code


```shell
python -m vtimellm.inference --model_base <path to the Vicuna v1.5 weights> 
```

Alternatively, you can also choose to conduct multi-turn conversations in [Jupyter Notebook](inference.ipynb). Similarly, you need to set 'args.model_base' to the path of Vicuna v1.5.

If you want to run the VTimeLLM-ChatGLM version, please refer to the code in [inference_for_glm.ipynb](inference_for_glm.ipynb).

### Run the gradio demo

We have provided an offline gradio demo as follows:

```shell
cd vtimellm
python demo_gradio.py --model_base <path to the Vicuna v1.5 weights> 
```
