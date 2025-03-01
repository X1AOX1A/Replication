# LLaDA

- [Source](https://github.com/ML-GSAI/LLaDA)

## Env Setup

```bash
conda create -n llada python=3.10 -y
conda activate llada
pip install torch==2.6.0 transformers==4.38.2

# if run demo
pip install gradio
```

## Run

Inference:

```bash
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Run cli chat:

```bash
python chat.py
```

Run demo:

```bash
python app.py
```

![demo](imgs/example_gradio.gif)
