## Env Setup

```bash
conda create -n llm2 python=3.10 -y
conda activate llm2
pip install -r requirements.txt
```

## Train & Eval

Train PRM

```bash
conda activate llm2
ts -G 4 -L train_verifier bash scripts/train_verifier.sh
```

Eval LLM

```bash
bash scripts/evaluate_gsm8k.sh
# only 2% accuracy achieved on the GSM8K dataset
```