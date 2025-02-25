# MSCTs

## Env Setup

```bash
conda create -n mscts python=3.10
conda activate mscts
pip install vllm==0.6.0 datasets transformers openai
```

## Vllm

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --host 0.0.0.0 \
    --port 10000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching
```

## Run

GSM8K

```bash
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
DATA_DIR_NAME="gsm8k-llama3-8b-new-mcts-8"
python run_with_earlystopping.py $MODEL_NAME $DATA_DIR_NAME
```