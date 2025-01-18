# ImplicitPRM


## Train

Env setup:

```bash
conda create -n ImplicitPRM python=3.11 -y
conda activate ImplicitPRM

cd train
pip install -e .
pip install flash_attn
```

To train Implicit PRM, you can run following commands:

```bash
conda activate ImplicitPRM
cd train/tasks
bash run_ce.sh      # 8*A800-80G, 5hrs; 4*A100-40G OOM
bash run_dpo.sh
```
The above scripts will automatically download the dataset  `Windy0822/ultrainteract_math_rollout` from huggingface and transform it to the format of OpenRLHF pipeline, which will be saved at the path indicated by the `--dataset` argument.

Other argument settings are similar to the OpenRLHF package.

## Eval

Env Setup:

```bash
conda create -n ImplicitPRM_eval python=3.10 -y
conda activate ImplicitPRM_eval

cd eval
pip install -r requirements.txt
```

Extract precomputed answers:

```bash
tar -xzvf eval/testset/math-llama3.1-8b-inst-64.tar.gz -C eval/testset/
tar -xzvf eval/testset/math-Meta-Llama-3.1-70B-Instruct-64.tar.gz -C eval/testset/
tar -xzvf eval/testset/math-Mistral-7B-Instruct-v0.2-64.tar.gz -C eval/testset/
```

### ImplicitPRM
To evaluate ImplicitPRM, you can run the following command:

```bash
conda activate ImplicitPRM_eval
cd eval
python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py \
         --load /data/lyx/CODES/ImplicitPRM/saves/openrlhf-checkpoints-final/ce \
         --ref-load meta-llama/Meta-Llama-3.1-8B-Instruct \
         --type implicit_prm
```

where `--load`, `--ref-load` indicates the path of your trained model and reference model.

### Baseline: NTP-PRM

To evaluate the NTP-PRM baseline, you can run the following command:

```bash
conda activate ImplicitPRM_eval
cd eval
python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py \
        --load /home/openrlhf-checkpoints-final/sft-prm \
        --tokenizer-path /home/openrlhf-checkpoints-final/sft-prm \
        --type baseline-ntp \
        --begin-of-action-token <|reserved_special_token_0|> \
        --prm-token <|reserved_special_token_0|> \
```
