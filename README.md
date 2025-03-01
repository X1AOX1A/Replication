# SMDM: Scaling up Masked Diffusion Models

[Source](https://github.com/ML-GSAI/SMDM/commits/main/)

## Env Setup

[Ref](https://github.com/ML-GSAI/SMDM/blob/main/CONDA.md)

```bash
conda create -n smdm python=3.9 -y
conda activate smdm

pip install torch torchvision torchaudio

# install flash-attention
pip uninstall ninja -y && pip install ninja -U

git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install packaging
python setup.py install

cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../../.. && rm -r flash-attention

# install xformers
pip install -U xformers

# install TinyLama requirements
git clone https://github.com/jzhang38/TinyLlama.git
cd TinyLlama
pip install -r requirements.txt tokenizers sentencepiece
cd .. && rm -r TinyLlama

# Install the dependencies needed for evaluation
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
pip install openai==0.28 fschat==0.2.34 anthropic
```

### Conditional generation
Please download the [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json) dataset and put the json file in `./data`.
Following [CLLM](https://github.com/hao-ai-lab/Consistency_LLM), we only used the first round of dialogue data.
```bash
cd data
wget -c https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json

cd ..
mkdir models
cd models
wget -c https://huggingface.co/nieshen/SMDM/resolve/main/mdm_safetensors/mdm-1028M-1600e18.safetensors
```

```sh
# Finetune MDMs
# For the unsupervised CFG, we set --cfg to 0.
# For the standard CFG, we set --cfg to 0.1
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    sft/finetune_mdm.py --model 1028 --pretrain_path models/mdm-1028M-1600e18.safetensors --cfg 0.
```