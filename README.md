# path-open-data
This repository will deal with the code for assessment of the Open-Ended Pathology VQA

For Qwen-2.5VL, install transformers from scratch

1. pip install git+https://github.com/huggingface/transformers accelerate -> This technique of installing transformers does not work well for DeepSeekVL2, need to install transformers==4.38.2 for this)
2. pip install qwen-vl-utils
3. Do NOT Install -> pip install flash-attn (this doesn't work because GLIB_C is not up to date on the server. Might need to setup a local fix if we need to make this work)

For DeepSeekVL2, we need to do the following:

1. Clone the Repo: https://github.com/deepseek-ai/DeepSeek-VL2
2. pip install transformers==4.38.2 (this requirement is given in the requirements.txt. Updating transformers doesn't work correctly)
2. pip install attrdict
3. pip install timm
4. pip install xformers
4. Remove all the LLamA_FlashAttention_2 based lines from deepseek_vl2/models/modeling_deepseek.py (https://github.com/deepseek-ai/DeepSeek-VL2/issues/87)
