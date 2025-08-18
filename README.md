# path-open-data
This repository will deal with the code for assessment of the Open-Ended Pathology VQA

For Qwen-2.5VL, install transformers from scratch

1. pip install git+https://github.com/huggingface/transformers accelerate -> This technique of installing transformers does not work well for DeepSeekVL2, need to install transformers==4.38.2 for this
2. pip install qwen-vl-utils
3. Do NOT Install -> pip install flash-attn (this doesn't work because GLIB_C is not up to date on the server. Might need to setup a local fix if we need to make this work)

For DeepSeekVL2, we need to do the following:

1. Clone the Repo: https://github.com/deepseek-ai/DeepSeek-VL2
2. pip install transformers==4.38.2 (this requirement is given in the requirements.txt. Updating transformers doesn't work correctly)
2. pip install attrdict
3. pip install timm
4. pip install xformers
4. Remove all the LLamA_FlashAttention_2 based lines from deepseek_vl2/models/modeling_deepseek.py (https://github.com/deepseek-ai/DeepSeek-VL2/issues/87)

For MiniCPMV2.6, we need to do the following
1. pip install git+https://github.com/huggingface/transformers accelerate
2. Install Flash Attention (For Cuda12.8 and Torch2.7.0): (https://github.com/Dao-AILab/flash-attention/issues/1644#issuecomment-2899396361)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

For Intern-VL3.0, we need to do the following
1. We can see the code to run the models on multile GPU using the link: https://huggingface.co/OpenGVLab/InternVL3-8B
2. pip install git+https://github.com/huggingface/transformers accelerate
3. Install Flash Attention (For Cuda12.8 and Torch2.7.0): (https://github.com/Dao-AILab/flash-attention/issues/1644#issuecomment-2899396361)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

For Gemma3-4b-it/MedGemma-4b-it, we need to do the following
1. pip install git+https://github.com/huggingface/transformers accelerate

For LLaVA-1.6-7B
1. git clone git@github.com:haotian-liu/LLaVA.git
2. pip install sentencepiece
3. pip install protobuf
4. pip install transformers==4.31.0
5. pip install shortuuid

For LLaVA-MED-Mistral-7B
1. git clone git@github.com:microsoft/LLaVA-Med.git
2. pip install sentencepiece
3. pip install protobuf
4. pip install transformers==4.36.2
5. pip install shortuuid

For Quilt-LLaVA-Vicuna-7B
1. git clone git@github.com:aldraus/quilt-llava.git
2. pip install sentencepiece
3. pip install protobuf
4. pip install transformers==4.31.0
5. pip install shortuuid

For LLaVA-Tri (MedTrinity-25M)
1. git clone git@github.com:UCSC-VLAA/MedTrinity-25M.git
2. pip install sentencepiece
3. pip install protobuf
4. pip install transformers==4.37.2
5. pip install shortuuid
6. pip install git+https://github.com/bfshi/scaling_on_scales.git

For Data Augmentation

1. git clone git@github.com:DIAGNijmegen/pathology-he-autoaugmetation.git or git clone git@github.com:DIAGNijmegen/pathology-he-auto-augment.git
2. pip install scikit-image
3. pip install albumentations==1.1.0
4. Change the following lines in pathology-he-autoaugmetation/randaugment/train/randaugement_new_ranges.py
_REPLACE = 225 -> This will replace the background to a semi-white color to match the slides background
available_ops = ['Scaling','TranslateX', 'TranslateY','ShearX', 'ShearY','Brightness', 'Sharpness','Color', 'Contrast','Rotate','Equalize', 'HsvH','HsvS','HsvV','HedH','HedE','HedD'] -> We are not doing Elastic, Gaussian Blur or Gaussian Noise operation since that tends to disturb the data to an extent which is not identifiable

