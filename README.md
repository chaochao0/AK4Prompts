# AK4Prompts

This repository is the implementation of "AK4Prompts: Automated Keywords-Ranking for Prompts in Text-To-Image Models"

### Installation

Create a conda environment with the following command:

```bash
conda create -n ak4prompts python=3.10
conda activate ak4prompts
pip install -r requirements.txt
```

### Inference

Run the `demo.ipynb`

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import make_image_grid
from AK4Prompts import AK4Prompts
from AK4Prompts_pipeline import AK4PromptsPipeline

device = 'cuda:0'
pipeline= AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16).to(device)

ak4prompts = AK4Prompts().to(device)
ak4prompts_path = "/checkpoints/SD1.5_LCMLoRA_S4_aes1_clip2.25_hps2.25/pytorch_model.bin"
ak4prompts.load_state_dict(torch.load(ak4prompts_path))
ak4prompts_pipeline = AK4PromptsPipeline(pipeline=pipeline,ak4prompts=ak4prompts,keywords_filename="keywords_list.txt")

prompt = "vase of mixed flowers"

scores_weights = {'aesthetic':1,'clip':5,'hps':3}
prompt_with_keywords = ak4prompts_pipeline.keywords_ranking(prompt=prompt,scores_weights=scores_weights,topk=10)

image = pipeline(prompt=prompt_with_keywords, num_inference_steps=4, guidance_scale=1.0).images[0]
image
```

The `demo.ipynb` script also demonstrates the evolution of images generated by setting different degrees of specific preferences.

<!-- ![hps](images\hps.png "hps") -->
<!-- ![hps](https://github.com/chaochao0/AK4Prompts/tree/master/images/hps.png "hps") -->
<p align="center">
  <img src="https://github.com/chaochao0/AK4Prompts/blob/master/images/hps.png" alt="hps" width="100%">
</p>

### Evaluation

Evaluates the model checkpoint and baselines. Evaluation includes calculating the rewards and storing the images to local.

```bash
python evaluation.py --topk 10 --aes 1.0 --clip 5.0 --hps 3.0 --guidance_scale 1.0
```

`topk`: choose topk keywords

`aes`,`clip`,`hps`: weights for customized keywords-ranking

`guidance_scale`: classifier-free guidance scale for TIS model

### Training Code

Accelerate will automatically handle multi-GPU setting. The code can work on a single GPU, as we automatically handle gradient accumulation as per the available GPUs in the CUDA_VISIBLE_DEVICES environment variable. If you are using a GPU with a small or big RAM, please edit the per_gpu_capacity variable accordingly. 
```bash
accelerate launch train.py --config config/train_config.py:aesthetic_clip_hps
```