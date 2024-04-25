from PIL import Image
from tqdm import tqdm
import sys
from AK4Prompts import AK4Prompts
import torch
from diffusers import StableDiffusionPipeline,DDIMScheduler,LCMScheduler,AutoPipelineForText2Image,StableDiffusionXLPipeline
from keywords.keywords_table import KeywordsTable,Config
import numpy as np
import json,os
from datasets import load_dataset
import torchvision
from aesthetic_scorer import AestheticScorerDiff
from transformers import CLIPModel
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import pandas as pd

def hps_loss_fn(inference_dtype=None, device=None, grad_scale=10):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        None,#'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    tokenizer = get_tokenizer(model_name)

    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    #force download of model via score
    hpsv2.score([], "")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

    def loss_fn(im_pix, prompts):
        with torch.no_grad():
            im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
            x_var = normalize(x_var).to(im_pix.dtype)
            caption = tokenizer(prompts)
            caption = caption.to(device)
            outputs = model(x_var, caption)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits = image_features @ text_features.T
            scores = torch.diagonal(logits)
            loss = 1.0 - scores
        return  loss*grad_scale, scores*grad_scale

    return loss_fn

def clip_loss_fn(clip_model=None,
                 clip_target=1,
                 grad_scale=0,
                 device=None,
                 torch_dtype=None):
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])
    clip_model.requires_grad_(False)
    target_size = 224

    def loss_fn(im_pix_un, text_pooled_states):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(
            target_size, antialias=True)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

        image_embeds = clip_model.get_image_features(im_pix)

        text_embeds = clip_model.text_projection(text_pooled_states)

        rewards = torch.nn.functional.cosine_similarity(
            image_embeds, text_embeds)

        if clip_target is None:  # default maximization cosine_similarity
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - clip_target)
        return loss * grad_scale,rewards * grad_scale
    return loss_fn

def aesthetic_loss_fn(aesthetic_model=None,
                      aesthetic_target=None,
                      grad_scale=0,
                      device=None,
                      torch_dtype=None):
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])
    # scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer = aesthetic_model
    scorer.requires_grad_(False)
    target_size = 224

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(
            target_size, antialias=True)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None:  # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards   #rewards.shape:[batch]
    return loss_fn

def evaluate(latent, train_neg_prompt_embeds, prompts, keywords_embs, ak4prompts, pipeline, device, loss_fn_aesthetic,loss_fn_clip,loss_fn_hps,loss_fn_reward,labels,topk,rewards_scale,guidance_scale):
    prompt_inputs = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    )
    prompt_ids = prompt_inputs.input_ids.to(device)
    # pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(
    #     device)
    text_encoder_output = pipeline.text_encoder(prompt_ids)
    prompt_embeds = text_encoder_output[0]
    pooled_states = text_encoder_output[1]

    attention_mask = (prompt_inputs.attention_mask==0).to(device)
    ak4prompts.eval()
    with torch.no_grad():
        aesthetic_out_fc,clip_out_fc,hps_out_fc = ak4prompts(prompt_embeds.to(torch.float32), attention_mask, keywords_embs.to(torch.float32))

        final_score = aesthetic_out_fc*rewards_scale[0] + clip_out_fc*rewards_scale[1] + hps_out_fc*rewards_scale[2]
        k_values, k_indices = torch.topk(final_score, k=topk)  

        k_values_aesthetic = torch.gather(aesthetic_out_fc, dim=1, index=k_indices)
        k_values_clip = torch.gather(clip_out_fc, dim=1, index=k_indices)
        k_values_hps = torch.gather(hps_out_fc, dim=1, index=k_indices)

    append_keywords_choosed = [","+",".join([labels[idx] for idx in indices]) for indices in k_indices]

    prompts_with_keywords = [prompt + append_keywords for prompt, append_keywords in zip(prompts,append_keywords_choosed)]

    
    with torch.no_grad():
        images = pipeline(
            prompt=prompts_with_keywords, num_inference_steps=4, latent=latent, guidance_scale=guidance_scale, output_type = "latent"
        ).images
        ims =  pipeline.vae.decode(images / pipeline.vae.config.scaling_factor).sample

    _, aesthetic_rewards = loss_fn_aesthetic(ims)
    loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
    loss_hps, hps_rewards = loss_fn_hps(ims, prompts)
    
    aesthetic_rewards_expand = aesthetic_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_aesthetic.shape[1])  #[batch] --> [batch,k_values.shape[1]]
    clip_rewards_expand = clip_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_clip.shape[1])  #[batch] --> [batch,k_values.shape[1]]
    hps_rewards_expand = hps_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_hps.shape[1])  #[batch] --> [batch,k_values.shape[1]]

    loss_aesthetic_rewards = loss_fn_reward(k_values_aesthetic,aesthetic_rewards_expand)
    loss_clip_rewards = loss_fn_reward(k_values_clip,clip_rewards_expand)
    loss_hps_rewards = loss_fn_reward(k_values_hps,hps_rewards_expand)
    return ims,prompts_with_keywords,aesthetic_rewards,clip_rewards,hps_rewards,loss_aesthetic_rewards,loss_clip_rewards,loss_hps_rewards


seed = 1042
from accelerate.utils import set_seed
set_seed(seed, device_specific=False)


batchsize = 1
test_files = "./test_prompt/test_dataset_2000.json"
prompt_dataset_test= load_dataset("json", data_files=test_files)['train']

test_dataloader = torch.utils.data.DataLoader(
        prompt_dataset_test,
        shuffle=False,
        batch_size=batchsize,
        num_workers=1,
        drop_last = True
)
column_names = "raw_prompt" 
inference_dtype = torch.float16

device = "cuda:0"

loss_fn_hps = hps_loss_fn(inference_dtype, device,grad_scale=10)

pipeline= AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16)
pipeline = pipeline.to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_model.text_model = pipeline.text_encoder.text_model
aesthetic_model = AestheticScorerDiff(clip_model=clip_model, dtype=inference_dtype)
clip_model.requires_grad_(False)


pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.unet.requires_grad_(False)
clip_model.requires_grad_(False)

pipeline.vae.to(device, dtype=inference_dtype)
pipeline.text_encoder.to(device, dtype=inference_dtype)
pipeline.unet.to(device, dtype=inference_dtype)
clip_model.to(device, dtype=inference_dtype)
aesthetic_model.to(device, dtype=inference_dtype)

loss_fn_aesthetic = aesthetic_loss_fn(aesthetic_model=aesthetic_model,
                                    grad_scale=1,
                                    aesthetic_target=10,
                                    torch_dtype=inference_dtype,
                                    device=device)
loss_fn_clip = clip_loss_fn(clip_model=clip_model, clip_target=1, grad_scale=10,
                            device=device, torch_dtype=inference_dtype)

loss_fn_reward = torch.nn.MSELoss() 

keywords_table = KeywordsTable(Config(device='cpu',keywords_filename='keywords_list.txt'))

keywords_embs = torch.from_numpy(np.array(keywords_table.flavors.embeds)).unsqueeze(dim=0).expand(batchsize,-1,-1).to(device,inference_dtype)
labels = keywords_table.flavors.labels

test_neg_prompt_embeds = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device))[0]
neg_prompt_embeds = test_neg_prompt_embeds.repeat(batchsize, 1, 1)


ak4prompts = AK4Prompts().to(device)

checkpointpath = "./checkpoints/SD1.5_LCMLoRA_S4_aes1_clip2.25_hps2.25/pytorch_model.bin"
ak4prompts.load_state_dict(torch.load(checkpointpath))


def test_model_score(topk,rewards_scale,guidance_scale):
    log_dir = f"{os.path.dirname(checkpointpath)}/test_model_score/top{topk}_{'rewards_scale'+'_'.join([str(s) for s in rewards_scale])}_guidance_scale{guidance_scale}"
    os.makedirs(log_dir, exist_ok=True)
    log_dir_images = os.path.join(log_dir,"images")
    os.makedirs(log_dir_images, exist_ok=True)
    log_dir_data = os.path.join(log_dir,"data")
    os.makedirs(log_dir_data, exist_ok=True)

    all_eval_images = []

    all_eval_aesthetic_rewards = []
    all_eval_clip_rewards = []
    all_eval_hps_rewards = []

    all_eval_loss_clip = []
    all_eval_loss_hps = []
    all_eval_loss_aesthetic = []
    progress_bar = tqdm(range(0, int(len(test_dataloader))),initial=0,desc="Steps")
    for step, batch in enumerate(test_dataloader):
        all_eval_images = []

        prompts = batch[column_names]
        generator = torch.cuda.manual_seed(seed+step)
        torch.manual_seed(seed+step)
        latent = torch.randn((batchsize,4, 64, 64), device=device, dtype=inference_dtype,generator=generator)
        with torch.no_grad():
            ims,prompts_with_keywords,aesthetic_rewards,clip_rewards,hps_rewards,loss_aesthetic_rewards,loss_clip_rewards,loss_hps_rewards = evaluate(latent, neg_prompt_embeds,
                                                            prompts, keywords_embs, ak4prompts, pipeline, device, loss_fn_aesthetic, loss_fn_clip,loss_fn_hps,loss_fn_reward,labels,topk,rewards_scale,guidance_scale)
            all_eval_images.append(ims)

            all_eval_aesthetic_rewards.append(aesthetic_rewards.squeeze())
            all_eval_clip_rewards.append(clip_rewards.squeeze())
            all_eval_hps_rewards.append(hps_rewards.squeeze())
            
            all_eval_loss_aesthetic.append(loss_aesthetic_rewards.squeeze())
            all_eval_loss_clip.append(loss_clip_rewards.squeeze())
            all_eval_loss_hps.append(loss_hps_rewards.squeeze())

        eval_images = torch.cat(all_eval_images)
        progress_bar.update(1)

        for i, eval_image in enumerate(eval_images):
            eval_image = (
                eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
            pil = Image.fromarray(
                (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            prompt = prompts[i]
            pil.save(
                f"{log_dir_images}/{step*batchsize+i:03d}_{prompt.replace('/','_'):.60}.png")

    metric= {}

    eval_aesthetic_rewards = torch.cat(all_eval_aesthetic_rewards).to('cpu').to(torch.float32)
    eval_clip_rewards = torch.cat(all_eval_clip_rewards).to('cpu').to(torch.float32)
    eval_hps_rewards = torch.cat(all_eval_hps_rewards).to('cpu').to(torch.float32)

    eval_aesthetic_rewards_mean = eval_aesthetic_rewards.mean()
    eval_aesthetic_rewards_min = torch.min(eval_aesthetic_rewards)
    eval_aesthetic_rewards_max = torch.max(eval_aesthetic_rewards)
    eval_aesthetic_rewards_avg = (eval_aesthetic_rewards_mean-eval_aesthetic_rewards_min)/(eval_aesthetic_rewards_max-eval_aesthetic_rewards_min)

    eval_clip_rewards_mean = eval_clip_rewards.mean()
    eval_clip_rewards_min = torch.min(eval_clip_rewards)
    eval_clip_rewards_max = torch.max(eval_clip_rewards)
    eval_clip_rewards_avg = (eval_clip_rewards_mean-eval_clip_rewards_min)/(eval_clip_rewards_max-eval_clip_rewards_min)

    eval_hps_rewards_mean = eval_hps_rewards.mean()
    eval_hps_rewards_min = torch.min(eval_hps_rewards)
    eval_hps_rewards_max = torch.max(eval_hps_rewards)
    eval_hps_rewards_avg = (eval_hps_rewards_mean-eval_hps_rewards_min)/(eval_hps_rewards_max-eval_hps_rewards_min)
    
    avg_rewards = (eval_aesthetic_rewards_avg+eval_clip_rewards_avg+eval_hps_rewards_avg)/3

    eval_loss_aesthetic = torch.stack(all_eval_loss_aesthetic)
    eval_loss_clip = torch.stack(all_eval_loss_clip)
    eval_loss_hps = torch.stack(all_eval_loss_hps)

    eval_loss_aesthetic_mean = eval_loss_aesthetic.mean()
    eval_loss_clip_mean = eval_loss_clip.mean()
    eval_loss_hps_mean = eval_loss_hps.mean()

    eval_aesthetic_rewards_std = eval_aesthetic_rewards.std()
    eval_clip_rewards_std = eval_clip_rewards.std()
    eval_hps_rewards_std = eval_hps_rewards.std()

    with open(f"{log_dir}/metric.json","w") as f:
        metric["avg_rewards"] = avg_rewards.item()

        metric["eval_aesthetic_rewards_mean"] = eval_aesthetic_rewards_mean.item()
        metric["eval_clip_rewards_mean"] = eval_clip_rewards_mean.item()
        metric["eval_hps_rewards_mean"] = eval_hps_rewards_mean.item()

        metric["eval_aesthetic_rewards_avg"] = eval_aesthetic_rewards_avg.item()
        metric["eval_clip_rewards_avg"] = eval_clip_rewards_avg.item()
        metric["eval_hps_rewards_avg"] = eval_hps_rewards_avg.item()

        metric["eval_aesthetic_rewards_min"] = eval_aesthetic_rewards_min.item()
        metric["eval_aesthetic_rewards_max"] = eval_aesthetic_rewards_max.item()
        metric["eval_clip_rewards_min"] = eval_clip_rewards_min.item()
        metric["eval_clip_rewards_max"] = eval_clip_rewards_max.item()        
        metric["eval_hps_rewards_min"] = eval_hps_rewards_min.item()
        metric["eval_hps_rewards_max"] = eval_hps_rewards_max.item()

        metric["eval_aesthetic_rewards_std"] = eval_aesthetic_rewards_std.item()
        metric["eval_clip_rewards_std"] = eval_clip_rewards_std.item()
        metric["eval_hps_rewards_std"] = eval_hps_rewards_std.item()
        json.dump(metric,f)

def test_most_appearance_keywords_score(topk,rewards_scale,guidance_scale):
    keywords_list = ["highly detailed","sharp focus","concept art","intricate","artstation","digital painting","smooth","elegant","illustration","cinematic lighting",\
                    "octane render","trending on artstation","8 k"]
    append_keywords = ','+','.join(keywords_list[:topk])

    log_dir = f"{os.path.dirname(checkpointpath)}/test_most_appearance_keywords_score/top{topk}_guidance_scale{guidance_scale}"
    os.makedirs(log_dir, exist_ok=True)
    log_dir_images = os.path.join(log_dir,"images")
    os.makedirs(log_dir_images, exist_ok=True)
    log_dir_data = os.path.join(log_dir,"data")
    os.makedirs(log_dir_data, exist_ok=True)

    all_eval_aesthetic_rewards = []
    all_eval_clip_rewards = []
    all_eval_hps_rewards = []
    progress_bar = tqdm(range(0, int(len(test_dataloader))),initial=0,desc="Steps")

    for step, batch in enumerate(test_dataloader):
        all_eval_images = []

        prompts = batch[column_names]

        prompts_with_keywords = [p + append_keywords for p in prompts]

        generator = torch.cuda.manual_seed(seed+step)
        torch.manual_seed(seed+step)
        latent = torch.randn((batchsize,4, 64, 64), device=device, dtype=inference_dtype,generator=generator)

        prompt_inputs = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(device)
        text_encoder_output = pipeline.text_encoder(prompt_ids)
        pooled_states = text_encoder_output[1]

        prompt_inputs = pipeline.tokenizer(
            prompts_with_keywords,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(device)

        images = pipeline(
            prompt=prompts_with_keywords, num_inference_steps=4, latent=latent, guidance_scale=1.0, output_type = "latent"
        ).images
        pipeline.upcast_vae()
        images = images.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
        ims =  pipeline.vae.decode(images / pipeline.vae.config.scaling_factor).sample
        ims = ims.to(inference_dtype)
        pipeline.vae.to(dtype=torch.float16)

        _, aesthetic_rewards = loss_fn_aesthetic(ims)
        loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
        loss_hps, hps_rewards = loss_fn_hps(ims,prompts)

        all_eval_images.append(ims)
        all_eval_aesthetic_rewards.append(aesthetic_rewards)
        all_eval_clip_rewards.append(clip_rewards)
        all_eval_hps_rewards.append(hps_rewards)

        eval_images = torch.cat(all_eval_images)
        progress_bar.update(1)

        for i, eval_image in enumerate(eval_images):
            eval_image = (
                eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
            pil = Image.fromarray(
                (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            prompt = prompts[i]
            pil.save(
                f"{log_dir_images}/{step*batchsize+i:03d}_{prompt.replace('/','_'):.60}.png")

    metric= {}

    eval_aesthetic_rewards = torch.cat(all_eval_aesthetic_rewards).to('cpu').to(torch.float32)
    eval_clip_rewards = torch.cat(all_eval_clip_rewards).to('cpu').to(torch.float32)
    eval_hps_rewards = torch.cat(all_eval_hps_rewards).to('cpu').to(torch.float32)

    eval_aesthetic_rewards_mean = eval_aesthetic_rewards.mean()
    eval_aesthetic_rewards_min = torch.min(eval_aesthetic_rewards)
    eval_aesthetic_rewards_max = torch.max(eval_aesthetic_rewards)
    eval_aesthetic_rewards_avg = (eval_aesthetic_rewards_mean-eval_aesthetic_rewards_min)/(eval_aesthetic_rewards_max-eval_aesthetic_rewards_min)

    eval_clip_rewards_mean = eval_clip_rewards.mean()
    eval_clip_rewards_min = torch.min(eval_clip_rewards)
    eval_clip_rewards_max = torch.max(eval_clip_rewards)
    eval_clip_rewards_avg = (eval_clip_rewards_mean-eval_clip_rewards_min)/(eval_clip_rewards_max-eval_clip_rewards_min)

    eval_hps_rewards_mean = eval_hps_rewards.mean()
    eval_hps_rewards_min = torch.min(eval_hps_rewards)
    eval_hps_rewards_max = torch.max(eval_hps_rewards)
    eval_hps_rewards_avg = (eval_hps_rewards_mean-eval_hps_rewards_min)/(eval_hps_rewards_max-eval_hps_rewards_min)
    
    avg_rewards = (eval_aesthetic_rewards_avg+eval_clip_rewards_avg+eval_hps_rewards_avg)/3

    eval_aesthetic_rewards_std = eval_aesthetic_rewards.std()
    eval_clip_rewards_std = eval_clip_rewards.std()
    eval_hps_rewards_std = eval_hps_rewards.std()

    with open(f"{log_dir}/metric.json","w") as f:
        metric["avg_rewards"] = avg_rewards.item()

        metric["eval_aesthetic_rewards_mean"] = eval_aesthetic_rewards_mean.item()
        metric["eval_clip_rewards_mean"] = eval_clip_rewards_mean.item()
        metric["eval_hps_rewards_mean"] = eval_hps_rewards_mean.item()

        metric["eval_aesthetic_rewards_avg"] = eval_aesthetic_rewards_avg.item()
        metric["eval_clip_rewards_avg"] = eval_clip_rewards_avg.item()
        metric["eval_hps_rewards_avg"] = eval_hps_rewards_avg.item()

        metric["eval_aesthetic_rewards_min"] = eval_aesthetic_rewards_min.item()
        metric["eval_aesthetic_rewards_max"] = eval_aesthetic_rewards_max.item()
        metric["eval_clip_rewards_min"] = eval_clip_rewards_min.item()
        metric["eval_clip_rewards_max"] = eval_clip_rewards_max.item()        
        metric["eval_hps_rewards_min"] = eval_hps_rewards_min.item()
        metric["eval_hps_rewards_max"] = eval_hps_rewards_max.item()

        metric["eval_aesthetic_rewards_std"] = eval_aesthetic_rewards_std.item()
        metric["eval_clip_rewards_std"] = eval_clip_rewards_std.item()
        metric["eval_hps_rewards_std"] = eval_hps_rewards_std.item()
        json.dump(metric,f)
    
def test_no_keywords_score(rewards_scale,guidance_scale):
    log_dir = f"{os.path.dirname(checkpointpath)}/test_no_keywords_score/guidance_scale{guidance_scale}"
    os.makedirs(log_dir, exist_ok=True)
    log_dir_images = os.path.join(log_dir,"images")
    os.makedirs(log_dir_images, exist_ok=True)
    log_dir_data = os.path.join(log_dir,"data")
    os.makedirs(log_dir_data, exist_ok=True)

    all_eval_aesthetic_rewards = []
    all_eval_clip_rewards = []
    all_eval_hps_rewards = []
    progress_bar = tqdm(range(0, int(len(test_dataloader))),initial=0,desc="Steps")
    for step, batch in enumerate(test_dataloader):
        all_eval_images = []

        prompts = batch[column_names]

        generator = torch.cuda.manual_seed(seed+step)
        torch.manual_seed(seed+step)
        latent = torch.randn((batchsize,4, 64, 64), device=device, dtype=inference_dtype,generator=generator)

        prompt_inputs = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(device)
        text_encoder_output = pipeline.text_encoder(prompt_ids)
        pooled_states = text_encoder_output[1]

        images = pipeline(
            prompt=prompts, num_inference_steps=4, latent=latent, guidance_scale=1.0, output_type = "latent"
        ).images
        pipeline.upcast_vae()
        images = images.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
        ims =  pipeline.vae.decode(images / pipeline.vae.config.scaling_factor).sample
        ims = ims.to(inference_dtype)
        pipeline.vae.to(dtype=torch.float16)

        _, aesthetic_rewards = loss_fn_aesthetic(ims)
        loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
        rewards = aesthetic_rewards + clip_rewards

        _, aesthetic_rewards = loss_fn_aesthetic(ims)
        loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
        loss_hps, hps_rewards = loss_fn_hps(ims, prompts)

        all_eval_images.append(ims)
        all_eval_aesthetic_rewards.append(aesthetic_rewards)
        all_eval_clip_rewards.append(clip_rewards)
        all_eval_hps_rewards.append(hps_rewards)

        eval_images = torch.cat(all_eval_images)
        progress_bar.update(1)

        for i, eval_image in enumerate(eval_images):
            eval_image = (
                eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
            pil = Image.fromarray(
                (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            prompt = prompts[i]
            pil.save(
                f"{log_dir_images}/{step*batchsize+i:03d}_{prompt.replace('/','_'):.60}.png")

    metric= {}

    eval_aesthetic_rewards = torch.cat(all_eval_aesthetic_rewards).to('cpu').to(torch.float32)
    eval_clip_rewards = torch.cat(all_eval_clip_rewards).to('cpu').to(torch.float32)
    eval_hps_rewards = torch.cat(all_eval_hps_rewards).to('cpu').to(torch.float32)

    eval_aesthetic_rewards_mean = eval_aesthetic_rewards.mean()
    eval_aesthetic_rewards_min = torch.min(eval_aesthetic_rewards)
    eval_aesthetic_rewards_max = torch.max(eval_aesthetic_rewards)
    eval_aesthetic_rewards_avg = (eval_aesthetic_rewards_mean-eval_aesthetic_rewards_min)/(eval_aesthetic_rewards_max-eval_aesthetic_rewards_min)

    eval_clip_rewards_mean = eval_clip_rewards.mean()
    eval_clip_rewards_min = torch.min(eval_clip_rewards)
    eval_clip_rewards_max = torch.max(eval_clip_rewards)
    eval_clip_rewards_avg = (eval_clip_rewards_mean-eval_clip_rewards_min)/(eval_clip_rewards_max-eval_clip_rewards_min)

    eval_hps_rewards_mean = eval_hps_rewards.mean()
    eval_hps_rewards_min = torch.min(eval_hps_rewards)
    eval_hps_rewards_max = torch.max(eval_hps_rewards)
    eval_hps_rewards_avg = (eval_hps_rewards_mean-eval_hps_rewards_min)/(eval_hps_rewards_max-eval_hps_rewards_min)
    
    avg_rewards = (eval_aesthetic_rewards_avg+eval_clip_rewards_avg+eval_hps_rewards_avg)/3

    eval_aesthetic_rewards_std = eval_aesthetic_rewards.std()
    eval_clip_rewards_std = eval_clip_rewards.std()
    eval_hps_rewards_std = eval_hps_rewards.std()

    with open(f"{log_dir}/metric.json","w") as f:
        metric["avg_rewards"] = avg_rewards.item()

        metric["eval_aesthetic_rewards_mean"] = eval_aesthetic_rewards_mean.item()
        metric["eval_clip_rewards_mean"] = eval_clip_rewards_mean.item()
        metric["eval_hps_rewards_mean"] = eval_hps_rewards_mean.item()

        metric["eval_aesthetic_rewards_avg"] = eval_aesthetic_rewards_avg.item()
        metric["eval_clip_rewards_avg"] = eval_clip_rewards_avg.item()
        metric["eval_hps_rewards_avg"] = eval_hps_rewards_avg.item()

        metric["eval_aesthetic_rewards_min"] = eval_aesthetic_rewards_min.item()
        metric["eval_aesthetic_rewards_max"] = eval_aesthetic_rewards_max.item()
        metric["eval_clip_rewards_min"] = eval_clip_rewards_min.item()
        metric["eval_clip_rewards_max"] = eval_clip_rewards_max.item()        
        metric["eval_hps_rewards_min"] = eval_hps_rewards_min.item()
        metric["eval_hps_rewards_max"] = eval_hps_rewards_max.item()

        metric["eval_aesthetic_rewards_std"] = eval_aesthetic_rewards_std.item()
        metric["eval_clip_rewards_std"] = eval_clip_rewards_std.item()
        metric["eval_hps_rewards_std"] = eval_hps_rewards_std.item()
        json.dump(metric,f)

def test_human_choosed_keywords_score(topk,rewards_scale,guidance_scale):
    ### Best Prompts for Text-to-Image Models and How to Find Them
    keywords_list = ["cinematic","colorful background","concept art","dramatic lighting","high detail","highly detailed","hyper realistic","intricate",\
                     "intricate sharp details","octane render",\
                    "smooth","studio lighting","trending on artstation."]
    append_keywords = ','+','.join(keywords_list[:topk])

    log_dir = f"{os.path.dirname(checkpointpath)}/test_human_choosed_keywords_score/top{topk}_guidance_scale{guidance_scale}"
    os.makedirs(log_dir, exist_ok=True)
    log_dir_images = os.path.join(log_dir,"images")
    os.makedirs(log_dir_images, exist_ok=True)
    log_dir_data = os.path.join(log_dir,"data")
    os.makedirs(log_dir_data, exist_ok=True)

    all_eval_aesthetic_rewards = []
    all_eval_clip_rewards = []
    all_eval_hps_rewards = []
    progress_bar = tqdm(range(0, int(len(test_dataloader))),initial=0,desc="Steps")
    for step, batch in enumerate(test_dataloader):
        all_eval_images = []

        prompts = batch[column_names]
        prompts_with_keywords = [p + append_keywords for p in prompts]

        generator = torch.cuda.manual_seed(seed+step)
        torch.manual_seed(seed+step)
        latent = torch.randn((batchsize,4, 64, 64), device=device, dtype=inference_dtype,generator=generator)

        prompt_inputs = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(device)
        text_encoder_output = pipeline.text_encoder(prompt_ids)
        pooled_states = text_encoder_output[1]

        prompt_inputs = pipeline.tokenizer(
            prompts_with_keywords,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(device)

        images = pipeline(
            prompt=prompts_with_keywords, num_inference_steps=4, latent=latent, guidance_scale=1.0, output_type = "latent"
        ).images
        images = images.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
        ims =  pipeline.vae.decode(images / pipeline.vae.config.scaling_factor).sample
        ims = ims.to(inference_dtype)

        _, aesthetic_rewards = loss_fn_aesthetic(ims)
        loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
        loss_hps, hps_rewards = loss_fn_hps(ims,prompts)

        all_eval_images.append(ims)
        all_eval_aesthetic_rewards.append(aesthetic_rewards)
        all_eval_clip_rewards.append(clip_rewards)
        all_eval_hps_rewards.append(hps_rewards)

        eval_images = torch.cat(all_eval_images)
        progress_bar.update(1)

        for i, eval_image in enumerate(eval_images):
            eval_image = (
                eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
            pil = Image.fromarray(
                (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            prompt = prompts[i]
            pil.save(
                f"{log_dir_images}/{step*batchsize+i:03d}_{prompt.replace('/','_'):.60}.png")

    metric= {}

    eval_aesthetic_rewards = torch.cat(all_eval_aesthetic_rewards).to('cpu').to(torch.float32)
    eval_clip_rewards = torch.cat(all_eval_clip_rewards).to('cpu').to(torch.float32)
    eval_hps_rewards = torch.cat(all_eval_hps_rewards).to('cpu').to(torch.float32)

    eval_aesthetic_rewards_mean = eval_aesthetic_rewards.mean()
    eval_aesthetic_rewards_min = torch.min(eval_aesthetic_rewards)
    eval_aesthetic_rewards_max = torch.max(eval_aesthetic_rewards)
    eval_aesthetic_rewards_avg = (eval_aesthetic_rewards_mean-eval_aesthetic_rewards_min)/(eval_aesthetic_rewards_max-eval_aesthetic_rewards_min)

    eval_clip_rewards_mean = eval_clip_rewards.mean()
    eval_clip_rewards_min = torch.min(eval_clip_rewards)
    eval_clip_rewards_max = torch.max(eval_clip_rewards)
    eval_clip_rewards_avg = (eval_clip_rewards_mean-eval_clip_rewards_min)/(eval_clip_rewards_max-eval_clip_rewards_min)

    eval_hps_rewards_mean = eval_hps_rewards.mean()
    eval_hps_rewards_min = torch.min(eval_hps_rewards)
    eval_hps_rewards_max = torch.max(eval_hps_rewards)
    eval_hps_rewards_avg = (eval_hps_rewards_mean-eval_hps_rewards_min)/(eval_hps_rewards_max-eval_hps_rewards_min)
    
    avg_rewards = (eval_aesthetic_rewards_avg+eval_clip_rewards_avg+eval_hps_rewards_avg)/3

    eval_aesthetic_rewards_std = eval_aesthetic_rewards.std()
    eval_clip_rewards_std = eval_clip_rewards.std()
    eval_hps_rewards_std = eval_hps_rewards.std()

    with open(f"{log_dir}/metric.json","w") as f:
        metric["avg_rewards"] = avg_rewards.item()

        metric["eval_aesthetic_rewards_mean"] = eval_aesthetic_rewards_mean.item()
        metric["eval_clip_rewards_mean"] = eval_clip_rewards_mean.item()
        metric["eval_hps_rewards_mean"] = eval_hps_rewards_mean.item()

        metric["eval_aesthetic_rewards_avg"] = eval_aesthetic_rewards_avg.item()
        metric["eval_clip_rewards_avg"] = eval_clip_rewards_avg.item()
        metric["eval_hps_rewards_avg"] = eval_hps_rewards_avg.item()

        metric["eval_aesthetic_rewards_min"] = eval_aesthetic_rewards_min.item()
        metric["eval_aesthetic_rewards_max"] = eval_aesthetic_rewards_max.item()
        metric["eval_clip_rewards_min"] = eval_clip_rewards_min.item()
        metric["eval_clip_rewards_max"] = eval_clip_rewards_max.item()        
        metric["eval_hps_rewards_min"] = eval_hps_rewards_min.item()
        metric["eval_hps_rewards_max"] = eval_hps_rewards_max.item()

        metric["eval_aesthetic_rewards_std"] = eval_aesthetic_rewards_std.item()
        metric["eval_clip_rewards_std"] = eval_clip_rewards_std.item()
        metric["eval_hps_rewards_std"] = eval_hps_rewards_std.item()
        json.dump(metric,f)


# evaluation
import argparse
parser = argparse.ArgumentParser(description='Evalution the model checkpoint and baselines')
parser.add_argument('--topk', type=int, help='choose topk keywords')
parser.add_argument('--guidance_scale', type=float, default=1.0, help='classifier-free guidance scale')
parser.add_argument('--aes', type=float, default=1.0, help='aesthetic weights for customized keywords-ranking')
parser.add_argument('--clip', type=float, default=5.0, help='clip weights for customized keywords-ranking')
parser.add_argument('--hps', type=float, default=3.0, help='hps weights for customized keywords-ranking')

args = parser.parse_args()
topk = args.topk
guidance_scale = args.guidance_scale
aes = args.aes
clip = args.clip
hps = args.hps

test_model_score(topk,[aes,clip,hps],guidance_scale)
test_no_keywords_score([aes,clip,hps],guidance_scale)
test_most_appearance_keywords_score(topk,[aes,clip,hps],guidance_scale)
test_human_choosed_keywords_score(topk,[aes,clip,hps],guidance_scale)


