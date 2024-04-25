from keywords.keywords_table import KeywordsTable, Config
from datasets import load_dataset, load_from_disk
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
import transformers
from diffusers.optimization import get_scheduler
import accelerate
from packaging import version
import datasets
from AK4Prompts import AK4Prompts
from ml_collections import config_flags
from absl import app, flags
from accelerate import Accelerator
import datetime
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel,LCMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import AutoProcessor, AutoModel, CLIPModel
import torchvision
import contextlib
import wandb
import torch.utils.checkpoint as checkpoint
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
from aesthetic_scorer import AestheticScorerDiff
import math
import shutil
import diffusers
import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
# import prompts as prompts_file
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/align_prop.py", "Training configuration.")
logger = get_logger(__name__)


def hps_loss_fn(inference_dtype=None, device=None, grad_scale=10):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        None,
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
    model.requires_grad_(False)
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


def aesthetic_loss_fn(aesthetic_model=None,
                      aesthetic_target=None,
                      grad_scale=0,
                      device=None,
                      accelerator=None,
                      torch_dtype=None):
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])
    # scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer = aesthetic_model
    scorer.requires_grad_(False)
    target_size = 224

    def loss_fn(im_pix_un):
        with torch.no_grad():
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


def clip_loss_fn(clip_model=None,
                 clip_target=1,
                 grad_scale=0,
                 device=None,
                 accelerator=None,
                 torch_dtype=None):
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])
    clip_model.requires_grad_(False)
    target_size = 224

    def loss_fn(im_pix_un, text_pooled_states):
        with torch.no_grad():
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

def evaluate(latent, train_neg_prompt_embeds, prompts, keywords_embs, adapter, pipeline, accelerator, inference_dtype, config, loss_fn_aesthetic,loss_fn_clip,loss_fn_hps,loss_fn_reward,labels):
    prompt_inputs = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    )
    prompt_ids = prompt_inputs.input_ids.to(accelerator.device)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(
        accelerator.device)
    text_encoder_output = pipeline.text_encoder(prompt_ids)
    prompt_embeds = text_encoder_output[0]
    pooled_states = text_encoder_output[1]

    attention_mask = (prompt_inputs.attention_mask==0).to(accelerator.device)

    adapter.eval()
    with torch.no_grad():
        aesthetic_out_fc,clip_out_fc,hps_out_fc = adapter(prompt_embeds, attention_mask, keywords_embs)
        max_prompt_available_len = 77 - torch.max(torch.sum(~attention_mask,dim=-1))   
        #topk
        num_k = min(config.test_topk,int(max_prompt_available_len))   
        final_score = aesthetic_out_fc*config.rewards_scale[0] + clip_out_fc*config.rewards_scale[1] + hps_out_fc*config.rewards_scale[2]
        k_values, k_indices = torch.topk(final_score, k=num_k) 
        k_values_aesthetic = torch.gather(aesthetic_out_fc, dim=1, index=k_indices)
        k_values_clip = torch.gather(clip_out_fc, dim=1, index=k_indices)
        k_values_hps = torch.gather(hps_out_fc, dim=1, index=k_indices)
    adapter.train()

    append_keywords_choosed = [","+",".join([labels[idx] for idx in indices]) for indices in k_indices]
    
    prompts_with_keywords = [prompt + append_keywords for prompt, append_keywords in zip(prompts,append_keywords_choosed)]
    
    prompt_inputs = pipeline.tokenizer(
        prompts_with_keywords,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    )
    prompt_ids = prompt_inputs.input_ids.to(accelerator.device)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(
        accelerator.device)
    text_encoder_output = pipeline.text_encoder(prompt_ids)
    new_prompt_embeds = text_encoder_output[0]
        
    new_prompt_embeds = new_prompt_embeds.to(inference_dtype)

    # Output the top keywords and weights corresponding to each prompt
    prompt_keywords=[]
    for i in range(len(prompts)):
        labels_choosed = ','.join([labels[index] for j,index in enumerate(k_indices[i])])
        aesthetic_values =  ','.join(["{:.10f}".format(float(k_values_aesthetic[i][j])) for j,index in enumerate(k_indices[i])])
        clip_values =  ','.join(["{:.10f}".format(float(k_values_clip[i][j])) for j,index in enumerate(k_indices[i])])
        hps_values = ','.join(["{:.10f}".format(float(k_values_hps[i][j])) for j,index in enumerate(k_indices[i])])
        final_rewards =  ','.join(["{:.10f}".format(float(k_values[i][j])) for j,index in enumerate(k_indices[i])])
        index = ','.join([str(int(index)) for j,index in enumerate(k_indices[i])])
        prompt_keywords.append([prompts[i],labels_choosed,aesthetic_values,clip_values,hps_values,final_rewards,index])

    pipeline.scheduler.set_timesteps(4)
    for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), total=len(pipeline.scheduler.timesteps), disable=True):
        t = torch.tensor([t],
                         dtype=inference_dtype,
                         device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        # noise_pred_uncond = pipeline.unet(
        #     latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = pipeline.unet(latent, t, new_prompt_embeds).sample

        # grad = (noise_pred_cond - noise_pred_uncond)
        # noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(
            noise_pred_cond, t[0].long(), latent).prev_sample
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample

    _, aesthetic_rewards = loss_fn_aesthetic(ims)
    loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
    loss_hps, hps_rewards = loss_fn_hps(ims, prompts)

    final_rewards = aesthetic_rewards*config.rewards_scale[0] + clip_rewards*config.rewards_scale[0] + hps_rewards*config.rewards_scale[1]
    
    aesthetic_rewards = aesthetic_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_aesthetic.shape[1])  #[batch] --> [batch,k_values.shape[1]]
    clip_rewards = clip_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_clip.shape[1])  #[batch] --> [batch,k_values.shape[1]]
    hps_rewards = hps_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_hps.shape[1])  #[batch] --> [batch,k_values.shape[1]]

    assert aesthetic_rewards.shape == k_values_aesthetic.shape
    assert clip_rewards.shape == k_values_clip.shape
    loss_aesthetic_rewards = loss_fn_reward(k_values_aesthetic,aesthetic_rewards)
    loss_clip_rewards = loss_fn_reward(k_values_clip,clip_rewards)
    loss_hps_rewards = loss_fn_reward(k_values_hps,hps_rewards)
    return ims,aesthetic_rewards,clip_rewards,hps_rewards,final_rewards,prompt_keywords,loss_aesthetic_rewards,loss_clip_rewards,loss_hps_rewards


def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from is not None:
        config.run_name = os.path.basename(os.path.dirname(config.resume_from))

    output_dir = os.path.join(config.logdir, config.run_name)

    accelerator_config = ProjectConfiguration(project_dir=output_dir)

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb_args = {"name":f"{config.run_name}","resume":config.resume_from is not None}
        # wandb_args = {"resume": "must", "id": "xxxxxxx"}
        if config.debug:
            wandb_args = {'mode': "disabled"}
        accelerator.init_trackers(
            project_name="AK4Prompts", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(
            config.logdir, wandb.run.name)
        accelerator.project_configuration.logging_dir = os.path.join(
            config.logdir, wandb.run.name)
        os.makedirs(config.logdir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{config}")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    prompt_dataset = load_dataset("json", data_files=config.prompt_dataset) 
    prompt_dataset_test= load_dataset("json", data_files=config.prompt_dataset_test)['train']

    column_names = 'raw_prompt' #prompt_dataset["train"].column_names[1]

    with accelerator.main_process_first():
        if config.train.total_samples_per_epoch is not None:
            prompt_dataset["train"] = (
                prompt_dataset["train"]
                .shuffle(seed=config.seed)
                .select(range(config.train.total_samples_per_epoch))
            )

    train_dataloader = torch.utils.data.DataLoader(
        prompt_dataset['train'],
        shuffle=True,
        batch_size=config.train.batch_size_per_gpu_available,
        num_workers=config.dataloader_num_workers,
    )
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(
            config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained.model, revision=config.pretrained.revision)
    pipeline.load_lora_weights(config.pretrained.lcm_lora)
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(4)
        
    clip_model = CLIPModel.from_pretrained(config.pretrained.clip_model)
    clip_model.text_model = pipeline.text_encoder.text_model
    aesthetic_model = AestheticScorerDiff(clip_model=clip_model, dtype=torch.float16)

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    clip_model.requires_grad_(False)

    # disable safety checker
    pipeline.safety_checker = None

    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=True,  # not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # switch to DDIM scheduler
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # pipeline.scheduler.set_timesteps(config.steps)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    clip_model.to(accelerator.device, dtype=inference_dtype)
    aesthetic_model.to(accelerator.device, dtype=inference_dtype)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    adapter = AK4Prompts()

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        adapter.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.train.lr_warmup_steps *
        config.train.gradient_accumulation_steps,
        # config.num_epochs * config.train.total_samples_per_epoch / config.train.batch_size_per_gpu
        num_training_steps=config.train.max_train_steps * \
        config.train.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    optimizer,train_dataloader, lr_scheduler, adapter = accelerator.prepare(
        optimizer,train_dataloader, lr_scheduler, adapter)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.train.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext

    # loss define
    loss_fn_hps = hps_loss_fn(inference_dtype, accelerator.device,grad_scale=config.hps_grad_scale)
    loss_fn_aesthetic = aesthetic_loss_fn(aesthetic_model=aesthetic_model,
                                            grad_scale=config.aesthetic_grad_scale,
                                            aesthetic_target=config.aesthetic_target,
                                            accelerator=accelerator,
                                            torch_dtype=inference_dtype,
                                            device=accelerator.device)

    loss_fn_clip = clip_loss_fn(clip_model=clip_model, clip_target=1, grad_scale=config.clip_grad_scale,
                                device=accelerator.device, accelerator=accelerator, torch_dtype=inference_dtype)

    loss_fn_reward = torch.nn.MSELoss()  #criterion2 = nn.L1Loss()

    timesteps = pipeline.scheduler.timesteps

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if config.resume_from:
        if config.resume_from != "latest":
            path = os.path.basename(config.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from}' does not exist. Starting a new training run."
            )
            config.resume_from = None
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            logger.info(f"Resuming from {os.path.join(output_dir, path)}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * config.train.gradient_accumulation_steps
            first_epoch = int(
                global_step // config.train.num_update_steps_per_epoch)
            resume_step = resume_global_step % (
                config.train.num_update_steps_per_epoch * config.train.gradient_accumulation_steps)
    
    with accelerator.main_process_first():
        keywords_table = KeywordsTable(Config(device=accelerator.device.type,keywords_filename=config.keywords_filename))
        keywords_embs = torch.from_numpy(
            np.array(keywords_table.flavors.embeds)).unsqueeze(dim=0).expand(config.train.batch_size_per_gpu_available,-1,-1).to(accelerator.device)
        keywords_encoder_hidden_states = torch.from_numpy(
            np.array(keywords_table.flavors.encoder_hidden_states)).to(accelerator.device)
        keywords_hidden_len = torch.from_numpy(
            np.array(keywords_table.flavors.hidden_len)).to(accelerator.device)
        labels = keywords_table.flavors.labels

    #################### TRAINING ####################
    # Train!
    total_batch_size = (
        config.train.batch_size_per_gpu_available
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {config.train.total_samples_per_epoch}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.train.batch_size_per_gpu_available}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size} {config.train.total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(
        f"  Total optimization steps = {config.train.max_train_steps}")

    progress_bar = tqdm(
        range(0, int(config.train.max_train_steps)),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in list(range(first_epoch, config.num_epochs)):
        adapter.train()
        info = defaultdict(list)
        info_vis = defaultdict(list)
        for step, batch in enumerate(train_dataloader):
            if (
                config.resume_from
                and epoch == first_epoch
                and step < resume_step
            ):
                continue
            with accelerator.accumulate(adapter):
                with autocast():
                    with torch.enable_grad():  # important b/c don't have on by default in module
                        latent = torch.randn((config.train.batch_size_per_gpu_available,
                                            4, 64, 64), device=accelerator.device, dtype=inference_dtype)

                        prompts = batch[column_names]
                        prompt_inputs = pipeline.tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=pipeline.tokenizer.model_max_length,
                        )
                        prompt_ids = prompt_inputs.input_ids.to(accelerator.device)
                        attention_mask = (prompt_inputs.attention_mask==0).to(accelerator.device)
                        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(
                            accelerator.device)
                        text_encoder_output = pipeline.text_encoder(prompt_ids)
                        prompt_embeds = text_encoder_output[0]
                        pooled_states = text_encoder_output[1]

                        aesthetic_out_fc,clip_out_fc,hps_out_fc = adapter(prompt_embeds, attention_mask, keywords_embs)
                        # random selection
                        num_k = torch.tensor([config.train_choose_num]).item()
                        k_indices = torch.randint(0, aesthetic_out_fc.size(1), (aesthetic_out_fc.size(0), num_k)).to(keywords_embs.device)
                        
                        k_values_aesthetic = torch.gather(aesthetic_out_fc, dim=1, index=k_indices)
                        k_values_clip =  torch.gather(clip_out_fc, dim=1, index=k_indices)
                        k_values_hps = torch.gather(hps_out_fc,dim=1,index=k_indices)

                        append_keywords_choosed = [","+",".join([labels[idx] for idx in indices]) for indices in k_indices]
                        
                        prompts_with_keywords = [prompt + append_keywords for prompt, append_keywords in zip(prompts,append_keywords_choosed)]
                        logger.info(prompts_with_keywords)
                        prompt_inputs = pipeline.tokenizer(
                            prompts_with_keywords,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=pipeline.tokenizer.model_max_length,
                        )
                        prompt_ids = prompt_inputs.input_ids.to(accelerator.device)
                        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(
                            accelerator.device)
                        text_encoder_output = pipeline.text_encoder(prompt_ids)
                        new_prompt_embeds = text_encoder_output[0]


                        new_prompt_embeds = new_prompt_embeds.detach()

                        pipeline.scheduler.set_timesteps(4)
                        for i, t in enumerate(timesteps):
                            t = torch.tensor([t],
                                                dtype=inference_dtype,
                                                device=latent.device)
                            t = t.repeat(
                                config.train.batch_size_per_gpu_available)

                            # noise_pred_uncond = pipeline.unet(
                                #     latent, t, train_neg_prompt_embeds).sample
                            noise_pred_cond = pipeline.unet(
                                    latent, t, new_prompt_embeds).sample
                                
                            # grad = (noise_pred_cond - noise_pred_uncond)
                            # noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                            latent = pipeline.scheduler.step(
                                noise_pred_cond, t[0].long(), latent).prev_sample

                        ims = pipeline.vae.decode(
                            latent.to(pipeline.vae.dtype) / 0.18215).sample

                        loss_aesthetic, aesthetic_rewards = loss_fn_aesthetic(ims)
                        loss_clip,clip_rewards = loss_fn_clip(ims,pooled_states)
                        loss_hps, hps_rewards = loss_fn_hps(ims, prompts)

                        aesthetic_rewards = aesthetic_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_aesthetic.shape[1])  #[batch] --> [batch,k_values.shape[1]]
                        clip_rewards = clip_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_clip.shape[1])  #[batch] --> [batch,k_values.shape[1]]
                        hps_rewards = hps_rewards.to(torch.float32).unsqueeze(dim=-1).expand(-1,k_values_hps.shape[1])  #[batch] --> [batch,k_values.shape[1]]

                        loss_aesthetic_rewards = loss_fn_reward(k_values_aesthetic,aesthetic_rewards)
                        loss_clip_rewards = loss_fn_reward(k_values_clip,clip_rewards)
                        loss_hps_rewards = loss_fn_reward(k_values_hps,hps_rewards)

                        loss_all = loss_aesthetic_rewards*config.rewards_scale[0] + \
                                    loss_clip_rewards*config.rewards_scale[1] + \
                                    loss_hps_rewards*config.rewards_scale[2]

                        info["loss_all"].append(loss_all)
                        info["loss_aesthetic_rewards"].append(loss_aesthetic_rewards)
                        info["loss_clip_rewards"].append(loss_clip_rewards)
                        info["loss_hps_rewards"].append(loss_clip_rewards)

                        info["lr"].append(torch.tensor([lr_scheduler.get_last_lr()[0]]).to(accelerator.device))

                        # backward pass
                        accelerator.backward(loss_all)
                                
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                adapter.parameters(), config.train.max_grad_norm)
                
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                assert (
                    step + 1
                ) % config.train.gradient_accumulation_steps == 0

                global_step += 1  # total_batch_size finished
                progress_bar.update(1)

                # log training and evaluation
                eval_aesthetic_rewards_mean = None
                eval_clip_rewards_mean = None
                eval_hps_rewards_mean = None
                eval_loss_aesthetic_mean = None
                eval_loss_clip_mean = None
                eval_loss_hps_mean = None
                if config.visualize_eval and \
                        ((global_step >= config.vis_freq and (global_step % config.vis_freq == 0)) or global_step == 1):

                    all_eval_images = []
                    all_eval_aesthetic_rewards = []
                    all_eval_clip_rewards = []
                    all_eval_hps_rewards = []

                    all_eval_loss_clip = []
                    all_eval_loss_hps = []
                    all_eval_loss_aesthetic = []
                    prompt_keywords = []
                    if config.same_evaluation:
                        generator = torch.cuda.manual_seed(config.seed)
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images,
                                                4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)
                    else:
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images,
                                                4, 64, 64), device=accelerator.device, dtype=inference_dtype)
                    with torch.no_grad():
                        for index in range(config.max_vis_images):
                            prompts = prompt_dataset_test[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)][column_names]

                            ims,aesthetic_rewards,clip_rewards,hps_rewards,final_rewards,prompt_keywords_i,loss_aesthetic_rewards,loss_clip_rewards,loss_hps_rewards = evaluate(latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available * (index+1)], train_neg_prompt_embeds,
                                                    prompts, keywords_embs, adapter, pipeline, accelerator, inference_dtype, config, loss_fn_aesthetic, loss_fn_clip,loss_fn_hps,loss_fn_reward,labels)
                            prompt_keywords.extend(prompt_keywords_i)
                            all_eval_images.append(ims)

                            all_eval_aesthetic_rewards.append(aesthetic_rewards.squeeze())
                            all_eval_clip_rewards.append(clip_rewards.squeeze())
                            all_eval_hps_rewards.append(hps_rewards.squeeze())
                            
                            all_eval_loss_aesthetic.append(loss_aesthetic_rewards.squeeze())
                            all_eval_loss_clip.append(loss_clip_rewards.squeeze())
                            all_eval_loss_hps.append(loss_hps_rewards.squeeze())

                    eval_aesthetic_rewards = torch.cat(all_eval_aesthetic_rewards)
                    eval_clip_rewards = torch.cat(all_eval_clip_rewards)
                    eval_hps_rewards = torch.cat(all_eval_hps_rewards)

                    eval_aesthetic_rewards_mean = eval_aesthetic_rewards.mean()
                    eval_clip_rewards_mean = eval_clip_rewards.mean()
                    eval_hps_rewards_mean = eval_hps_rewards.mean()

                    eval_loss_aesthetic = torch.stack(all_eval_loss_aesthetic)
                    eval_loss_clip = torch.stack(all_eval_loss_clip)
                    eval_loss_hps = torch.stack(all_eval_loss_hps)

                    eval_loss_aesthetic_mean = eval_loss_aesthetic.mean()
                    eval_loss_clip_mean = eval_loss_clip.mean()
                    eval_loss_hps_mean = eval_loss_hps.mean()

                    eval_images = torch.cat(all_eval_images)
                    eval_image_vis = []
                    if accelerator.is_main_process:

                        name_val = wandb.run.name
                        log_dir = f"logs/{name_val}/eval_vis"
                        os.makedirs(log_dir, exist_ok=True)
                        for i, eval_image in enumerate(eval_images):
                            eval_image = (
                                eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray(
                                (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            prompt = prompt_dataset_test[i][column_names]
                            pil.save(
                                f"{log_dir}/{epoch:03d}_{global_step:03d}_{i:03d}_{prompt:.20}.png")
                            pil = pil.resize((256, 256))
                            aesthetic_reward = eval_aesthetic_rewards[i]
                            clip_reward = eval_clip_rewards[i]
                            hps_reward = eval_hps_rewards[i]
                           
                            eval_image_vis.append(wandb.Image(
                                pil, caption=f"{prompt:.60} | {aesthetic_reward[0]:.4f} | {clip_reward[0]:.4f} | {hps_reward[0]:.4f}"))
                        accelerator.log(
                            {"eval_images": eval_image_vis}, step=global_step)
                        columns = ["Prompt", "Predicted Top Keywords", "Keywords Aesthetic Rewards", "Keywords CLIP Rewards","Keywords HPS Rewards","Final Rewards", "Keywords ID"]
                        prompt_keywords_table = wandb.Table(data=prompt_keywords,columns=columns)
                        accelerator.log({f"{global_step}_prompt_keywords":prompt_keywords_table}, step=global_step)

                info = {k: torch.mean(torch.stack(v))
                        for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                # logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                info.update({"epoch": epoch, "inner_epoch": step,
                            "eval_aesthetic_rewards_mean": eval_aesthetic_rewards_mean, "eval_clip_rewards_mean": eval_clip_rewards_mean,"eval_hps_rewards_mean":eval_hps_rewards_mean,"eval_loss_aesthetic_mean":eval_loss_aesthetic_mean,"eval_loss_clip_mean":eval_loss_clip_mean,"eval_loss_hps_mean":eval_loss_hps_mean})
                accelerator.log(info, step=global_step)

                progress_bar.set_postfix(**{"loss": info['loss_all'].item(),"loss_aesthetic": info['loss_aesthetic_rewards'].item(
                ), "loss_clip": info['loss_clip_rewards'].item(),"loss_hps": info['loss_hps_rewards'].item(),"lr": lr_scheduler.get_last_lr()[0]})

                if config.visualize_train:
                    ims = torch.cat(info_vis["image"])
                    rewards = torch.cat(info_vis["rewards_img"])
                    prompts = info_vis["prompts"]
                    images = []
                    for i, image in enumerate(ims):
                        image = (image.clone().detach() /
                                    2 + 0.5).clamp(0, 1)
                        pil = Image.fromarray(
                            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((256, 256))
                        prompt = prompts[i]
                        reward = rewards[i]
                        images.append(wandb.Image(
                            pil, caption=f"{prompt:.25} | {reward:.2f}"))

                    accelerator.log(
                        {"images": images},
                        step=global_step,
                    )

                info = defaultdict(list)
                if global_step >= config.checkpointing_steps and global_step % config.checkpointing_steps == 0 and accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if config.num_checkpoint_limit is not None:
                        checkpoints = os.listdir(
                            accelerator_config.project_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= config.num_checkpoint_limit:
                            num_to_remove = len(
                                checkpoints) - config.num_checkpoint_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    accelerator_config.project_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        accelerator_config.project_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        # make sure we did an optimization step at the end of the inner epoch
        assert accelerator.sync_gradients

if __name__ == "__main__":
    app.run(main)
