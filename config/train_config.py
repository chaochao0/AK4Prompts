import ml_collections
import os


def general():
    config = ml_collections.ConfigDict()

    ###### Datasets ###### 
    config.dataloader_num_workers = 0
    config.prompt_dataset = "prompts/train_prompts.json"
    config.prompt_dataset_test = "prompts/test_prompts_2000.json"
    config.keywords_filename = "keywords_list.txt"

    config.run_name = "SD1.5_LCMLoRA_S4_aes1_clip2.25_hps2.25"

    ###### General ######    

    config.resume_from = None
    
    ###### Accelerator ###### 
    config.debug =False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"

    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 1042    
      

    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = False
    config.visualize_eval = True

    config.same_evaluation = True
    

    config.train = train = ml_collections.ConfigDict()
    config.train.loss_coeff = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4   #Initial learning rate (after the potential warmup period) to use.
    train.lr_scheduler = "constant" #"constant"   #'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    train.lr_warmup_steps = 200    #"Number of steps for the warmup in the lr scheduler."
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8 
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0    
    
    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.

    pretrained.model = "runwayml/stable-diffusion-v1-5"
    pretrained.lcm_lora = "latent-consistency/lcm-lora-sdv1-5"
    pretrained.clip_model = "openai/clip-vit-large-patch14"
    # revision of the model to load.
    pretrained.revision = "main"

    # not use for lcm-lora
    config.sd_guidance_scale = 7.5
    config.steps = 4 

    return config


def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):
    config.train.total_samples_per_epoch = (total_samples_per_epoch//per_gpu_capacity)*per_gpu_capacity
    #  Samples per epoch
    # config.train.total_samples_per_epoch = total_samples_per_epoch  #(~~~~ this is desired ~~~~)
    config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #(~~~~ this is desired ~~~~)
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus
    config.train.batch_size_per_gpu_available = per_gpu_capacity    #(this quantity depends on the gpu used)
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available
    
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available

    assert config.train.data_loader_iterations%config.train.gradient_accumulation_steps==0, "data_loader_iterations must be divisible by gradient_accumulation_steps"
    config.train.num_update_steps_per_epoch = config.train.data_loader_iterations // config.train.gradient_accumulation_steps
    config.train.max_train_steps = config.num_epochs * config.train.num_update_steps_per_epoch
    return config


def aesthetic_clip_hps():
    config = general()
    config.num_epochs = 40

    # scale
    config.aesthetic_target = 10
    config.aesthetic_grad_scale = 1
    config.clip_grad_scale = 10
    config.hps_grad_scale = 10
    config.rewards_scale = [1.0, 2.25, 2.25]

    # test for choosing top10 keywords
    config.test_topk=10  
    # during training, keywords train_choose_num were randomly selected to generate
    config.train_choose_num = 1  

    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 0.01    #loss = loss * config.train.loss_coeff
    config.train.adam_weight_decay = 0.01
    config.train.learning_rate = 1e-5

    # After every vis_freq total_batch_size training iterations, generate max_vis_images images with batch_size_per_gpu_available available per GPU.
    config.max_vis_images = 40
    config.vis_freq = 500 

    config.checkpointing_steps = 500
    config.num_checkpoint_limit = 500

    config = set_config_batch(config,total_samples_per_epoch=141408, total_batch_size = 32, per_gpu_capacity=4)  
    return config
    

def get_config(name):
    return globals()[name]()