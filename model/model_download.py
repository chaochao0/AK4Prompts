from transformers import CLIPModel,CLIPProcessor
from diffusers import DiffusionPipeline,StableDiffusionXLPipeline,AutoPipelineForText2Image

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(repo_id)                                          
pipe.save_pretrained("../models/SD/stable-diffusion-1.5")


# clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# clip.save_pretrained("./models/CLIP/clip-vit-large-patch14")

# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# processor.save_pretrained("./models/CLIP/clip-vit-large-patch14")


# pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
# pipeline.save_pretrained("./models/sdxl-turbo")
