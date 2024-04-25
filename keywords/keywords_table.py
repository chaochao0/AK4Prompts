import hashlib
import math
import numpy as np
from transformers import CLIPTextModel,CLIPTokenizer
import os
import requests
import time
import torch

from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from tqdm import tqdm
from typing import List, Optional

from safetensors.numpy import load_file, save_file




@dataclass 
class Config:
    # sd settings
    SD_model_path: Optional[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/SD/stable-diffusion-v1-5")
    SD_model_name: str = os.path.basename(SD_model_path)


    # interrogator settings
    keywords_filename: str = None   
    cache_path: str = os.path.join(os.path.dirname(__file__),'text_embeddings')   # path to store cached text embeddings
    # generate_cache: bool = True # when true, cached embeds are generated if cache_path don't have
    chunk_size: int = 2048      # batch size for CLIP, use smaller for lower VRAM
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    flavor_intermediate_count: int = 2048   #
    quiet: bool = False # when quiet progress bars are not shown

class KeywordsTable():
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if 'cuda' in self.device else torch.float32

        self.flavors = LabelTable(load_list(self.config.data_path, self.config.keywords_filename),os.path.splitext(self.config.keywords_filename)[0], self.config)


class LabelTable():
    def __init__(self, labels:List[str], desc:str, config: Config):

        config = config
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.encoder_hidden_states = []
        self.hidden_len = []
        self.labels = labels
        self.dtype = torch.float16 if 'cuda' in self.device else torch.float32


        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()
        sanitized_name = self.config.SD_model_name.replace('/', '_').replace('@', '_')
        self._load_cached(desc, hash, sanitized_name)

        if len(self.labels) != len(self.embeds):
            start_time = time.time()

            tokenizer = CLIPTokenizer.from_pretrained(config.SD_model_path, subfolder="tokenizer",local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(config.SD_model_path,subfolder="text_encoder",local_files_only=True).to(config.device, dtype=self.dtype)

            self.embeds = []
            self.encoder_hidden_states = []
            self.hidden_len = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                inputs = tokenizer(chunk.tolist(),padding = True,return_tensors="pt")
                with torch.no_grad(), torch.cuda.amp.autocast():
                    output = text_encoder(inputs.input_ids.to(self.device))
                    hidden_len = inputs.attention_mask.sum(dim=-1)-2
                    encoder_hidden_states = output[0]
                    pooled_output = output[1]
                    encoder_hidden_states = encoder_hidden_states.half().cpu().numpy()
                    pooled_output = pooled_output.half().cpu().numpy()
                    hidden_len = hidden_len.cpu().numpy()
                for i in range(pooled_output.shape[0]): 
                    self.embeds.append(pooled_output[i])
                    self.hidden_len.append(hidden_len[i])

            if desc and self.config.cache_path:
                os.makedirs(self.config.cache_path, exist_ok=True)
                cache_filepath = os.path.join(self.config.cache_path, f"{sanitized_name}_{desc}.safetensors")
                tensors = {
                    "embeds": np.stack(self.embeds,dtype=np.float16),
                    "hash": np.array([ord(c) for c in hash], dtype=np.int8),
                    #"encoder_hidden_states": np.stack(self.encoder_hidden_states,dtype=np.float16),
                    "len": np.array(self.hidden_len,dtype=np.int8)
                }
                save_file(tensors, cache_filepath)
                
            end_time = time.time()
            if not config.quiet:
                print(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")
            del tokenizer,text_encoder

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]
            #self.encoder_hidden_states = [e.astype(np.float32) for e in self.encoder_hidden_states]

    def _load_cached(self, desc:str, hash:str, sanitized_name:str) -> bool:
        if self.config.cache_path is None or desc is None:
            return False

        cached_safetensors = os.path.join(self.config.cache_path, f"{sanitized_name}_{desc}.safetensors")             

        if os.path.exists(cached_safetensors):
            try:
                tensors = load_file(cached_safetensors)
            except Exception as e:
                print(f"Failed to load {cached_safetensors}")
                print(e)
                return False
            if 'hash' in tensors and 'embeds' in tensors:
                if np.array_equal(tensors['hash'], np.array([ord(c) for c in hash], dtype=np.int8)):
                    self.embeds = tensors['embeds']
                    #self.encoder_hidden_states = tensors['encoder_hidden_states']
                    self.hidden_len = tensors['len']
                    if len(self.embeds.shape) == 2:
                        self.embeds = [self.embeds[i] for i in range(self.embeds.shape[0])]
                        #self.encoder_hidden_states = [self.encoder_hidden_states[i] for i in range(self.encoder_hidden_states.shape[0])]
                        self.hidden_len = [self.hidden_len[i] for i in range(self.hidden_len.shape[0])]
                    return True

        return False
    

def load_list(data_path: str, filename: Optional[str] = None) -> List[str]:
    """Load a list of strings from a file."""
    if filename is not None:
        data_path = os.path.join(data_path, filename)
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items
