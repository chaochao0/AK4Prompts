import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout

from typing import Optional
from torch import Tensor

class MultiAttention(nn.Module):
    def __init__(self,sd_text_size=768):
        super(MultiAttention, self).__init__()
        self.cross_attn = MultiheadAttention(embed_dim=sd_text_size, num_heads=8, dropout=0.1, batch_first=True)
        self.linear1 = nn.Linear(sd_text_size, 1024)
        self.dropout = Dropout(0.1)
        self.linear2 = nn.Linear(1024, sd_text_size)

        self.norm1 = LayerNorm(sd_text_size)
        self.norm2 = LayerNorm(sd_text_size)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

        self.activation = nn.ReLU()

    def forward(self,q,k,v,attn_mask=None,key_padding_mask=None):

        x = self.norm1(q + self._sa_block(q,k,v, attn_mask, key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

    # cross-attention block
    def _sa_block(self, q: Tensor,k: Tensor,v: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.cross_attn(q, k, v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class AK4Prompts(nn.Module):
    def __init__(self):
        super(AK4Prompts, self).__init__()

        self.attention1 = MultiAttention()
        self.attention2 = MultiAttention()
        self.attention3 = MultiAttention()

        self.aesthetic_project_layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )
        self.clip_project_layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )
        self.hps_project_layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )


    def forward(self,prompt_encoder_hidden_states,prompt_attention_mask,keywords_embs):
        out_attention = self.attention1(q=keywords_embs, k=prompt_encoder_hidden_states, v=prompt_encoder_hidden_states,key_padding_mask=prompt_attention_mask)
        out_attention = self.attention2(q=out_attention, k=prompt_encoder_hidden_states, v=prompt_encoder_hidden_states,key_padding_mask=prompt_attention_mask)
        out_attention = self.attention3(q=out_attention, k=prompt_encoder_hidden_states, v=prompt_encoder_hidden_states,key_padding_mask=prompt_attention_mask)

        aesthetic_out_fc = (self.aesthetic_project_layers(out_attention)).squeeze(dim=-1)  ### batch_size, 1000
        clip_out_fc = (self.clip_project_layers(out_attention)).squeeze(dim=-1)
        hps_out_fc = (self.hps_project_layers(out_attention)).squeeze(dim=-1)
        return aesthetic_out_fc,clip_out_fc,hps_out_fc
    
        