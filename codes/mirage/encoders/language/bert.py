from requests import models
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig, AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import sys
import torch.nn.functional as F
sys.path.append(os.getcwd())


class AttentionPool2d(nn.Module):
    def __init__(self, max_length: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(max_length + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]





class ClinicalBert(nn.Module):
    def __init__(self, pretrained=True, vocab_size=28996, pool_method='cls', max_length=77,  pretrained_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.encoder = BertModel(BertConfig(hidden_size=768, vocab_size=vocab_size))
        if pool_method == 'mean':
            self.pool = lambda x: torch.mean(x['last_hidden_state'], dim=[1])
        elif pool_method == 'cls':
            self.pool = lambda x: x['pooler_output'] 
        elif pool_method == 'attention':
            self.atten_pool = AttentionPool2d(max_length=max_length, embed_dim=768, num_heads=8)
            self.pool = lambda x: self.atten_pool(x['last_hidden_state'])
        elif pool_method is None:
            self.pool = lambda x: x
        self.max_length = max_length
        
        
    def forward(self, x):
        features = self.encoder(**x)
        #print(features.shape)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def get_width(self):
        return self.encoder.config.hidden_size

class BioMedBert(nn.Module):
    def __init__(self, pretrained=True, vocab_size=30522, pool_method='cls', max_length=77, pretrained_name="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
        else:
            self.encoder = BertModel(BertConfig(hidden_size=768, vocab_size=vocab_size))
        if pool_method == 'mean':
            self.pool = lambda x: torch.mean(x['last_hidden_state'], dim=[1])
        elif pool_method == 'cls':
            self.pool = lambda x: x['pooler_output'] 
        elif pool_method == 'attention':
            self.atten_pool = AttentionPool2d(max_length=max_length, embed_dim=768, num_heads=8)
            self.pool = lambda x: self.atten_pool(x['last_hidden_state'])
        elif pool_method is None:
            self.pool = lambda x: x
        self.max_length = max_length
        
    def forward(self, x):
        features = self.encoder(**x)
        #print(features.shape)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def get_width(self):
        return self.encoder.config.hidden_size
    



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceBioMedBert(nn.Module):
    def __init__(self, pretrained=True, vocab_size=30522, pool_method='mean', max_length=77, pretrained_name="kamalkraj/BioSimCSE-BioLinkBERT-BASE"):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained("kamalkraj/BioSimCSE-BioLinkBERT-BASE")
        else:
            self.encoder = BertModel(BertConfig(hidden_size=768, vocab_size=vocab_size))
        if pool_method == 'mean':
            self.pool = lambda x, y: mean_pooling(x, y).view(y.size(0), -1)
        else:
            self.pool = lambda x, y: x['last_hidden_state']
        self.max_length = max_length
        
    def forward(self, x, need_token=False):
        features = self.encoder(**x)
        #print(features.shape)
        #features = self.pool(features)

        pool_features = self.pool(features, x['attention_mask'])
        if need_token:
            return pool_features, features
        return pool_features
    
    def get_width(self):

        return self.encoder.config.hidden_size


