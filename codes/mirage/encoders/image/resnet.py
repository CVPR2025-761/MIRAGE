
from pyexpat import features
import timm
import torch.nn as nn
from timm.models import resnet50d
from torchvision.models import resnet50
from timm.models.layers import StdConv2d
import torch
import torch.nn.functional as F
import sys
import os




class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
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


class ResNet(nn.Module):
    def __init__(self, in_channels=3, image_size=224, name=None, pool_method='attention', pretrained=True, load_from=None) -> None:
        super().__init__()
        self.name = name
        self.spacial_dim = image_size // 32
        if name == 'resnet50':
            self.encoder = resnet50_224(in_channels=in_channels, pretrained=pretrained, load_from=load_from)
        if pool_method == 'mean':
            self.pool = lambda x: torch.mean(x, dim=[2, 3]).view(x.size(0), -1)
        elif pool_method == 'attention':
            self.atten_pool = AttentionPool2d(spacial_dim=self.spacial_dim , embed_dim=2048, num_heads=8)
            self.pool = lambda x: self.atten_pool(x).view(x.size(0), -1)
        else:
            # view to b h*w c
            self.pool = lambda x: x.view(x.size(0), -1, x.size(1))
        self._modify_forward()
        

    def _modify_forward(self):
        if self.name == 'resnet50':
            def forward_wrapper(x, need_token=False):
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.relu(x)
                x = self.encoder.maxpool(x)
                x = self.encoder.layer1(x)
                x = self.encoder.layer2(x)
                x = self.encoder.layer3(x)
                x = self.encoder.layer4(x)
                global_x = self.pool(x)
                #global_x = global_x.view(global_x.size(0), -1)
                if need_token:
                    return global_x, x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
                return global_x
        else:
            raise NotImplementedError
        self.encoder.forward = forward_wrapper

        
    def forward(self, x, need_token=False):
        try:
            return self.encoder.forward_features(x, need_token=need_token)
        except:
            return self.encoder.forward(x, need_token=need_token)
    

    def get_global_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4

    def get_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 

    def get_local_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 
          


          

def resnet50_224(in_channels=3, load_from=None, **kwargs):
    model =  resnet50(**kwargs)
    if in_channels != 3:
        old_conv = model.conv1
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.conv1.weight = torch.nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
        model.fc = nn.Identity()
    if load_from is not None:
        old_state_dict = torch.load(load_from, map_location='cpu')['state_dict']
        state_dict = {}
        for k, v in old_state_dict.items():
            if k.startswith('backbone.'):
                state_dict[k.replace('backbone.', '')] = v
        model.load_state_dict(state_dict, strict=False)
    return model 

if __name__ == '__main__':
    model = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='attention')
    x = torch.randn(1, 1, 224, 224)
    _, _, features = model.encoder(x, return_features=True)
    for f in features:
        print(f.shape)