import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import os

class ViT(nn.Module):
    def __init__(self, in_channels=3, image_size=224, name='vit_base_patch16_224', pool_method='cls', pretrained=True, load_from=None):
        super().__init__()
        self.name = name
        self.encoder = timm.create_model(name, pretrained=pretrained)
        
        if in_channels != 3:
            # Modify the input embedding layer to accept in_channels
            # ViT models have a patch embedding layer
            self.encoder.patch_embed.proj = nn.Conv2d(
                in_channels, self.encoder.patch_embed.proj.out_channels,
                kernel_size=self.encoder.patch_embed.proj.kernel_size,
                stride=self.encoder.patch_embed.proj.stride,
                padding=self.encoder.patch_embed.proj.padding,
                bias=False
            )
        
        if load_from is not None:
            old_state_dict = torch.load(load_from, map_location='cpu')['state_dict']
            state_dict = {}
            for k, v in old_state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k.replace('backbone.', '')] = v
            self.encoder.load_state_dict(state_dict, strict=False)
        
        self.pool_method = pool_method
        self._modify_forward()
    
    def _modify_forward(self):
        def forward_wrapper(x, need_token=False):
            x = self.encoder.patch_embed(x)
            cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
            if self.encoder.pos_embed is not None:
                x = x + self.encoder.pos_embed[:, 1:, :]
            x = torch.cat((cls_token, x), dim=1)
            if self.encoder.pos_embed is not None:
                x = x + self.encoder.pos_embed[:, :x.shape[1], :]
            x = self.encoder.pos_drop(x)
            x = self.encoder.blocks(x)
            x = self.encoder.norm(x)
            
            if self.pool_method == 'cls':
                global_x = x[:, 0]
            elif self.pool_method == 'mean':
                global_x = x.mean(dim=1)
            else:
                # return the token embeddings without pooling
                global_x = x[:, 1:]
            if need_token:
                return global_x, x[:, 1:]
            else:
                return global_x
        self.encoder.forward = forward_wrapper
    
    def forward(self, x, need_token=False):
        return self.encoder.forward(x, need_token=need_token)
    
    def get_global_width(self):
        return self.encoder.embed_dim
    
    def get_width(self):
        return self.encoder.embed_dim
    
    def get_local_width(self):
        return self.encoder.embed_dim

if __name__ == '__main__':
    model = Vit(name='vit_base_patch16_224', in_channels=1, pretrained=True, pool_method='cls')
    x = torch.randn(1, 1, 224, 224)
    global_x, token_embeddings = model(x, need_token=True)
    print(global_x.shape)
    print(token_embeddings.shape)
