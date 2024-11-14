import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
timm.models.swin_transformer

class SwinTransformer(nn.Module):
    def __init__(self, in_channels=3, image_size=224, name='swin_base_patch4_window7_224', pool_method='mean', pretrained=True, load_from=None):
        super().__init__()
        self.name = name
        self.encoder = timm.create_model(name, pretrained=pretrained)
        
        if in_channels != 3:
            # Modify the input embedding layer to accept in_channels
            self.encoder.patch_embed.proj = nn.Conv2d(
                in_channels, self.encoder.patch_embed.proj.out_channels,
                kernel_size=self.encoder.patch_embed.proj.kernel_size,
                stride=self.encoder.patch_embed.proj.stride,
                padding=self.encoder.patch_embed.proj.padding,
                bias=self.encoder.patch_embed.proj.bias is not None
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
            x = self.encoder.forward_features(x)  # [B, H, W, C]
            x = x.view(x.size(0), -1, x.size(-1))  # [B, num_tokens, num_features]
            if self.pool_method == 'mean':
                global_x = x.mean(dim=1)  # Global average pooling over tokens
            elif self.pool_method == 'max':
                global_x = x.max(dim=1)[0]  # Global max pooling over tokens
            else:
                global_x = x[:, 0]  # Return token embeddings without pooling
            if need_token:
                return global_x, x  # Return global feature and token embeddings
            else:
                return global_x
        self.encoder.forward = forward_wrapper
    
    def forward(self, x, need_token=False):
        return self.encoder.forward(x, need_token=need_token)
    
    def get_global_width(self):
        return self.encoder.num_features  # Swin Transformer uses num_features
    
    def get_width(self):
        return self.encoder.num_features
    
    def get_local_width(self):
        return self.encoder.num_features

if __name__ == '__main__':
    model = SwinTransformer(name='swin_base_patch4_window7_224', in_channels=3, pretrained=True, pool_method='mean')
    x = torch.randn(1, 3, 224, 224)
    global_x, token_embeddings = model(x, need_token=True)
    print(global_x.shape)        # Expected shape: [1, num_features]
    print(token_embeddings.shape)  # Expected shape: [1, num_tokens, num_features]
