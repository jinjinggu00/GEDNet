import torch
import torch.nn as nn
import timm
import math
from timm.models.convnext import ConvNeXtBlock
from .fdconv import FDConv

class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # -> (B, C, H, W)
        return x

class HybridStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, downsample=True, use_fdconv=True):
        super().__init__()
        self.use_fdconv = use_fdconv
        self.downsample = nn.Sequential(
            LayerNormChannelsFirst(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        ) if downsample else nn.Identity()
        
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(in_chs=out_channels) for _ in range(depth)]
        )
        
        if self.use_fdconv and FDCONV_AVAILABLE:
            self.fdconv_block = FDConv(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=3, padding=1,
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.convnext_blocks(x)
        if self.use_fdconv and FDCONV_AVAILABLE:
            x = self.fdconv_block(x)
        return x

class GatingModuleHybrid(nn.Module):
    def __init__(self, num_classes=4, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 use_fdconv_stages=[True, True, True, True]):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNormChannelsFirst(dims[0])
        )

        self.stages = nn.ModuleList()
        in_channels = dims[0]
        for i in range(4):
            stage = HybridStage(
                in_channels=in_channels,
                out_channels=dims[i],
                depth=depths[i],
                downsample=(i > 0),
                use_fdconv=use_fdconv_stages[i]
            )
            self.stages.append(stage)
            in_channels = dims[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class GatingModuleDeepFDConv(nn.Module):
    def __init__(self, model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', num_classes=4):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False, 
            num_classes=0,
            global_pool=''
        )
        
        backbone_out_channels = self.backbone.num_features
        
        if FDCONV_AVAILABLE:
            self.attention = FDConv(
                in_channels=backbone_out_channels,
                out_channels=backbone_out_channels,
                kernel_size=3,
                padding=1
            )
        else:
            self.attention = nn.Identity()
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(backbone_out_channels, eps=1e-6),
            nn.Linear(backbone_out_channels, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.head(x)
        return x

class GatingModulePosAblation(nn.Module):
    """
    insert_stage = 0 
    insert_stage = 1 
    insert_stage = 2 
    insert_stage = 3 
    """
    def __init__(self, model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', num_classes=4, insert_stage=3):
        super().__init__()
        self.insert_stage = insert_stage
        
        base_model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        
        modules_part1 = [base_model.stem]
        for i in range(insert_stage + 1):
            modules_part1.append(base_model.stages[i])
        self.backbone_part1 = nn.Sequential(*modules_part1)
        
        modules_part2 =[]
        for i in range(insert_stage + 1, 4):
            modules_part2.append(base_model.stages[i])
        self.backbone_part2 = nn.Sequential(*modules_part2)
        
        dims =[96, 192, 384, 768]
        mid_channels = dims[insert_stage]
        
        if FDCONV_AVAILABLE:
            self.attention = FDConv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        else:
            self.attention = nn.Identity()
            
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.backbone_part1(x)
        x = self.attention(x)
        x = self.backbone_part2(x)
        x = self.head(x)
        return x
      
def create_gating_model(model_name: str, num_classes: int = 4, pretrained: bool = True, img_size: int = 224):
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes,
        img_size=img_size
    )
    return model
