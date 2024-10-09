import torch
import torch.nn as nn
import torch.nn.functional as F
BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class Upscaler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upscaler, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu1 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu3 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu1(x)
        x = self.conv3_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.upsample2(x)
        x = self.relu3(x)
        x = self.conv3_2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UNET(nn.Module):
    def __init__(self,depth=True):
        super(UNET, self).__init__()
        
        # Upsample modules
        self.upsample_modules = nn.ModuleDict({
            'upsample_3_to_2': self._make_upsample_module(384, 192),
            'upsample_2_to_1': self._make_upsample_module(192, 96),
            'upsample_1_to_0': self._make_upsample_module(96, 48),
        })
        
        # Fusion and refinement modules
        self.fusion_and_refinement = nn.ModuleDict({
            'fuse_and_refine_3': self._make_layer(BasicBlock, 384, 192, 4),
            'fuse_and_refine_2': self._make_layer(BasicBlock, 192, 96, 4),
            'fuse_and_refine_1': self._make_layer(BasicBlock, 96, 48, 4),
        })

        # Final layers to produce the depth map
        if depth:
            self.final_layers = nn.Sequential(
                nn.Conv2d(48,48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False),
                nn.Conv2d(48,1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            self.final_layers = nn.Sequential(
                nn.Conv2d(48,48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False),
                nn.Conv2d(48,23, kernel_size=1, stride=1, padding=0),
            )

        self.upscaling = Upscaler(48, 48)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpol_feature = nn.AdaptiveAvgPool2d(2)



    def _make_layer(self, block, input_channels, output_channels, num_blocks, stride=1):
        layers = []
        downsample = None
        if input_channels != output_channels * block.expansion or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels * block.expansion, momentum=BN_MOMENTUM),
            )

        # First block with potential downsample
        layers.append(block(input_channels, output_channels, stride, downsample))

        # No downsampling for subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(block(output_channels * block.expansion, output_channels))

        return nn.Sequential(*layers)

    def _make_upsample_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, features):
        # Assume 'features' is a list of feature maps from the backbone
        features = {'branch1': features[0], 'branch2': features[1], 'branch3': features[2], 'branch4': features[3]}      
        upsampled_feature = features['branch4']
        for branch in ['branch3', 'branch2', 'branch1']:
            upsample_module = self.upsample_modules[f'upsample_{branch[-1]}_to_{int(branch[-1])-1}']
            upsampled_feature = upsample_module(upsampled_feature)
            
            fused_feature = torch.cat([upsampled_feature, features[branch]], dim=1)
            refine_module = self.fusion_and_refinement[f'fuse_and_refine_{branch[-1]}']
            upsampled_feature = refine_module(fused_feature)

        depth_map = self.upscaling(upsampled_feature)
        depth_map = self.final_layers(depth_map)
        depth_features = upsampled_feature
        
        return depth_map,depth_features

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            else:
                pass