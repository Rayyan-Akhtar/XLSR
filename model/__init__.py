import torch
import torch.nn as nn


class BuildingBlock(nn.Module):
    def __init__(self, in_ch=32, out_ch=32, groups=4):
        super(BuildingBlock, self).__init__()
        self.in_ch, self.out_ch, self.groups = in_ch, out_ch, groups
        self.input_conv = nn.Conv2d(self.in_ch, self.out_ch, (3, 3), 1, 1, 1, groups=self.groups, padding_mode="reflect")
        self.activation = nn.ReLU()
        self.output_conv = nn.Conv2d(self.out_ch, self.out_ch, 1, 1)
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.activation(x)
        x = self.output_conv(x)
        return x


class XLSR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XLSR, self).__init__()
        self.in_ch, self.out_ch = in_ch, out_ch

        self.project = nn.Conv2d(3, self.in_ch, (3, 3), 1, 1, padding_mode="reflect")

        self.skip = nn.Conv2d(3, self.in_ch//2, (3, 3), 1, 1, padding_mode="reflect")

        self.blocks = nn.ModuleList([
            BuildingBlock(self.in_ch, self.in_ch),
            BuildingBlock(self.in_ch, self.in_ch),
            BuildingBlock(self.in_ch, self.in_ch),
            BuildingBlock(self.in_ch, self.in_ch)
        ])

        self.conv1x1 = nn.Conv2d(self.in_ch + self.in_ch//2, self.in_ch, 1, 1, padding_mode="reflect")

        self.activation = nn.ReLU()

        self.conv = nn.Conv2d(self.in_ch, self.in_ch+self.in_ch//2, (3, 3), 1, 1, padding_mode="reflect")

        self.depth2space = nn.PixelShuffle(upscale_factor=4)

        self.clipped_relu = lambda x: torch.clamp(x, -1, 1)

    def forward(self, x):
        input_tensor = x

        x = self.project(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        
        skip = self.skip(input_tensor)

        concatenate = torch.cat([x, skip], dim=1)

        conv1x1 = self.conv1x1(concatenate)

        relu = self.activation(conv1x1)

        conv = self.conv(relu)

        depth2space = self.depth2space(conv)

        clipped_relu = self.clipped_relu(depth2space)

        return clipped_relu

