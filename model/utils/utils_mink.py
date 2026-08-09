import torch.nn as nn
import MinkowskiEngine as ME


class ResnetConv3Mink(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 d_out,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1,
                 n_layers_block=3):
        super().__init__()
        if dimension <= 0:
            raise ValueError(
                f"Minkowski residual block dimension must be positive, got {dimension}"
            )

        self.n_layers_block = n_layers_block

        self.conv1 = ME.MinkowskiConvolution(d_in, d_hid, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(d_hid, momentum=bn_momentum)
        
        self.conv2 = ME.MinkowskiConvolution(d_hid, d_hid, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(d_hid, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(d_hid, d_out, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(d_out, momentum=bn_momentum)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        if self.n_layers_block==1:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            return out

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.downsample is not None:
          residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
