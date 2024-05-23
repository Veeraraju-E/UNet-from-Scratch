# U-Net Implementation for Binary Image Segmentation
# original U-Net decoder section has un-padded convolutional layers + ReLU
# 1. Import dependencies
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# at every stage of the architecture, we have like two convs + ReLU layers, followed by down-sampling
# to avoid repetitively forming the two convs every time, let's just create a class for that and use its instance later


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # output h, w same as input; also we set the bias to False, as we are going to use BatchNorm2d
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# keep in mind that we have to also perform the max pooling of (2,2) with stride 2 for the down-sampling (by half) task
# let's now start defining the class for the overall model
class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 ):
        super(UNet, self).__init__()
        # to store all the down sampling layers, we need a list;
        # but we can't use a simple list, as we need to do model.eval() and batch normalization
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.filters = [64, 128, 256, 512]

        # Down sampling architecture
        for f in self.filters:
            self.downs.append(DoubleConv(in_channels, f))   # in the first iteration, we map in_channels = 3 to 64
            in_channels = f
        # so, in our self.downs list, we basically have all the double convolutional layers of the down sampling section
        # refer to the annotated image for better understanding

        for f in reversed(self.filters):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            # this doubles the h, w as it's ConvTranspose2d
            self.ups.append(DoubleConv(f*2, f))  # this is the 2 Conv layers after doubling

        # so, similarly we have all the double convolutional layers of the up sampling section
        # we'll add the pooling layers in the forward pass

        # finally, the bottleneck layers
        self.bottleneck = DoubleConv(self.filters[-1], self.filters[-1]*2)    # 512 -> 1024
        self.final_conv = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []   # store all the skip connection layers
        # print('in forward', x.shape)
        for down in self.downs:
            # print(down)
            x = down(x)
            # print('after down', x.shape)
            skip_connections.append(x)
            x = self.pool(x)
            # print('after pooling', x.shape)

        # basically, x is now the left-most layer in the bottleneck
        x = self.bottleneck(x)
        # print('after bottleneck', x.shape)
        skip_connections = skip_connections[::-1]  # reverse the order

        # now, for the up sampling layers
        # in our self.ups, we have the ConvTranspose2d layer, followed by the Double Conv layers
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # what can happen here is that the h/w of the skip_connection is greater than that of x, because max pooling
            # layers (2x2) always floor the dimensions, hence it's imp to crop the skip_connection/ add padding to x etc
            # here, we take a relatively simpler approach -> resizing
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])    # only the height and width are going to be changed
            # now, concatenate along the channel dim - (B, C, H, W) - hence along dim = 1
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(1, 1)
    predictions = model(x)
    assert predictions.shape == x.shape


if __name__ == '__main__':
    test()
