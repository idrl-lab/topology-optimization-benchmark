import torch.nn as nn
# %%vNet Code
__all__ = ["VNet_user"]


class VNet_user(nn.Module):
    def __init__(self):
        super().__init__(in_channels=1)
        self.main = nn.Sequential(
            # ^ in_channels out _channels kernel_size stride padding
            # ~ 40*40
            # padding=(kernel_size - 1)/2
            nn.Conv2d(1, 16, 1, 1, 0, bias=False),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 20*20
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 10*10
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 5*5
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 10*10
            nn.UpsamplingNearest2d(10),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 20*20
            nn.UpsamplingNearest2d(20),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            # ~ 40*40
            nn.UpsamplingNearest2d(40),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        output = self.main(input)
        return output


def weights_init(m):
    # custom weights initialization
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    test = VNet_user()
    
    '''
    Total params: 1, 741, 984
    Trainable params: 1, 741, 984
    '''
