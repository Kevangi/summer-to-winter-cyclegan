import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            layers = [
                nn.Conv2d(in_filters, out_filters,
                          kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        model = []
        model += discriminator_block(in_channels, 64, stride=2)
        model += discriminator_block(64, 128)
        model += discriminator_block(128, 256)
        model += discriminator_block(256, 512, stride=1)

        model += [
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)