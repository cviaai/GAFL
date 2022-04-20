import torch
import torch.nn.functional as TF

from models.adaptive_layer import AdaptiveLayer
from models.resnet_blocks import ResBlock


class ResNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, blocks, filters, image_size, adaptive_layer_type=None):
        super(ResNet, self).__init__()

        self.name = 'ResNet'
        if adaptive_layer_type is not None:
            self.name = '_'.join([self.name, 'adaptive', adaptive_layer_type])

        self.adaptive_layer_type = adaptive_layer_type
        if self.adaptive_layer_type is not None:
            self.adaptive_layer = AdaptiveLayer((n_channels, ) + image_size, adjustment=self.adaptive_layer_type)

        self.init_conv = torch.nn.Sequential(torch.nn.Conv2d(n_channels, filters[0],
                                                             kernel_size=7, stride=2, padding=3, bias=False),
                                             torch.nn.BatchNorm2d(filters[0]),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder = torch.nn.ModuleList()
        for i, num_layers in enumerate(blocks):
            if i == 0:
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=filters[i], num_features=filters[i],
                                             downsampling=False))
            else:
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=filters[i - 1], num_features=filters[i],
                                             downsampling=True))

        self.fc = torch.nn.Linear(filters[-1], n_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.adaptive_layer_type is not None:
            x = self.adaptive_layer(x)

        x = self.init_conv(x)

        for layer in self.encoder:
            x = layer(x)

        x = TF.avg_pool2d(x, x.size()[2:])

        return self.fc(x.view(x.size(0), -1))
