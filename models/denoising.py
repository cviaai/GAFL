import torch
from models.adaptive_layer import AdaptiveLayer


class DnCNN(torch.nn.Module):
    def __init__(self, n_channels, num_features, num_layers, image_size, adaptive_layer_type=None):
        super(DnCNN, self).__init__()

        self.name = 'DnCNN'
        if adaptive_layer_type is not None:
            self.name = '_'.join([self.name, 'adaptive', adaptive_layer_type])

        self.adaptive_layer_type = adaptive_layer_type
        if self.adaptive_layer_type is not None:
            self.adaptive_layer = AdaptiveLayer((n_channels, ) + image_size, adjustment=self.adaptive_layer_type)

        layers = [torch.nn.Sequential(torch.nn.Conv2d(n_channels, num_features, kernel_size=3, stride=1, padding=1),
                                      torch.nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(torch.nn.Sequential(torch.nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                              torch.nn.BatchNorm2d(num_features),
                                              torch.nn.ReLU(inplace=True)))

        layers.append(torch.nn.Conv2d(num_features, n_channels, kernel_size=3, padding=1))
        self.layers = torch.nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input):
        x = input
        if self.adaptive_layer_type is not None:
            x = self.adaptive_layer(x)

        residual = self.layers(x)

        return torch.sigmoid(input - residual)
