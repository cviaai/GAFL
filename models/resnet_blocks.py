import torch


class ResLayer(torch.nn.Module):
    def __init__(self, num_input_features, num_features, downsample):
        super(ResLayer, self).__init__()

        self.downsample = downsample

        stride = 1
        if self.downsample is not None:
            stride = 2
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(num_input_features, num_features,
                                                          kernel_size=3, stride=stride, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(num_features))

        self.relu = torch.nn.ReLU(inplace=True)

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(num_features, num_features,
                                                          kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(num_features))

    def forward(self, x):
        residual = x

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual

        return self.relu(x)


class ResBlock(torch.nn.Sequential):
    def __init__(self, num_layers, num_input_features, num_features, downsampling):
        super(ResBlock, self).__init__()

        self.downsample = None
        if downsampling:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(num_input_features, num_features,
                                                                  kernel_size=1, stride=2, bias=False),
                                                  torch.nn.BatchNorm2d(num_features))

        self.res_block = torch.nn.ModuleList()
        for i in range(num_layers):
            self.res_block.append(ResLayer(num_input_features, num_features, self.downsample))
            num_input_features = num_features
            self.downsample = None

    def forward(self, x):
        for layer in self.res_block:
            x = layer(x)
        return x
