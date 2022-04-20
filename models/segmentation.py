import torch
from models.adaptive_layer import AdaptiveLayer, GeneralAdaptiveLayer
from models.unet_blocks import double_conv, out_conv, down_step, up_step


class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, init_features, depth, image_size, adaptive_layer_type=None):
        super(UNet, self).__init__()

        self.name = 'UNet'
        if adaptive_layer_type is not None:
            self.name = '_'.join([self.name, 'adaptive', adaptive_layer_type])

        self.features = init_features
        self.depth = depth

        self.adaptive_layer_type = adaptive_layer_type
        if self.adaptive_layer_type in ('spectrum', 'spectrum_log', 'phase'):
            self.adaptive_layer = AdaptiveLayer((n_channels, ) + image_size,
                                                adjustment=self.adaptive_layer_type)
        elif self.adaptive_layer_type == 'general_spectrum':
            self.adaptive_layer = GeneralAdaptiveLayer((n_channels, ) + image_size,
                                                       adjustment=self.adaptive_layer_type,
                                                       activation_function_name='relu')

        self.down_path = torch.nn.ModuleList()
        self.down_path.append(double_conv(n_channels, self.features, self.features))
        for i in range(1, self.depth):
            self.down_path.append(down_step(self.features, 2 * self.features))
            self.features *= 2

        self.up_path = torch.nn.ModuleList()
        for i in range(1, self.depth):
            self.up_path.append(up_step(self.features, self.features // 2))
            self.features //= 2
        self.out_conv = out_conv(self.features, n_classes)

    def forward_down(self, input):
        downs = [input]
        for down_step in self.down_path:
            downs.append(down_step(downs[-1]))

        return downs

    def forward_up(self, downs):
        current_up = downs[-1]
        for i, up_step in enumerate(self.up_path):
            current_up = up_step(current_up, downs[-2 - i])

        return current_up

    def forward(self, x):
        if self.adaptive_layer_type is not None:
            x = self.adaptive_layer(x)

        downs = self.forward_down(x)
        up = self.forward_up(downs)

        return self.out_conv(up)
