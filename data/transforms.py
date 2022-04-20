import random
import math
import torch
import torchvision
import torchvision.transforms.functional as TF


class ToTensor(object):
    def __init__(self, encode_map=False):
        self.encode_map = encode_map
        self.transform = torchvision.transforms.ToTensor()

    @staticmethod
    def encode_segmentation_map(mask):
        labels_map = torch.zeros(mask.shape)
        labels_map[mask > 0] = 1

        return labels_map.to(dtype=torch.int64)

    def __call__(self, sample):
        image, mask = sample
        if self.encode_map:
            return self.transform(image), self.encode_segmentation_map(self.transform(mask))
        else:
            return self.transform(image), self.transform(mask)


class Resize(object):
    def __init__(self, size):
        self.resize = torchvision.transforms.Resize(size,
                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        image, mask = sample
        return self.resize(image), self.resize(mask)


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.flip = lambda image: TF.hflip(image) if random.random() < p else image

    def __call__(self, sample):
        image, mask = sample
        return self.flip(image), self.flip(mask)


class RandomRotation(object):
    def __init__(self, degrees):
        angle = torchvision.transforms.RandomRotation.get_params((-degrees, degrees))
        self.rotate = lambda image: TF.rotate(image, angle)

    def __call__(self, sample):
        image, mask = sample
        return self.rotate(image), self.rotate(mask)


class RandomScale(object):
    def __init__(self, scale):
        self.scale = scale

    def scale(self, image):
        ret = torchvision.transforms.RandomAffine.get_params((0, 0), None, self.scale, None, image.size)
        return TF.affine(image, *ret, resample=False, fillcolor=0)

    def __call__(self, sample):
        image, mask = sample
        return self.scale(image), self.scale(mask)


class BrightContrastJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.transform = torchvision.transforms.ColorJitter(brightness, contrast, 0, 0)

    def __call__(self, sample):
        image, mask = sample
        return self.transform(image), mask


class GaussianNoise(object):
    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation

    @staticmethod
    def noise_overlay(tensor, standard_deviation):
        if type(standard_deviation) is tuple:
            min_value = standard_deviation[0] / 255.0
            max_value = standard_deviation[1] / 255.0
        else:
            min_value = standard_deviation / 255.0
            max_value = standard_deviation / 255.0

        return torch.clamp(tensor +
                           torch.empty_like(tensor).normal_(mean=0.0, std=1.0) *
                           torch.empty_like(tensor).uniform_(min_value, max_value),
                           min=0.0, max=1.0)

    def __call__(self, sample):
        image, clean_image = sample
        return self.noise_overlay(image, self.standard_deviation), clean_image


class RicianNoise(object):
    def __init__(self, variance=(0, 0.1)):
        self.variance = variance

    @staticmethod
    def noise_overlay(tensor, variance):
        if type(variance) is tuple:
            variance = random.uniform(variance[0], variance[1])
        else:
            variance = variance

        return torch.clamp(torch.sqrt(torch.pow(tensor +
                                                torch.empty_like(tensor).normal_(mean=0.0, std=variance), 2) +
                                      torch.pow(torch.empty_like(tensor).normal_(mean=0.0, std=variance), 2)),
                           min=0.0, max=1.0)

    def __call__(self, sample):
        image, clean_image = sample
        return self.noise_overlay(image, self.variance), clean_image


class RandomErasing(object):
    def __init__(self, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def noise_overlay(self, tensor):
        for attempt in range(100):
            area = tensor.shape[1] * tensor.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < tensor.shape[2] and h < tensor.shape[1]:
                x1 = random.randint(0, tensor.shape[1] - h)
                y1 = random.randint(0, tensor.shape[2] - w)
                if tensor.shape[0] == 3:
                    tensor[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    tensor[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    tensor[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    tensor[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return tensor

        return tensor

    def __call__(self, sample):
        image, clean_image = sample
        return self.noise_overlay(image), clean_image
