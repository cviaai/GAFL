import torch


class CombinedLoss(torch.nn.Module):
    def __init__(self, losses, weights):
        super(CombinedLoss, self).__init__()

        self.losses = losses
        self.weights = [weight/sum(weights) for weight in weights]

    def forward(self, predictions, targets):
        loss = 0.0
        for loss_function, weight in zip(self.losses, self.weights):
            loss += weight * loss_function(predictions, targets)

        return loss


class MultilabelDiceLoss(torch.nn.Module):
    def __init__(self, n_classes=2):
        super(MultilabelDiceLoss, self).__init__()

        self.n_classes = n_classes
        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def _input_format(targets):
        if len(targets.shape) == 4:
            targets = targets.squeeze(dim=1)

        return targets

    def forward(self, predictions, targets):
        predictions = self.softmax(predictions)
        targets = self._input_format(targets)

        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.n_classes)
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).to(predictions.dtype)

        weights = (torch.ones(one_hot_targets.shape[:2]) / self.n_classes).to(predictions.device)

        intersections = torch.sum(predictions * one_hot_targets, dim=(2, 3))
        unions = torch.sum(torch.pow(predictions, 2) + torch.pow(one_hot_targets, 2), dim=(2, 3))

        dice_coefficients = torch.sum(weights * (2 * intersections) / (unions + 1e-16), dim=1)

        return 1 - torch.mean(dice_coefficients)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    @staticmethod
    def _input_format(targets):
        if len(targets.shape) == 4:
            targets = targets.squeeze(dim=1)

        return targets

    def forward(self, predictions, targets):
        targets = self._input_format(targets)
        return self.loss(predictions, targets)
