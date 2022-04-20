import torch
from sklearn.metrics import f1_score


class DiceMetric(object):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    @staticmethod
    def _input_format(predictions, targets):
        if len(targets.shape) == 4:
            predictions = torch.argmax(predictions, dim=1)
        if len(targets.shape) == 4:
            targets = targets.squeeze(dim=1)

        return predictions, targets

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets)

        one_hot_predictions = torch.nn.functional.one_hot(predictions, num_classes=self.n_classes)
        one_hot_predictions = one_hot_predictions.permute(0, 3, 1, 2).to(predictions.dtype)

        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.n_classes)
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).to(predictions.dtype)

        weights = (torch.ones(one_hot_targets.shape[:2]) / self.n_classes).to(predictions.device)

        intersections = torch.sum(one_hot_predictions * one_hot_targets, dim=(2, 3))
        unions = torch.sum(one_hot_predictions + one_hot_targets, dim=(2, 3))

        dice_coefficients = torch.sum(weights * (2 * intersections) / (unions + 1e-16), dim=1)

        return torch.mean(dice_coefficients)


class F1Score(object):
    def __init__(self, average='macro'):
        self.average = average

    @staticmethod
    def _input_format(predictions, targets):
        return torch.argmax(predictions, dim=1).cpu().tolist(), targets.cpu().tolist()

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets)
        return f1_score(y_true=targets, y_pred=predictions, average=self.average)
