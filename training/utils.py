import random
import numpy as np
import torch
from ptflops import get_model_complexity_info


def make_reproducible(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def print_model_info(model, params):
    macs, params = get_model_complexity_info(model, (1,) + params['image_size'],
                                             print_per_layer_stat=False, as_strings=False)
    print(f"Model name: {model.name}")
    print('Computational complexity: ', str(round(macs / 10. ** 6, ndigits=6)) + ' MMac')
    print('Number of parameters: ', str(round(params / 10 ** 6, ndigits=6)) + ' M')
    print()
