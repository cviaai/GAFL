import os

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils import get_params, parse_args

from piq import MultiScaleSSIMLoss
from piq import psnr, ssim

from data.transforms import ToTensor, Resize, GaussianNoise, RandomErasing
from data.dataset import DenoisingDataset, data_loaders

from models.denoising import DnCNN

from training.utils import make_reproducible, print_model_info
from training.model_training import train
from training.losses import CombinedLoss

# for printing
torch.set_printoptions(precision=2)

# for reproducibility
make_reproducible(seed=0)


def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    params = get_params()
    params.update({'random_seed': args.random_seed})

    train_transform = torchvision.transforms.Compose([Resize(size=params['image_size']),
                                                      ToTensor(),
                                                      RandomErasing()])
                                                      # GaussianNoise((0.0, 50.0))])
    val_transform = torchvision.transforms.Compose([Resize(size=params['image_size']),
                                                    ToTensor(),
                                                    RandomErasing()])
                                                    # GaussianNoise(25.0)])
    train_dataloader, val_dataloader = data_loaders(dataset=DenoisingDataset,
                                                    train_transform=train_transform,
                                                    val_transform=val_transform,
                                                    params=params)

    model = DnCNN(n_channels=1,
                  num_features=params['num_features'], num_layers=params['num_layers'],
                  image_size=params['image_size'], adaptive_layer_type=params['adaptive_layer']).to(device)
    print_model_info(model, params)

    criterion = CombinedLoss([MultiScaleSSIMLoss(), torch.nn.L1Loss()], [0.8, 0.2])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = {'PSNR': psnr, 'SSIM': ssim}

    writer = None
    if not args.nolog:
        writer = SummaryWriter(log_dir=os.path.join(params['log_dir'], model.name))
        print("To see the learning process, use command in the new terminal:\n" +
              "tensorboard --logdir <path to log directory>")
        print()

    train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs=params['n_epochs'],
          device=device,
          writer=writer)


if __name__ == "__main__":
    main(parse_args())
