import os
import yaml
import argparse


def get_params():
    with open(os.path.join("configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)

    params = {}
    params.update({'data_path': configs['paths']['data'],
                   'log_dir': configs['paths']['log_dir']})

    params.update({'n_classes': int(configs['data_parameters']['n_classes']),
                   'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size'])})

    params.update({'adaptive_layer': configs['model_parameters']['AdaptiveLayer']['adjustment']})
    if params['adaptive_layer'] == 'None':
        params['adaptive_layer'] = None

    params.update({'init_features': int(configs['model_parameters']['segmentation']['UNet']['init_features']),
                   'depth': int(configs['model_parameters']['segmentation']['UNet']['depth'])})

    params.update({'blocks': tuple(map(int, configs['model_parameters']['classification']['ResNet']['blocks'].split(', '))),
                   'filters': tuple(map(int, configs['model_parameters']['classification']['ResNet']['filters'].split(', ')))})

    params.update({'num_features': int(configs['model_parameters']['denoising']['DnCNN']['num_features']),
                   'num_layers': int(configs['model_parameters']['denoising']['DnCNN']['num_layers'])})

    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})

    return params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', type=int, default=0, help='cuda device id number')
    parser.add_argument('-rs', '--random_seed', type=int, default=1, help='random seed')
    parser.add_argument('--nolog', action='store_true', help='turn off logging')

    return parser.parse_args()
