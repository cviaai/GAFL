from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import sys
import os
import random
import numpy as np
import pandas as pd


class UltrasoundDataset(object):
    def __init__(self, data_path, val_size=0.2, random_seed=1):
        self.data_path = data_path
        self.dataset = os.path.basename(os.path.normpath(self.data_path))
        self.dataset_table_path = self.dataset + '.csv'

        self.make_dataset_table()
        self.train_val_split(val_size, random_seed=random_seed)

    def make_dataset_table(self):
        print('dataset csv table creating...')
        data = []

        if self.dataset not in ['Endocrinology', 'BUSI', 'BPUI']:
            print('The folder with the dataset must have one of these names: Endocrinology, BUSI, BPUI')
            sys.exit()

        if self.dataset == 'Endocrinology':
            classes_dict = {}
            for _, row in pd.read_csv(self.data_path + os.sep + 'TIRADS.txt').iterrows():
                if classes_dict.get(row['Patient']) is None:
                    classes_dict[row['Patient']] = {row['file']: row['TIRADS']}
                else:
                    classes_dict[row['Patient']].update({row['file']: row['TIRADS']})

            for patient_path in [os.path.join(self.data_path, patient_name)
                                 for patient_name in sorted(os.listdir(self.data_path))
                                 if os.path.isdir(os.path.join(self.data_path, patient_name))]:
                images = [os.path.splitext(image_id)[0]
                          for image_id in sorted(os.listdir(os.path.join(patient_path, 'Images')))]
                for image_id in images:
                    image_path = os.path.join(patient_path, 'Images', '{}.tif'.format(image_id))
                    mask_path = os.path.join(patient_path, 'Masks', '{}.labels.tif'.format(image_id))

                    mask = Image.open(mask_path)
                    frames = [frame for frame in range(mask.n_frames)]
                    classes = []
                    for frame in frames:
                        mask.seek(frame)
                        if not np.array(mask).any():
                            classes.append(0)
                        else:
                            classes.append(classes_dict[os.path.split(patient_path)[-1]][image_id])

                    image_paths = [image_path for _ in range(len(frames))]
                    mask_paths = [mask_path for _ in range(len(frames))]

                    data.append(np.array([image_paths, mask_paths, frames, classes]).T)

            pd.DataFrame(np.concatenate(data),
                         columns=['image', 'mask', 'frame', 'class']).to_csv(self.dataset_table_path, index=False)
        elif self.dataset == 'BUSI':
            classes = sorted([name for name in os.listdir(self.data_path)])
            classes_dict = dict((class_type, index) for index, class_type in enumerate(classes))
            for class_type in classes:
                class_type_path = os.path.join(self.data_path, class_type)
                image_paths = sorted([os.path.join(class_type_path, name) for name in os.listdir(class_type_path)
                                      if 'mask' not in name])

                for image_path in image_paths:
                    mask_path = ''.join([os.path.splitext(image_path)[0], '_mask.png'])
                    data.append(np.array([[image_path, mask_path, classes_dict[class_type]]]))

            pd.DataFrame(np.concatenate(data),
                         columns=['image', 'mask', 'class']).to_csv(self.dataset_table_path, index=False)
        elif self.dataset == 'BPUI':
            image_names = sorted([name for name in os.listdir(self.data_path) if 'mask' not in name])
            for image_name in image_names:
                image_path = os.path.join(self.data_path, image_name)
                mask_path = os.path.join(self.data_path, ''.join([os.path.splitext(image_name)[0], '_mask.tif']))
                data.append(np.array([[image_path, mask_path]]))

            pd.DataFrame(np.concatenate(data), columns=['image', 'mask']).to_csv(self.dataset_table_path, index=False)

    def train_val_split(self, val_size=0.2, random_seed=1):
        dataset_table = pd.read_csv(self.dataset_table_path)

        test_number = int(len(dataset_table) * val_size) + 1
        train_number = len(dataset_table) - test_number
        phase = ['train'] * train_number + ['val'] * test_number
        random.Random(random_seed).shuffle(phase)

        pd.concat([dataset_table,
                   pd.DataFrame(phase, columns=['phase'])], axis=1).to_csv(self.dataset_table_path, index=False)


class BaseDataset(Dataset):
    def __init__(self, dataset_table_path, phase='train', transform=None):
        super(BaseDataset, self).__init__()

        self.dataset = os.path.splitext(dataset_table_path)[0]

        dataset_table = pd.read_csv(dataset_table_path)
        self.dataset_table = dataset_table[dataset_table['phase'] == phase]

        self.transform = transform

    def read_data(self, index, data_type='image'):
        if data_type == 'class':
            return self.dataset_table.iloc[index]['class']

        image = Image.open(self.dataset_table.iloc[index][data_type])
        if self.dataset == 'Endocrinology':
            image.seek(self.dataset_table.iloc[index]['frame'])
        return image.convert('L')

    def do_transform(self):
        return self.transform is not None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_table)


class SegmentationDataset(BaseDataset):
    def __init__(self, dataset_table_path, phase='train', transform=None):
        super(SegmentationDataset, self).__init__(dataset_table_path, phase, transform)

    def __getitem__(self, index):
        image = self.read_data(index, 'image')
        mask = self.read_data(index, 'mask')

        if self.do_transform():
            image, mask = self.transform((image, mask))

        return image, mask


class ClassificationDataset(BaseDataset):
    def __init__(self, dataset_table_path, phase='train', transform=None):
        super(ClassificationDataset, self).__init__(dataset_table_path, phase, transform)

    def __getitem__(self, index):
        image = self.read_data(index, 'image')
        class_type = self.read_data(index, 'class')

        if self.do_transform():
            image = self.transform(image)

        return image, class_type


class DenoisingDataset(BaseDataset):
    def __init__(self, dataset_table_path, phase='train', transform=None):
        super(DenoisingDataset, self).__init__(dataset_table_path, phase, transform)

    def __getitem__(self, index):
        image = self.read_data(index, 'image')
        clean_image = self.read_data(index, 'image')

        if self.do_transform():
            image, clean_image = self.transform((image, clean_image))

        return image, clean_image


def data_loaders(dataset, train_transform, val_transform, params):
    data = UltrasoundDataset(data_path=params['data_path'], val_size=0.2, random_seed=params['random_seed'])

    train_dataset = dataset(dataset_table_path=data.dataset_table_path,
                            phase='train',
                            transform=train_transform)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=params['batch_size'],
                                  shuffle=True,
                                  num_workers=4)

    val_dataset = dataset(dataset_table_path=data.dataset_table_path,
                          phase='val',
                          transform=val_transform)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=params['batch_size'],
                                num_workers=4)

    return train_dataloader, val_dataloader
