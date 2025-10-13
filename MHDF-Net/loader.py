import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
import time
from imgaug import augmenters as iaa
import numpy as np
import torchvision
import config
class ImgTrainTransform:

    def __init__(self, size=config.img_size, normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale(size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),

            # noise
            iaa.Sometimes(0.1,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)),
                              iaa.CoarseDropout(0.02, size_percent=0.25)
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # brightness
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                          ])),

        ])

    def __call__(self, img):
        img = self.aug.augment_image(np.array(img)).copy()
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)


class ImgEvalTransform:

    def __init__(self, size=config.img_size, normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)


from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


class MyDataset (data.Dataset):
    """
    This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, imgs_path, labels, meta_data=None, transform=None):
        """
        The constructor gets the images path and their respectively labels and meta-data (if applicable).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (and meta-data if applicable). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param labels (list) a list of labels for each image
        :param meta_data (list): a list of meta-data regarding each image. If None, there is no information.
        Defaul is None.
        :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
        """

        super().__init__()
        self.imgs_path = imgs_path
        self.labels = labels
        self.meta_data = meta_data

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
        # an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()


    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)


    def __getitem__(self, item):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(self.imgs_path[item]).convert("RGB")

        # Applying the transformations
        image = self.transform(image)

        img_id = self.imgs_path[item].split('/')[-1].split('.')[0]

        if self.meta_data is None:
            meta_data = []
        else:
            meta_data = self.meta_data[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, meta_data, img_id


def get_data_loader (imgs_path, labels, meta_data=None, transform=None, batch_size=30, shuf=True, num_workers=4,
                     pin_memory=True):
    """
    This function gets a list og images path, their labels and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param meta_data (list, optional): a list of meta-data regarding each image. If it's None, it means there's
    no meta-data. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    dt = MyDataset(imgs_path, labels, meta_data, transform)

    dl = data.DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory, drop_last=True)
    return dl

def get_loader():
    # Dataset variables
    _folder = 1
    _base_path = config.base_path
    _csv_path_train = os.path.join(_base_path, "/root/autodl-tmp/images/pad-ufes-20_parsed_folders.csv")# The file path of the training dataset metadata.
    _imgs_folder_train = os.path.join(_base_path, "/root/autodl-tmp/images")

    _use_meta_data = True
    _batch_size = config.batch_size

    meta_data_columns = ["smoke_False", "smoke_True", "drink_False", "drink_True", "background_father_POMERANIA",
                         "background_father_GERMANY", "background_father_BRAZIL", "background_father_NETHERLANDS",
                         "background_father_ITALY", "background_father_POLAND",	"background_father_UNK",
                         "background_father_PORTUGAL", "background_father_BRASIL", "background_father_CZECH",
                         "background_father_AUSTRIA", "background_father_SPAIN", "background_father_ISRAEL",
                         "background_mother_POMERANIA", "background_mother_ITALY", "background_mother_GERMANY",
                         "background_mother_BRAZIL", "background_mother_UNK", "background_mother_POLAND",
                         "background_mother_NORWAY", "background_mother_PORTUGAL", "background_mother_NETHERLANDS",
                         "background_mother_FRANCE", "background_mother_SPAIN", "age", "pesticide_False",
                         "pesticide_True", "gender_FEMALE", "gender_MALE", "skin_cancer_history_True",
                         "skin_cancer_history_False", "cancer_history_True", "cancer_history_False",
                         "has_piped_water_True", "has_piped_water_False", "has_sewage_system_True",
                         "has_sewage_system_False", "fitspatrick_3.0", "fitspatrick_1.0", "fitspatrick_2.0",
                         "fitspatrick_4.0", "fitspatrick_5.0", "fitspatrick_6.0", "region_ARM", "region_NECK",
                         "region_FACE", "region_HAND", "region_FOREARM", "region_CHEST", "region_NOSE", "region_THIGH",
                         "region_SCALP", "region_EAR", "region_BACK", "region_FOOT", "region_ABDOMEN", "region_LIP",
                         "diameter_1", "diameter_2", "itch_False", "itch_True", "itch_UNK", "grew_False", "grew_True",
                         "grew_UNK", "hurt_False", "hurt_True", "hurt_UNK", "changed_False", "changed_True",
                         "changed_UNK", "bleed_False", "bleed_True", "bleed_UNK", "elevation_False", "elevation_True",
                         "elevation_UNK"]


    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("-" * 50)
    print("- Loading  data...")
    _folder = 1
    test_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    # Loading validation data

    val_imgs_id = test_csv_folder['img_id'].values
    val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = test_csv_folder['diagnostic_number'].values

    if _use_meta_data:
        val_meta_data = test_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        val_meta_data = None
    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=0, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    print("- Loading training data...")
    train_imgs_id = train_csv_folder['img_id'].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['diagnostic_number'].values
    if _use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        train_meta_data = None
    train_data_loader = get_data_loader(train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=0, pin_memory=True)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)
    return train_data_loader, val_data_loader
