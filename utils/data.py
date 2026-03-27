import numpy as np
import torchvision
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, read_images_labels, read_images_labels_imageneta, read_images_labels_vfn


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None




def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t





class Food101_lt(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(101).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"

        train_dset = open('./data_txt/food101/train_lt.txt','r').readlines()
        test_dset = open('./data_txt/food101/test.txt','r').readlines()

        self.train_data, self.train_targets = read_images_labels(train_dset)
        self.test_data, self.test_targets = read_images_labels(test_dset)




class VFN(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(187).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"

        train_dset = open('./data_txt/VFN186/Healthy_train/train_lt.txt','r').readlines()
        test_dset = open('./data_txt/VFN186/balanced_test.txt','r').readlines()

        self.train_data, self.train_targets = read_images_labels_vfn(train_dset)
        self.test_data, self.test_targets = read_images_labels_vfn(test_dset)


class VFN_insulin(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(187).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"

        train_dset = open('./data_txt/VFN186/Type1_train/train_insulin.txt','r').readlines()
        test_dset = open('./data_txt/VFN186/balanced_test.txt','r').readlines()

        self.train_data, self.train_targets = read_images_labels_vfn(train_dset)
        self.test_data, self.test_targets = read_images_labels_vfn(test_dset)


class VFN_t2d(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(187).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"

        train_dset = open('./data_txt/VFN186/Type2_train/train_t2d.txt','r').readlines()
        test_dset = open('./data_txt/VFN186/balanced_test.txt','r').readlines()

        self.train_data, self.train_targets = read_images_labels_vfn(train_dset)
        self.test_data, self.test_targets = read_images_labels_vfn(test_dset)
