import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import kornia
import PIL
import torch

def im_to_float(im):
    return im.float().div( 255 )

class TensorDatasetWithTransform(TensorDataset):
    def __init__(self, data, labels, transforms):
        super().__init__(data, labels)
        self.transforms = transforms

    def __getitem__(self, item):
        example, label = super().__getitem__(item)
        return self.transforms(example), label
def get_cifar_train_transforms():
    transform = transforms.Compose(
                [transforms.RandomResizedCrop( (32, 32), scale=(0.8, 0.8) ),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return transform

def get_test_transform():
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return test_transform

def get_test_torch_transform():
    test_torch_transform = transforms.Compose(
        [im_to_float,
         transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return test_torch_transform

def get_img_and_mask_transforms(desired_size):
    img_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(desired_size),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    tgt_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(desired_size, interpolation=PIL.Image.NEAREST),
                 lambda x: x.long()])
    return img_transform, tgt_transform
