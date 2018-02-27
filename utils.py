from PIL import Image
import numpy as np

import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

import os
import os.path

from torchvision.datasets.folder import default_loader, is_image_file
import torchvision.transforms as transforms
import torch.utils.data as data

def make_dataset(dir, merged=True, min_height=224, min_width=224):
    images = []
    if dir is None:
        return images
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if not merged and '-s.tif' not in fname:
                    continue
                path = os.path.join(root, fname)
                '''
                with Image.open(path) as im:
                    width, height = im.size
                if width>=min_width and height >=min_height:
                    images.append(path)
                '''
                if '.tif' in fname:
                    images.append(path)

    return images


def main_part_extract(im):
    height, width = im.shape
    thresh = threshold_otsu(im)
    bw = im > thresh
    cl = remove_small_objects(bw)
    x, y = np.nonzero(cl)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xmin, xmax = padding_to_224(xmin, xmax, height)
    ymin, ymax = padding_to_224(ymin, ymax, width)
    return im[xmin:xmax+1, ymin:ymax+1]


def padding_to_224(xmin, xmax, height):
    if xmax - xmin < 223:
        deltah = np.ceil((224 - (xmax - xmin + 1)) / 2.)
        if xmin <= deltah:
            xmin = 0
            xmax = 223
        elif height - xmax <= deltah:
            xmax = height
            xmin = height - 223
        else:
            xmax = xmax + deltah
            xmin = xmin - deltah
    return xmin, xmax


class Meiosis_Dataset(data.Dataset):
    def __init__(self, root_merged=None, root_unmerged=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs_merged = make_dataset(root_merged)
        imgs_unmerged = make_dataset(root_unmerged, merged=False)
        imgs = imgs_merged + imgs_unmerged

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        # img = self.loader(path)
        img = skio.imread(path)
        img = np.floor(img / 16).astype('uint8')
        img = main_part_extract(img)
        if len(img.shape)==2:
            np.expand_dims(img, axis=0)
        img = Image.fromarray(img)
        
        preprocess = transforms.Compose([
            # transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x[0] if len(x.shape)==3 else x)
        ])
        
        img = preprocess(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img

    def __len__(self):
        return len(self.imgs)
    
#
