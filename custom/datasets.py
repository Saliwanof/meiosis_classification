from PIL import Image
import skimage.io as skio
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from .utils import main_part_extract, padding_to_224

def make_dataset(path, label={'leptotene', 'zygotene', 'pachytene_ab', 'pachytene_n'}):
    im_paths = []
    if path is None:
        return im_paths
    path = Path(path)
    im_leptotene = list(map(str, list(path.glob('**/leptotene/*S.tif'))))
    im_zygotene = list(map(str, list(path.glob('**/zygotene/*S.tif'))))
    im_pachytene_ab = list(map(str, list(path.glob('**/Abnormal pachytene/*S.tif'))))
    im_pachytene_n = list(map(str, list(path.glob('**/Normal pachytene/*S.tif'))))
    
    im_paths = {'leptotene':im_leptotene, 'zygotene':im_zygotene, 'pachytene_ab':im_pachytene_ab, 'pachytene_n':im_pachytene_n}

    return im_paths[label]


class Meiosis_Dataset(data.Dataset):
    def __init__(self, path=None, label=None, transform=None, target=None):
        im_paths = make_dataset(path, label=label)

        self.im_paths = im_paths
        self.transform = transform
        self.target = target

    def __getitem__(self, index):
        path = self.im_paths[index]
        img = skio.imread(path)
        img = np.floor(img / 16).astype('uint8')
        img = main_part_extract(img)
        if len(img.shape)==2:
            np.expand_dims(img, axis=0)
        img = Image.fromarray(img)
        
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224, (.9, 1.), (1., 1.)),
            # transforms.RandomCrop(224),
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x[0] if len(x.shape)==3 else x)
        ])
        
        img = preprocess(img)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.LongTensor([self.target])
        
        return img, target

    def __len__(self):
        return len(self.im_paths)
    
#
