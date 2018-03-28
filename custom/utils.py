import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

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