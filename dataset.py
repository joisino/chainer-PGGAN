import numpy as np
import chainer
import os
from PIL import Image

class YuiDataset(chainer.dataset.DatasetMixin):
    def __init__(self, directory, depth):
        self.directory = directory
        self.files = os.listdir(directory)
        self.depth = depth
        
    def __len__(self):
        return len(self.files)

    def random_box(self, size, d):
        l = np.random.randint(0, d+1)
        t = np.random.randint(0, d+1)
        r = size[0] - (d-l)
        b = size[1] - (d-t)
        return (l, t, r, b) 

    def get_example(self, i):
        img = Image.open(os.path.join(self.directory, self.files[i]))
        img = img.crop(self.random_box(img.size, 16))

        size = 2**(2+self.depth)
        img = img.resize((size, size))
        
        img = np.array(img, dtype=np.float32) / 256
        if len(img.shape) == 2:
            img = np.broadcast_to(img, (3, img.shape[0], img.shape[1]))
        else:
            img = np.transpose(img, (2, 0, 1))
        
        return img
