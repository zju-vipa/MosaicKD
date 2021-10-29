import torch
import torch.nn.functional as F
from PIL import Image

from os.path import join
import os

class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []
        train_dir = join(root, 'train')
        self.wnids = sorted(os.listdir(train_dir))
        self.wnid2index = {v: k for k, v in enumerate(self.wnids)}

        if self.split=='train':
            for wnid in self.wnids:
                train_index = self.wnid2index[wnid]
                subdir = join(self.root, 'train', wnid)
                for directory, _, names in os.walk(subdir):
                    for name in names:
                        filename = join(directory, name)
                        if filename.endswith('JPEG'):
                            self.data.append((filename, train_index))
        elif self.split=='val':
            val_dir = join(root, 'val')
            with open(join(val_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')
                infos = [info.strip().split('\t')[:2] for info in infos]

                self.data = [(join(val_dir, 'images', info[0]), self.wnid2index[info[1]]) for info in infos]

        print('Tiny ImageNet: split %s, size %d'%(self.split, len(self.data)))

    def __getitem__(self, index):
        image, target = self.data[index]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target
    
    def __len__(self):
        return len(self.data)