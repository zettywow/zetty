from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import torch, os
import random
from tqdm import tqdm

"""
def get_train_dataset(imgs_folder,load_function):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(imgs_folder,train_transform,loader=load_function)
    class_num = dataset[-1][1] + 1
    return dataset, class_num


def get_train_loader(path,batch_size,load_function):
    dataset,classnum = get_train_dataset(path,load_function)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle = True,num_workers=3)
    return loader,classnum
"""

def get_dfw_loader(dfw_path,data_list_file,batch_size,collate_fn):
    dataset = dfw_dataset(dfw_path,data_list_file)
    #loader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,num_workers=3)
    loader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 3,collate_fn=collate_fn)
    return loader


def collate_fn(batch):
    anchor = torch.stack([triplet[0] for triplet in batch],dim = 0)
    positive = torch.stack([triplet[1] for triplet in batch],dim = 0)
    negative = torch.stack([triplet[2] for triplet in batch],dim = 0)
    out = torch.cat([anchor,positive,negative])
    assert out.shape[1:] == (3,112,112)
    return out

class dfw_dataset(Dataset):

    def __init__(self, root, data_list_file):
        #self.input_shape = input_shape
        self.root = root

        with open(data_list_file, 'r') as fd:
            lines = fd.readlines()
        
        lines = [line[:-1].split('\t') for line in lines]
        self.lines = np.random.permutation(lines)

        normalize = trans.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        #normalize = trans.Normalize(mean=[0.5], std=[0.5])

        
        self.transforms = trans.Compose([
                trans.RandomCrop(112),
                trans.ColorJitter(contrast=0.2,hue=0.2),
                trans.RandomGrayscale(p=0.1),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                normalize
                ])
        

    def __getitem__(self, index):
        triplet = self.lines[index]
        img_trip = []
        for single in triplet:
            img_path = os.path.join(self.root,str(single))
            data = Image.open(img_path)
            if data.mode != 'RGB':
                data = data.convert('RGB')

            i = random.randint(0,8)
            j = random.randint(0,8)
            data = data.resize((112+i,112+j))
            data = self.transforms(data)
            assert data.shape == (3,112,112)
            img_trip.append(data)
        #out = torch.stack(img_trip,dim = 0)
        #assert out.shape == (3,3,112,112)
        return img_trip

    def __len__(self):
        return len(self.lines)


