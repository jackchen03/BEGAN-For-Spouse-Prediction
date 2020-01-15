import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from os import listdir
from os.path import join
import random
import math
import pandas as pd



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

def kinship_loader(dir_name):
    if(os.listdir(dir_name) == []):
      print(dir_name + ' is an empty dir.')
    else:
      img_name = random.choice(os.listdir(dir_name))
      img = Image.open(dir_name + img_name).convert('RGB')
      return img

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


# You should build custom dataset as below.
class CelebA(data.Dataset):
    def __init__(self,dataPath='../dataset/img_align_celeba_64_crop/',loadSize=64,fineSize=64,flip=1):
        super(CelebA, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = os.path.join(self.dataPath,self.image_list[index])
        img = default_loader(path) 
        w,h = img.size

        if(h != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            #x1 = random.randint(0, self.loadSize - self.fineSize)
            #y1 = random.randint(0, self.loadSize - self.fineSize)
             
            x1 = math.floor((self.loadSize - self.fineSize)/2)
            y1 = math.floor((self.loadSize - self.fineSize)/2)
            img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

        if(self.flip == 1):
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = ToTensor(img) # 3 x 256 x 256

        img = img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

# You should build custom dataset as below.
class Kinship(data.Dataset):
    def __init__(self,dataPath='../dataset/FID_align/',loadSize=64,fineSize=64,flip=1):
        super(Kinship, self).__init__()
        # list all images into a list
        self.dataroot = dataPath
        self.mom_son_dir = []
        for i in range(1, 1001):
          if(i == 1000): number = str(i)
          elif (i > 99): number = str(0) + str(i)
          elif (i > 9): number = str(0) + str(0) + str(i)
          else: number = str(0) + str(0) + str(0) + str(i)
          df = pd.read_csv(self.dataroot + 'F' + number + '/' + 'mid.csv', sep=',')
          for j in range(1, len(df['Gender']) ):
            if(df['Gender'][j-1] == 'Female'):
              for k in range(len(df[str(j)])):
                if(df[str(j)][k] == 1 and df['Gender'][k] == 'Male'):
                  mom_dir = self.dataroot + 'F' + number + '/' + 'MID' + str(j) + '/'
                  son_dir = self.dataroot + 'F' + number + '/' + 'MID' + str(k+1) + '/'
                  if(os.listdir(mom_dir) == [] or os.listdir(son_dir) == []):
                    print('Empty dir detected.')
                  else: self.mom_son_dir.append([mom_dir, son_dir])

        self.dataset_size = len(self.mom_son_dir)
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        mom_path = self.mom_son_dir[index][0]
        son_path = self.mom_son_dir[index][1]
        mom_img = kinship_loader(mom_path) 
        son_img = kinship_loader(son_path) 
        w,h = mom_img.size

        if(h != self.loadSize):
            mom_img = mom_img.resize((self.loadSize, self.loadSize), Image.BILINEAR)
            son_img = son_img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            #x1 = random.randint(0, self.loadSize - self.fineSize)
            #y1 = random.randint(0, self.loadSize - self.fineSize)
             
            x1 = math.floor((self.loadSize - self.fineSize)/2)
            y1 = math.floor((self.loadSize - self.fineSize)/2)
            mom_img = mom_img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            son_img = son_img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

        if(self.flip == 1):
            if random.random() < 0.5:
                mom_img = mom_img.transpose(Image.FLIP_LEFT_RIGHT)
                son_img = son_img.transpose(Image.FLIP_LEFT_RIGHT)

        mom_img = ToTensor(mom_img) # 3 x 256 x 256
        son_img = ToTensor(son_img) # 3 x 256 x 256

        # # transform to grey scale
        # mom_img = mom_img[0, ...] * 0.299 + mom_img[1, ...] * 0.587 + mom_img[2, ...] * 0.114
        # mom_img = mom_img.unsqueeze(0)
        # son_img = son_img[0, ...] * 0.299 + son_img[1, ...] * 0.587 + son_img[2, ...] * 0.114
        # son_img = son_img.unsqueeze(0)

        mom_img = mom_img.mul_(2).add_(-1)
        son_img = son_img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return mom_img, son_img

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.dataset_size

    def length(self):
        return self.dataset_size

# # You should build custom dataset as below.
# class AFAD(data.Dataset):
#     def __init__(self,dataPath='../dataset/tarball-lite-master/AFAD-Lite_aligned/',loadSize=64,fineSize=64,flip=1):
#         super(AFAD, self).__init__()
#         # list all images into a list
#         self.image_list = [(str(i) + '/' + str(j) + '/' + x) for i in range(18,40) for j in range(111,113) for x in listdir(dataPath + str(i) + '/' + str(j) + '/') ]
#         print(len(self.image_list))
#         self.dataPath = dataPath
#         self.loadSize = loadSize
#         self.fineSize = fineSize
#         self.flip = flip

#     def __getitem__(self, index):
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         path = os.path.join(self.dataPath,self.image_list[index])
#         img = default_loader(path) 
#         w,h = img.size

#         if(h != self.loadSize):
#             img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

#         if(self.loadSize != self.fineSize):
#             #x1 = random.randint(0, self.loadSize - self.fineSize)
#             #y1 = random.randint(0, self.loadSize - self.fineSize)
             
#             x1 = math.floor((self.loadSize - self.fineSize)/2)
#             y1 = math.floor((self.loadSize - self.fineSize)/2)
#             img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

#         if(self.flip == 1):
#             if random.random() < 0.5:
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)

#         img = ToTensor(img) # 3 x 256 x 256

#         img = img.mul_(2).add_(-1)
#         # 3. Return a data pair (e.g. image and label).
#         return img

#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return len(self.image_list)



def get_loader(root, batch_size, scale_size, kinship=True, num_workers=12, shuffle=True):
    if kinship == True:
        dataset = Kinship(root,scale_size,scale_size,1)
        print(dataset.length())
    else:
        dataset = CelebA(root,scale_size,scale_size,1)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)
    return data_loader


