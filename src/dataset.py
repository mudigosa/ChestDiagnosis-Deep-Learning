import torch
import numpy as np
from PIL import Image

# Build the dataset
class ChestXrayDataSet(torch.utils.data.Dataset):
  def __init__(self,data,path,transform=None):
    self.data = data
    self.path = path
    self.len = data.shape[0]
    self.transform = transform
  
  def __len__(self):
    return self.len

  def __getitem__(self,idx):
    image_path = self.path + self.data.iloc[idx,0]
    image = Image.open(image_path).convert('RGB')

    if self.transform:
      image = self.transform(image)

    labels = list(self.data.iloc[idx,1:])
    labels = torch.FloatTensor(labels)

    return image, labels