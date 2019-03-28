import random
import os
import numpy as np
import cv2
import torch
import torch.utils.data as dataf
from torch.utils.data import DataLoader


def load_data(data_path, batch_size):
    all_path = []
    real_label = 1
    signal = os.listdir(data_path)
    for fsingal in signal:
        filepath = data_path + fsingal
        all_path.append(filepath)

    random.shuffle(all_path)
    count = len(all_path)
    data_x = np.empty((count,1,28,28),dtype='float32')
    data_y = []


    i = 0

    for item in all_path:
        img = cv2.imread(item,0)
        img = cv2.resize(img,(28,28))
        arr = np.asarray(img,dtype='float32')
        data_x[i,:,:,:] = arr
        i += 1
        data_y.append(real_label)

    data_x = data_x/255.0
    data_y = np.asarray(data_y)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    train_data = dataf.TensorDataset(data_x,data_y)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

    return train_loader
