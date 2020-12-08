import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg



class S3DIS_cls(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        print('loading', partition, 'data')
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]

        # Shuffle the training data
        # But it seems like it doesn't work
        if self.partition == 'train':
            indices = list(range(self.data[item].shape[0]))
            np.random.shuffle(indices)
            self.data[item] = self.data[item][indices]
            self.seg[item] = self.seg[item][indices]

        labels = torch.zeros(13)
        for idx in set(seg):
            labels[idx] = 1.
        return pointcloud, labels

    def __len__(self):
        return self.data.shape[0]
