import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import torchvision.transforms as T
class Clevrer(Dataset):
    def __init__(self, npy_dir,seq_len=32, downsample_ratio=4,downsample_mode='Notuniform'):
        self.npy_files = sorted(glob.glob(os.path.join(npy_dir, '*','*.npy')))
        self.dsr = downsample_ratio
        self.down_mode = downsample_mode
        self.seq_len = seq_len
        self.clip_num = 128 // self.dsr // self.seq_len

    def __len__(self):
        return len(self.npy_files) * self.clip_num

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx // self.clip_num]
        video_data = np.load(npy_file)
        video_data = np.transpose(video_data, (0, 3, 1, 2)) # move channel axis to the first dimension
        if self.down_mode == 'uniform':
            video_data = video_data[::self.dsr]
            video_data = video_data[
                         idx % self.clip_num * self.seq_len:
                         idx % self.clip_num * self.seq_len + min(self.seq_len,video_data.shape[0])]
            video_data = torch.from_numpy(video_data).float()
            return video_data
        else:
            frame_num = np.asarray(range(0,128//self.dsr))
            frame_num_uneven = [min(x*self.dsr + np.random.randint(-self.dsr//2,self.dsr//2+1),127) for x in frame_num]
            frame_num_uneven[0] = 0
            uneven_video_data = video_data[frame_num_uneven]
            uneven_video_data = uneven_video_data[
                         idx % self.clip_num * self.seq_len:
                         idx % self.clip_num * self.seq_len + min(self.seq_len, video_data.shape[0])]
            video_data = video_data[::self.dsr]
            video_data = video_data[
                         idx % self.clip_num * self.seq_len:
                         idx % self.clip_num * self.seq_len + min(self.seq_len, video_data.shape[0])]
            uneven_video_data = torch.from_numpy(uneven_video_data).float()
            video_data = torch.from_numpy(video_data).float()
            return video_data,uneven_video_data,idx // self.clip_num

