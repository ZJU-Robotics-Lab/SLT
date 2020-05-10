import torch
from torch.utils.data import Dataset

import os
import pandas as pd
from PIL import Image
import threading
import numpy as np

    
class PhoenixDataset(Dataset):
    def __init__(self, root, mode, interval, transform=None):
        '''
        Args:
            root: the original root where the downloaded dataset are placed
            mode: 'train', 'dev', 'test'
            interval: read a frame at intervals
        '''
        root = os.path.join(
            root, 'phoenix2014-release/phoenix-2014-multisigner')
        csv_path = os.path.join(
            root, 'annotations/manual/' + mode + '.corpus.csv')
        self.csv_file = pd.read_csv(csv_path)
        self.video_root = os.path.join(
            root, 'features/fullFrame-210x260px/' + mode)
        self.interval = interval
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    # read frames through multi-threading
    def read_frames(self, frame_paths, iThread, frames, lock):
        frames_tmp = []
        for p in frame_paths:
            frame = Image.open(p)
            if self.transform:
                frame = self.transform(frame)
            frames_tmp.append(frame)
        lock.acquire()
        frames.append((iThread, frames_tmp))
        lock.release()
        
    def __getitem__(self, idx):
        video_path = os.path.join(
            self.video_root, self.csv_file.iloc[idx, 0].split('|')[0])

        video_path = os.path.join(video_path, '1')
        paths = sorted(os.listdir(video_path))
        paths = [os.path.join(video_path, p) for i, p in enumerate(paths) if i%self.interval==0]
        
        frames = []
        lock=threading.Lock()
        nFrames_per_thread = 10
        paths = [paths[i:i+nFrames_per_thread] 
                 for i in range(0,len(paths), nFrames_per_thread)]
        threads = [threading.Thread(target=self.read_frames,args=([paths[i], i, frames, lock]))
                      for i in range(len(paths))]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        frames = torch.stack([p for t in sorted(frames) for p in t[1]],dim=0)
        
        annotation = self.csv_file.iloc[idx,0].split('|')[3].lower()
        return {'video': frames, 'annotation': annotation}



        
        
        
        
        
        
        
        