import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
import json
import numpy
import sys
from utils.textUtils import *
import torch.nn.functional as F
from PIL import Image

class Record:
    def __init__(self,path,sentence):
        self.image_path = path
        self.sentence = sentence

"""
Implementation of CSL Phoenix Dataset
"""
class Phoenix(Dataset):
    def __init__(self,video_root='',annotation_file='',
                dictionary=None,
                frames=100,
                stride=16,
                upsample_rate=1,
                transform=None):
        super(Phoenix,self).__init__()
        self.video_root = video_root
        self.annotation_file = annotation_file
        self.frames = frames
        self.stride = stride
        self.dictionary = dictionary
        self.stride = stride
        self.upsample_rate = upsample_rate
        self.transform = transform
        self.prepare()
        self.get_data_list()

    def prepare(self):
        df = pd.read_csv(self.annotation_file,sep='|')
        lang_model = spacy.load('de')
        punctuation = ['_','NULL','ON','OFF','EMOTION','LEFTHAND','IX','PU']

        self.punctuation = punctuation
        self.lang_model = lang_model
        self.df = df

    def process_sentence(self,sentence):
        sentence = [tok.text for tok in self.lang_model.tokenizer(sentence) 
            if not tok.text in self.punctuation]
        sentence = ['<bos>'] + sentence + ['<eos>']
        indices = []
        for word in sentence:
            if word in self.dictionary.keys():
                indices.append(self.dictionary[word])
            else:
                # the index of <unk> is 3
                indices.append(3)
        return indices
    
    def get_data_list(self):
        self.data_list = []
        for i in range(len(self.df)):
            row = self.df.loc[i]
            skeleton_path = row['id']
            sentence = row['annotation']
            sentence = self.process_sentence(sentence)
            record = Record(skeleton_path,sentence)
            self.data_list.append(record)

    def read_image(self,filename):
        image =  Image.open(filename).convert('RGB')
        return image


    def read_images(self, frame_path):
        frame_path = os.path.join(self.video_root,frame_path,'1')
        image_list = os.listdir(frame_path)
        image_list.sort()
        # if len(image_list) < self.frames:
        #     print("Too few images(%d) in your data folder: %s"%(len(image_list),str(frame_path)))
        images = []
        start = 0
        step = max(int(len(os.listdir(frame_path))/self.frames),1)
        for i in range(self.frames):
            index = min(start+i*step,len(image_list)-1)
            image = self.read_image(os.path.join(frame_path, image_list[index]))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __getitem__(self, idx):
        record = self.data_list[idx]
        image_path = record.image_path
        tokens = record.sentence
        N = len(tokens)
        images = self.read_images(image_path)
        tokens = torch.LongTensor(tokens)

        return images, tokens
        
    def __len__(self):
        return len(self.data_list)

# Test
if __name__ == '__main__':
    # Path settings
    train_video_root = "/mnt/data/haodong/openpose_output/train"
    train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"
    dev_video_root = "/mnt/data/haodong/openpose_output/dev"
    dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"
    # Build dictionary
    dictionary = build_dictionary([train_annotation_file,dev_annotation_file])
    dataset = Phoenix(image_root=train_video_root,annotation_file=train_annotation_file,dictionary=dictionary)
    print(dataset[3000]['input'].size())


