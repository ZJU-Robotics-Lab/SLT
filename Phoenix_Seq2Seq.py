import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from datasets.Phoenix import Phoenix
from datasets.phoenixDataset import PhoenixDataset
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from train import train_seq2seq
from validation import val_seq2seq,test_seq2seq
from utils.ioUtils import *
from utils.textUtils import build_dictionary,reverse_phoenix_dictionary
from torchtext.data import Field
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# Path setting
train_video_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
dev_video_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev"
test_video_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/test"
train_anno_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
dev_anno_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
test_anno_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/test.corpus.csv"

model_path = "./checkpoint"
create_path(model_path)
create_path('log')
create_path('runs')
store_name = 'phoenix_seq2seq'
sum_path = "runs/phoenix_seq2seq/{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 100
batch_size = 1
learning_rate = 5e-5
weight_decay = 1e-5
sample_size = 224
sample_interval = 8
enc_hid_dim = 1000
emb_dim = 256
dec_hid_dim = 1000
dropout = 0.2
clip = 5
# Options
log_interval = 100
checkpoint = None

best_wer = 100.0
start_epoch = 0
if __name__ == '__main__':
    #-----------------------------load dataset--------------------------
    TRG = Field(sequential=True, use_vocab=True,
                init_token='<sos>', eos_token= '<eos>',
                lower=True, tokenize='spacy',
                tokenizer_language='de')


    root = '/mnt/data/public/datasets'
    csv_dir = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
    csv_dir = os.path.join(csv_dir, 'annotations/manual/train.corpus.csv')
    csv_file = pd.read_csv(csv_dir)
    tgt_sents = [csv_file.iloc[i, 0].lower().split('|')[3].split()
                for i in range(len(csv_file))]
    TRG.build_vocab(tgt_sents, min_freq=1)
    vocab_size = len(TRG.vocab)

    def collate_fn(batch):
        videos = [item['video'] for item in batch]
        video_lens = torch.tensor([len(v) for v in videos])
        videos = pad_sequence(videos, batch_first=True)
        videos = videos.permute(0,2,1,3,4).contiguous()
        annotations = [['<sos>']+item['annotation'].split()+['<eos>'] for item in batch]
        annotation_lens = torch.tensor([len(anno) for anno in annotations])
        annotations = TRG.process(annotations)
        return {'videos': videos,
                'annotations': annotations,
                'video_lens': video_lens,
                'annotation_lens': annotation_lens}

    transform = transforms.Compose([
        transforms.Resize(sample_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])

    train_loader = DataLoader(
        PhoenixDataset(root, mode='train', interval=sample_interval, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=16,
        collate_fn=collate_fn, pin_memory=True)

    val_loader = DataLoader(
        PhoenixDataset(root, mode='dev', interval=sample_interval, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=16,
        collate_fn=collate_fn, pin_memory=True)

    test_loader = DataLoader(
        PhoenixDataset(root, mode='test', interval=sample_interval, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=16,
        collate_fn=collate_fn, pin_memory=True)

    # Create Model
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(output_dim=vocab_size, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    # Resume model
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start training
    print("Training Started".center(60, '#'))
    wer = 100.0
    for epoch in range(start_epoch,start_epoch+epochs):
        # Train the model
        train_seq2seq(model, criterion, optimizer, clip, train_loader, device, epoch, log_interval, writer)
        # Validate the model
        val_seq2seq(model, criterion, val_loader, device, epoch, log_interval, writer)
        # Test the model
        wer = test_seq2seq(model, criterion, test_loader, device, epoch, log_interval, writer)
        # Save model
        # remember best wer and save checkpoint
        is_best = wer<best_wer
        best_wer = min(wer, best_wer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best': best_wer
        }, is_best, model_path, store_name)
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))