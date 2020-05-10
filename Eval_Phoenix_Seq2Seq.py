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
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from train import train_seq2seq
from validation import val_seq2seq,test_seq2seq
from utils.ioUtils import *
from utils.textUtils import build_dictionary,reverse_phoenix_dictionary

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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 100
batch_size = 4
frames = 100
learning_rate = 1e-4
weight_decay = 1e-5
sample_size = 128
sample_duration = 48
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
clip = 1
# Options
log_interval = 100
checkpoint = 'None'

best_wer = 100.0
start_epoch = 0
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_dictionary(train_anno_file)
    reverse_dict = reverse_phoenix_dictionary(dictionary)
    vocab_size = len(reverse_dict)
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Phoenix(frames=frames,video_root=train_video_root,annotation_file=train_anno_file,
        dictionary=dictionary,transform=transform)
    val_set = Phoenix(frames=frames,video_root=dev_video_root,annotation_file=dev_anno_file,
        dictionary=dictionary,transform=transform)
    test_set = Phoenix(frames=frames,video_root=test_video_root,annotation_file=test_anno_file,
        dictionary=dictionary,transform=transform)
    print("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
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

    # Start evaluation
    print("Evaluation Started".center(60, '#'))
    for epoch in range(start_epoch,start_epoch+epochs):
        # Validate the model
        val_seq2seq(model, criterion, val_loader, device, epoch, log_interval, writer)
        # Test the model
        wer = test_seq2seq(model, criterion, test_loader, device, epoch, log_interval, writer)

    print("Evaluation Finished".center(60, '#'))