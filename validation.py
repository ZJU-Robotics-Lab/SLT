import torch
from sklearn.metrics import accuracy_score
from tools import wer
from utils.Recorder import Recorder
from utils.Averager import AverageMeter

def val_seq2seq(model, criterion, dataloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_wer = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc, avg_wer]
    names = ['val loss','val acc','val wer']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            # measure data loading time
            recoder.data_tok()
            # get the data and labels
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)

            # compute the loss
            loss = criterion(outputs, target)

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

            # compute wer
            # prediction: ((trg_len-1)*batch_size)
            # target: ((trg_len-1)*batch_size)
            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
            target = target.view(-1, batch_size).permute(1,0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))
            batch_wer = sum(wers)/len(wers)

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

            # update average value
            vals = [loss.item(),score,batch_wer]
            b = imgs.size(0)
            recoder.update(vals,count=b)

            # logging
            if batch_idx==0 or batch_idx % log_interval == log_interval-1 or batch_idx==len(dataloader)-1:
                recoder.log(epoch,batch_idx,len(dataloader),mode='Eval')

    return recoder.get_avg('val acc')

def test_seq2seq(model, criterion, dataloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_wer = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc, avg_wer]
    names = ['test loss','test acc','test wer']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            # measure data loading time
            recoder.data_tok()
            # get the data and labels
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)

            # compute the loss
            loss = criterion(outputs, target)

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

            # compute wer
            # prediction: ((trg_len-1)*batch_size)
            # target: ((trg_len-1)*batch_size)
            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
            target = target.view(-1, batch_size).permute(1,0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))
            batch_wer = sum(wers)/len(wers)

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

            # update average value
            vals = [loss.item(),score,batch_wer]
            b = imgs.size(0)
            recoder.update(vals,count=b)

            # logging
            if batch_idx==0 or batch_idx % log_interval == log_interval-1 or batch_idx==len(dataloader)-1:
                recoder.log(epoch,batch_idx,len(dataloader),mode='Test')

    return recoder.get_avg('val acc')

