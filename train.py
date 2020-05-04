import torch
from sklearn.metrics import accuracy_score
from tools import wer
from utils.Recorder import Recorder
from utils.Averager import AverageMeter

def train_seq2seq(model, criterion, optimizer, clip, dataloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_wer = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc, avg_wer]
    names = ['train loss','train acc','train wer']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for batch_idx, (imgs, target) in enumerate(dataloader):
        # measure data loading time
        recoder.data_tok()
        # get the data and labels
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(imgs, target)

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
            # add mask(remove padding, sos, eos)
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))
        batch_wer = sum(wers)/len(wers)

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),score,batch_wer]
        b = imgs.size(0)
        recoder.update(vals,count=b)

        if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
            recoder.log(epoch,batch_idx,len(dataloader))
            # Reset average meters 
            recoder.reset() 
