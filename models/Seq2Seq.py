import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

"""
Implementation of Sequence to Sequence Model
Encoder: encode video spatial and temporal dynamics e.g. CNN+RNN
Decoder: decode the compressed info from encoder
"""
class Encoder(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, arch="resnet18"):
        super(Encoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.rnn = nn.GRU(
            input_size=resnet.fc.in_features,
            hidden_size=self.enc_hid_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # RNN
        # output: (batch, seq_len, num_directions * enc_hid_dim)
        # h_n: (batch, num_layers * num_directions, enc_hid_dim)
        output, h_n = self.rnn(cnn_embed_seq, None)
        # (batch, dec_hid_dim)
        h_n = torch.tanh(self.fc(torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)))

        return output, h_n


# class EncoderPlus(nn.Module):
#     def __init__(self, lstm_hidden_size=512, clip_len=5, arch="resnet18"):
#         super(EncoderPlus, self).__init__()
#         self.lstm_hidden_size = lstm_hidden_size
#         self.clip_len = clip_len

#         # frame encoder
#         if arch == "resnet18":
#             resnet = models.resnet18(pretrained=True)
#         elif arch == "resnet34":
#             resnet = models.resnet34(pretrained=True)
#         elif arch == "resnet50":
#             resnet = models.resnet50(pretrained=True)
#         elif arch == "resnet101":
#             resnet = models.resnet101(pretrained=True)
#         elif arch == "resnet152":
#             resnet = models.resnet152(pretrained=True)
#         # delete the last fc layer
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.lstm = nn.LSTM(
#             input_size=resnet.fc.in_features,
#             hidden_size=self.lstm_hidden_size,
#             batch_first=True,
#         )
#         # clip encoder
#         resnet3d = models.video.r3d_18(pretrained=True, progress=True)
#         modules3d = list(resnet3d.children())[:-1]
#         self.resnet3d = nn.Sequential(*modules3d)
#         self.lstm3d = nn.LSTM(
#             input_size=resnet3d.fc.in_features,
#             hidden_size=self.lstm_hidden_size,
#             batch_first=True,
#         )

#     def forward(self, x):
#         # frame level
#         cnn_embed_seq = []
#         # x: (batch_size, channel, t, h, w)
#         for t in range(x.size(2)):
#             out = self.resnet(x[:, :, t, :, :])
#             out = out.view(out.size(0), -1)
#             cnn_embed_seq.append(out)
#         cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
#         # batch first
#         cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)
#         # LSTM
#         # use faster code paths
#         self.lstm.flatten_parameters()
#         frame_out, (frame_h_n, frame_c_n) = self.lstm(cnn_embed_seq, None)

#         # clip level
#         cnn3d_embed_seq = []
#         for t in range(x.size(2)-self.clip_len+1):
#             out = self.resnet3d(x[:, :, t:t+self.clip_len, :, :])
#             # print(out.shape)
#             out = out.view(out.size(0), -1)
#             cnn3d_embed_seq.append(out)
#         cnn3d_embed_seq = torch.stack(cnn3d_embed_seq, dim=0)
#         # batch first
#         cnn3d_embed_seq = cnn3d_embed_seq.transpose_(0, 1)
#         # LSTM
#         # use faster code paths
#         self.lstm3d.flatten_parameters()
#         clip_out, (clip_h_n, clip_c_n) = self.lstm3d(cnn3d_embed_seq, None)

#         # num_layers * num_directions = 1
#         frame_h_n = frame_h_n.squeeze(0)
#         frame_c_n = frame_c_n.squeeze(0)
#         clip_h_n = clip_h_n.squeeze(0)
#         clip_c_n = clip_c_n.squeeze(0)
#         return frame_out, (frame_h_n, frame_c_n), clip_out, (clip_h_n, clip_c_n)


class Attn(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(Attn, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, num_directions * hidden_size)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # calculate energy
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))
        # (batch, attn_dim, seq_len)
        energy = energy.permute(0, 2, 1)

        # (batch, 1, attn_dim)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # calculate attention(batch, seq_len)
        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim+enc_hid_dim*2, dec_hid_dim)
        self.fc = nn.Linear(self.attention.attn_in+emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, num_directions * hidden_size)
        # (batch, seq_len)
        a = self.attention(decoder_hidden, encoder_outputs)

        # (batch, 1, seq_len)
        a = a.unsqueeze(1)

        # (batch, 1, num_directions * hidden_size)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        # (1, batch, num_directions * hidden_size)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, encoder_outputs):
        # input: (batch)
        # decoder_hidden: (batch, dec_hid_dim)
        # encoder_outputs: (batch, seq_len, num_directions * hidden_size)
        # (1, batch_size)
        input = input.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        embedded = self.dropout(self.embedding(input))

        # (1, batch, num_directions * hidden_size)
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)

        # rnn_input(1, batch_size, emb_dim+num_directions * hidden_size): concat embedded and context 
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        # output(seq_len, batch, num_directions * hidden_size)
        # decoder_hidden(num_layers * num_directions, batch, hidden_size)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        # (batch, emb_dim)
        embedded = embedded.squeeze(0)
        # (batch, num_directions * hidden_size)
        output = output.squeeze(0)
        # (batch, num_directions * hidden_size)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # (batch, hidden_size)
        decoder_hidden = decoder_hidden.squeeze(0)

        # prediction
        prediction = self.fc(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return prediction, decoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, trg len)
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, hidden = self.encoder(imgs)

        # first input to the decoder is the <sos> tokens
        input = target[:,0]

        for t in range(1, trg_len):
            # decode
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:,t] if teacher_force else top1

        return outputs


# Test
if __name__ == '__main__':
    # test encoder
    encoder = Encoder(enc_hid_dim=512, dec_hid_dim=512)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # print(encoder(imgs))

    # test encoderPlus
    # encoderPlus = EncoderPlus(lstm_hidden_size=512)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # print(encoderPlus(imgs))

    # test decoder
    attn = Attn(enc_hid_dim=512, dec_hid_dim=512, attn_dim=64)
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5, attention=attn)
    # input = torch.LongTensor(16).random_(0, 500)
    # hidden = torch.randn(16, 512)
    # cell = torch.randn(16, 512)
    # context = torch.randn(16, 512)
    # print(decoder(input, hidden, cell, context))

    # test seq2seq
    device = torch.device("cpu")
    # seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print(seq2seq(imgs, target).argmax(dim=2).permute(1,0)) # batch first
