import random

import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam

from lib.utils.device import available_device
from lib.utils.path import data_path, model_path
from seq2seq.dataset import Eng2Kor, loader
from seq2seq.models import Decoder, Encoder


def main():
    device = available_device()

    dataset = Eng2Kor(data_path() / 'CH11.txt')

    encoder = Encoder(input_size=len(dataset.eng_bow), hidden_size=64).to(device)
    decoder = Decoder(64, len(dataset.kor_bow), dropout_p=0.1).to(device)

    encoder_optim = Adam(encoder.parameters(), lr=1e-4)
    decoder_optim = Adam(decoder.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    for ep in range(200):
        iterator = tqdm.tqdm(loader(dataset), total=len(dataset))
        total_loss = 0

        for data, label in iterator:
            data = torch.tensor(data, dtype=torch.long).to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)

            encoder_hidden = torch.zeros(1, 1, 64).to(device)
            encoder_outputs = torch.zeros(11, 64).to(device)

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            loss = 0

            for ei in range(len(data)):
                encoder_output, encoder_hidden = encoder(data[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[0]]).to(device)

            decoder_hidden = encoder_hidden
            use_teacher_forcing = random.random() < 0.5

            if use_teacher_forcing:
                for di in range(len(label)):
                    decoder_output = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )

                    target = torch.tensor(label[di], dtype=torch.long).to(device)
                    target = torch.unsqueeze(target, dim=0).to(device)
                    loss += criterion(decoder_output, target)
                    decoder_input = target
            else:
                for di in range(len(label)):
                    decoder_output = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )

                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()

                    target = torch.tensor(label[di], dtype=torch.long).to(device)
                    target = torch.unsqueeze(target, dim=0).to(device)
                    loss += criterion(decoder_output, target)

                    if decoder_input.item() == 1:
                        break

            total_loss += loss.item() / len(dataset)
            iterator.set_description(f'epoch{ep + 1} loss:{total_loss}')
            loss.backward()

            encoder_optim.step()
            decoder_optim.step()

    torch.save(encoder.state_dict(), model_path('attn_enc.pth'))
    torch.save(decoder.state_dict(), model_path('attn_dec.pth'))


if __name__ == '__main__':
    main()
