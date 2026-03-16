import random

import torch

from lib.utils.device import available_device
from lib.utils.path import data_path, model_path
from seq2seq.dataset import Eng2Kor
from seq2seq.models import Decoder, Encoder


def main():
    device = available_device()

    dataset = Eng2Kor(data_path() / 'CH11.txt')

    checkpoint = torch.load(model_path('attn_enc.pth'), map_location=device)
    print(checkpoint.keys())

    encoder = Encoder(input_size=len(dataset.eng_bow), hidden_size=64).to(device)
    decoder = Decoder(64, len(dataset.kor_bow), dropout_p=0.1).to(device)

    encoder.load_state_dict(torch.load(model_path('attn_enc.pth'), map_location=device))
    decoder.load_state_dict(torch.load(model_path('attn_dec.pth'), map_location=device))

    idx = random.randint(0, len(dataset))
    input_sentence = dataset.eng_corpus[idx]
    pred_sentence = ''

    data, label = dataset[idx]
    data = torch.tensor(data, dtype=torch.long).to(device)
    label = torch.tensor(label, dtype=torch.long).to(device)

    encoder_hidden = torch.zeros(1, 1, 4).to(device)
    encoder_outputs = torch.zeros(11, 64).to(device)

    for ei in range(len(data)):
        encoder_output, encoder_hidden = encoder(data[ei], encoder_hidden)
        encoder_output[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[0]]).to(device)
    decoder_hidden = encoder_hidden

    for _ in range(11):
        decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)
        _, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        if decoder_input.item() == 1:
            break

        pred_sentence += list(dataset.kor_bow.keys())[decoder_input] + ' '

    print(input_sentence)
    print(pred_sentence)


if __name__ == '__main__':
    main()
