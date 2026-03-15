import torch

from lib.utils.device import available_device
from lib.utils.path import model_path
from lstm.dataset import TextGeneration
from lstm.models import LSTM


def generate(model, BOW, string='finding an', strlen=10):
    device = available_device()

    print(f'input word: {string}')

    with torch.no_grad():
        for _ in range(strlen):
            print(f'DEBUG: split words -> {string.split()}')
            words = torch.tensor([BOW[w] for w in string.split()], dtype=torch.long).to(
                device
            )

            input_tensor = torch.unsqueeze(words[-2:], dim=0)
            output = model(input_tensor)
            output_word = torch.argmax(output).cpu().numpy()
            string += ' ' + list(BOW.keys())[output_word]
            # string += list(BOW.keys())[output_word]
            # string += ' '

    print(f'predicated sentence: {string}')


def run():
    device = available_device()
    dataset = TextGeneration()

    model = LSTM(num_embeddings=len(dataset.BOW)).to(device)

    model.load_state_dict(torch.load(model_path('lstm.pth'), map_location=device))
    pred = generate(model, dataset.BOW)

    print(f'pred: {pred}')
