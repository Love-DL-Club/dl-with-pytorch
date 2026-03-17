import matplotlib.pyplot as plt
import torch

from crnn.dataset import Captcha
from crnn.models import CRNN
from lib.utils.device import available_device
from lib.utils.path import data_path, model_path


def main():
    device = available_device()

    ch12_path = str(data_path() / 'CH12')
    dataset = Captcha(path=ch12_path)
    testset = Captcha(path=ch12_path, train=False)

    model = CRNN(output_size=len(dataset.BOW)).to(device)
    model.load_state_dict(torch.load(model_path('CRNN.pth'), map_location=device))

    with torch.no_grad():
        test_img, label = testset[0]
        input_tensor = torch.unsqueeze(torch.tensor(test_img), dim=0)
        input_tensor = input_tensor.permute(0, 3, 1, 2).to(device)

        pred = torch.argmax(model(input_tensor), dim=-1)

        prev_letter = pred[0].item()
        pred_word = ''

        for letter in pred:
            if letter.item() != 0 and letter.item() != prev_letter:
                pred_word += list(testset.BOW.keys())[letter.item()]

            prev_letter = letter.item()

        plt.imshow(test_img)
        plt.title('prediction: ' + pred_word)
        plt.show()
