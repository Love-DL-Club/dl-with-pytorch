import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.sgd import SGD
from torchvision.models.resnet import resnet18, resnet34

from lib.utils.device import available_device
from lib.utils.path import model_path
from no_data_gan.models import Generator


def main(epochs=500):
    device = available_device()

    teacher = resnet34(weights=None, num_classes=10)
    teacher.load_state_dict(torch.load(model_path('teacher.pth'), map_location=device))
    teacher.to(device)
    teacher.eval()

    student = resnet18(weights=None, num_classes=10)
    student.to(device)

    generator = Generator().to(device)

    criterion = nn.L1Loss()

    G_optim = Adam(generator.parameters(), lr=1e-3)
    S_optim = SGD(student.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

    for ep in range(epochs):
        for _ in range(5):
            noise = torch.randn(256, 256, 1, 1, device=device)

            S_optim.zero_grad()
            fake = generator(noise).detach()
            teacher_output = teacher(fake)
            student_output = student(fake)

            S_loss = criterion(student_output, teacher_output.detach())

            print(f'epoch{ep} S_loss: {S_loss}')
            S_loss.backward()
            S_optim.step()

        nosie = torch.randn(256, 256, 1, 1, device=device)
        G_optim.zero_grad()

        fake = generator(nosie)

        teacher_output = teacher(fake)
        student_output = student(fake)

        G_loss = -1 * criterion(student_output, teacher_output)

        G_loss.backward()
        G_optim.step()

        print(f'epoch{ep} G_loss: {G_loss}')
