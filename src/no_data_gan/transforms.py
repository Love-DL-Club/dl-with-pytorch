import torchvision.transforms as T

mean = (0.4914, 0.4822, 0.4465)
std = (0.247, 0.243, 0.261)


def create():
    return T.Compose(
        [
            T.RandomCrop((32, 32), padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
