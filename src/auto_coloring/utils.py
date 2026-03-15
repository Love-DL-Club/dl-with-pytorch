import cv2


def rgb2lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab2rgb(lab):
    return cv2.cvtColor(lab, cv2.COLOR_Lab2LRGB)
