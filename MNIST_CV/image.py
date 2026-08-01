# The image processing pipeline will reside here
import cv2 as cv
import numpy as np

def color_correction(frame):

    green_mask = np.all(frame == [0, 255, 0], axis=-1)
    # Convert all Non-drawn pixels to white
    frame[green_mask] = [0, 0, 0]
    # Convert all drawn pixels to black
    frame[~green_mask] = [255, 255, 255]

    return frame

def blur(frame):
    return cv.blur(frame, (5, 5))

def pixelate(frame, height = 28, width = 28):
    pixelated = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)
    return pixelated

def pipeline(img, func_list):
    for func in func_list:
        img = func(img)

    return img

def process_images(img_list):
    for i in range(len(img_list)):
        img_list[i] = pipeline(img_list[i], [color_correction, blur, pixelate])