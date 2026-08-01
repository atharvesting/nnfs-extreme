# The image processing pipeline will reside here
import cv2 as cv
import numpy as np

def color_correction(frame):
    green_mask = np.all(frame == [0, 255, 0], axis=-1)
    # MNIST convention: digit strokes = white (255), background = black (0)
    frame[green_mask]  = [255, 255, 255]  # drawn stroke → white
    frame[~green_mask] = [0, 0, 0]        # background   → black

    return frame

def grayscale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def blur(frame):
    return cv.blur(frame, (8, 8))

def pixelate(frame, height = 28, width = 28):
    pixelated = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)
    return pixelated

def pipeline(img, func_list):
    for func in func_list:
        img = func(img)

    return img

def process_images(img_list):
    for i in range(len(img_list)):
        img_list[i] = pipeline(img_list[i], [color_correction, grayscale, blur, pixelate])

def to_flat_arrays(img_list: list[np.ndarray]):
    flat_list: list[np.ndarray] = []
    for img in img_list:
        flat_list.append((img.flatten() / 255.0).astype(np.float32))

    return flat_list

def to_binary(flat_list: list[np.ndarray]):
    i = 1
    for array in flat_list:
        array.tofile(f"data/output/binary{i}.bin")