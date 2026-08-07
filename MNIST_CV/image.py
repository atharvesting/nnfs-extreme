# The image processing pipeline will reside here
from typing import Callable
import cv2 as cv
import numpy as np
from numpy.typing import NDArray

def color_correction(frame: NDArray) -> NDArray:
    green_mask = np.all(frame == [0, 255, 0], axis=-1)
    # MNIST convention: digit strokes = white (255), background = black (0)
    frame[green_mask]  = [255, 255, 255]  # drawn stroke → white
    frame[~green_mask] = [0, 0, 0]        # background   → black

    return frame

def grayscale(frame: NDArray) -> NDArray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def blur(frame: NDArray) -> NDArray:
    return cv.blur(frame, (8, 8))

def pixelate(frame: NDArray, height: int = 28, width: int = 28) -> NDArray:
    pixelated: NDArray = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)
    return pixelated

def pipeline(frame: NDArray, func_list: list[Callable]) -> NDArray:
    for func in func_list:
        frame = func(frame)
    return frame

def process_images(frame_list: list[NDArray]) -> None:
    for i in range(len(frame_list)):
        frame_list[i] = pipeline(frame_list[i], [color_correction, grayscale, blur, pixelate])

def to_flat_arrays(frame_list: list[NDArray]) -> list[NDArray]:
    flat_list: list[NDArray] = []
    for frame in frame_list:
        flat_list.append((frame.flatten() / 255.0).astype(np.float32))
    return flat_list

def to_binary(flat_list: list[np.ndarray]) -> None:
    i: int = 1
    for array in flat_list:
        array.tofile(f"data/output/binary{i}.bin")