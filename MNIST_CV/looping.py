import nnfs_extreme as nnfs
import cv2 as cv
from hand_tracking import HandTracker
from image import process_images, to_flat_arrays
from numpy.typing import NDArray
from numpy import ndarray
from typing import Callable
from frame_source import Source, CVFrameSource, SerialFrameSource

def loop(func: Callable):

    def wrapper(source: Source):
        
        ht = HandTracker(num_hands=1)
        net = nnfs.Network("data/output/mnist_784-30-10_ep30_lr9p000.bin")

        while True:
            frame = source.get_frame()
            frame = cv.flip(frame, 1)
            
            frame = func(frame, ht, net)

            if not isinstance(frame, ndarray):
                raise TypeError("Function must return a numpy array.")

            cv.imshow('Camera', frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                ht.clear_sketch()
            elif key == ord('q'):
                break

        source.release()
        
    return wrapper


@loop
def pipe(frame: NDArray, width: int, height: int, ht: HandTracker, net: nnfs.Network) -> NDArray:

    ht.detect(frame)
    if ht.latest_result:
        frame = ht.draw_landmarks_on_image(frame, ht.latest_result) # type: ignore
    
    boxes: list[tuple[tuple[int, int], tuple[int, int]]] = ht.bounding_boxes()
    box_images = ht.get_box_images(frame, boxes)
    frame = ht.sketch(frame, boxes)
    process_images(box_images)
    arrays = to_flat_arrays(box_images)
    
    for i, array in enumerate(arrays):
        output: list[float] = net.feedforward(array)
        predicted: int = output.index(max(output))
        cv.putText(frame, str(predicted), (50 * (i + 1), 100), fontFace=1, fontScale=4, color=255, thickness=4)
        # print(f"Image {i+1}: digit = {predicted}  (confidence {max(output):.4f})")

    return frame
