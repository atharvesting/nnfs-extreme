import nnfs_extreme as nnfs
import cv2 as cv
from hand_tracking import HandTracker
from image import process_images, to_flat_arrays
from numpy.typing import NDArray
from numpy import ndarray

def loop(func):
    def wrapper():

        cap = cv.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        if not cap.isOpened():
            print("Couldn't open camera.")
            exit()
        
        ht = HandTracker(num_hands=1)
        net = nnfs.Network("data/output/mnist_784-30-10_ep10_lr2p000.bin")

        while True:
            success, frame = cap.read()
            frame = cv.flip(frame, 1)

            if not success:
                print("Can't receive frame.")
                break

            frame = func(frame, ht, net)
            if not isinstance(frame, ndarray):
                raise TypeError("Function must return a numpy array.")

            cv.imshow('Camera', frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                ht.clear_sketch()
            elif key == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        
    return wrapper


@loop
def pipe(frame: NDArray, ht: HandTracker, net: nnfs.Network) -> NDArray:
    ht.detect(frame)
    if ht.latest_result:
        frame = ht.draw_landmarks_on_image(frame, ht.latest_result)
    boxes = ht.bounding_boxes()
    frame = ht.sketch(frame, boxes)
    box_images = ht.get_box_images(frame, boxes)
    process_images(box_images)
    arrays = to_flat_arrays(box_images)
    
    for i, array in enumerate(arrays):
        output = net.feedforward(array)
        predicted = output.index(max(output))
        print(f"Image {i+1}: digit = {predicted}  (confidence {max(output):.4f})")

    return frame
