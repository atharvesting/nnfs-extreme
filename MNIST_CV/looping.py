import nnfs_extreme as nnfs
import cv2 as cv
from hand_tracking import HandTracker
from image import process_images, to_flat_arrays
from functools import wraps
import matplotlib.pyplot as plt
import time

def loop(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):

        cap = cv.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        if not cap.isOpened():
            print("Couldn't open camera.")
            exit()
        
        ht = HandTracker(num_hands=4)

        while True:
            success, frame = cap.read()
            frame = cv.flip(frame, 1)

            if not success:
                print("Can't receive frame.")
                break

            frame = func(*args, **kwargs)
            if not isinstance(frame, np.ndarray):
                raise TypeError("Function must return a numpy array.")

            cv.imshow('Camera', frame)

            if cv.waitKey(1) & 0xFF == ord('c'):
                ht.clear_sketch()
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        
    return wrapper

def start_loop():
    cap = cv.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Couldn't open camera.")
        exit()

    ht = HandTracker(num_hands=1)

    while True:
        success, frame = cap.read()
        frame = cv.flip(frame, 1)

        if not success:
            print("Can't receive frame.")
            break

        ht.detect(frame)
        if ht.latest_result:
            frame = ht.draw_landmarks_on_image(frame, ht.latest_result)
        boxes = ht.bounding_boxes()
        frame = ht.sketch(frame, boxes)
        cv.imshow('Camera', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            ht.clear_sketch()
        elif key == ord('q'):
            break

    success, frame = cap.read()
    

    # i = 1
    # for img in box_images:
    #     cv.imwrite(f"data/output/img_{i}.png", img)
    #     i += 1
    
    cap.release()
    cv.destroyAllWindows()

    frame = cv.flip(frame, 1)
    boxes = ht.bounding_boxes()
    frame = ht.sketch(frame, boxes)
    box_images = ht.get_box_images(frame, boxes)
    process_images(box_images)

    arrays = to_flat_arrays(box_images)
    net = nnfs.Network("data/output/mnist_784-30-10_ep10_lr2p000.bin")
    for i, array in enumerate(arrays):
        output = net.feedforward(array)           # list[float] of length 10
        print(output)
        predicted = output.index(max(output))     # index of highest activation = predicted digit
        print(f"Image {i+1}: digit = {predicted}  (confidence {max(output):.4f})")


    plt.subplot(1, 2, 1)
    plt.imshow(arrays[0].reshape(28, 28), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(arrays[1].reshape(28, 28), cmap="gray")
    plt.show()