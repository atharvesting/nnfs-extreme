import cv2 as cv
from hand_tracking import HandTracker
from functools import wraps

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
        frame = ht.sketch(frame)
        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('c'):
            ht.clear_sketch()
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

start_loop()