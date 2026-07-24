import cv2 as cv
from hand_tracking import HandTracker

def start_loop():
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

        results = ht.detect(frame)
        ht.process_result(results, frame)
        frame = ht.draw_landmarks_on_image(frame, results)
        frame = ht.sketch(frame)
        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('c'):
            ht.clear_sketch()
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

start_loop()