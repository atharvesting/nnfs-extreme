import cv2 as cv
from frame_source import SerialFrameSource # TODO: Fix import

ser = SerialFrameSource()
print("Connection established!")

while True:
    frame = ser.get_frame()
    cv.imshow("ESP32 Stream", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break