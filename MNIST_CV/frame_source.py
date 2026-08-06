import cv2 as cv
import serial
import numpy as np
from enum import Enum
from abc import ABC

class Source(ABC):
    @abstractmethod
    def get_frame(self):
        pass
    @abstractmethod
    def release(self):
        pass

class CVFrameSource(Source):
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:

        self.cap = cv.VideoCapture(index)
        self.cap.set(3, width)
        self.cap.set(4, height)

        if not self.cap.isOpened():
            print(f"Couldn't open OpenCV camera.")
            exit()

    def get_frame(self):
        success, frame = self.cap.read()
        if success:
            return frame
        else:
            raise Exception

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()


class SerialFrameSource(Source):
    def __init__(self, 
                 port: str = 'COM6', 
                 baud_rate: int = 921600, 
                 timeout: int = 2, 
                #  width: int = 320, 
                #  height: int = 240
                 ):
        
        self.ser = serial.Serial(port, baud_rate, timeout)

    def get_frame(self):
        if self.ser.read(9) == b'START_IMG':
            size_bytes = self.ser.read(4)
            img_size = int.from_bytes(size_bytes, byteorder='little')
            raw_data = self.ser.read(img_size)

            if self.ser.read(7) == b'END_IMG':
                img_np = np.frombuffer(raw_data, dtype=np.uint8)
                img = cv.imdecode(img_np, cv.IMREAD_COLOR)

                return img

    def release(self):
        self.ser.close()