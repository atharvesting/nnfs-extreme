import cv2 as cv
import serial
import numpy as np
from abc import ABC, abstractmethod

class Source(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
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
                 baud_rate = 921600, 
                 timeout = 2,
                 width: int = 1024, 
                 height: int = 768
                 ):
        
        self.ser = serial.Serial(port, baud_rate, timeout=timeout)
        self.last_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def get_frame(self):

        header = self.ser.read_until(b'START_IMG')
        if not header.endswith(b'START_IMG'):
            return self.last_frame

        size_bytes = self.ser.read(4)
        if len(size_bytes) < 4:
            return self.last_frame

        img_size = int.from_bytes(size_bytes, byteorder='little')
        raw_data = self.ser.read(img_size)
        if len(raw_data) < img_size:
            return self.last_frame

        if self.ser.read(7) == b'END_IMG':
            img_np = np.frombuffer(raw_data, dtype=np.uint8)
            img = cv.imdecode(img_np, cv.IMREAD_COLOR)

            if img is not None:
                self.last_frame = cv.flip(img, 0)

        return self.last_frame

    def release(self):
        self.ser.close()