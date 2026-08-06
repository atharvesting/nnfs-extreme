# Entry point of MNIST_CV

from looping import pipe
from frame_source import CVFrameSource, SerialFrameSource

if __name__ == "__main__":
    source = CVFrameSource()
    # source = SerialFrameSource()
    pipe(source)