# NNFS Extreme · MNIST CV

The project-group is part of a mult-layered and multi-faceted progression, which was conceptualized by me to take a journey through low-level C++, ML mathematics, Concurrency, Computer Vision, all the way to resource-efficient-aggresive TinyML.

- [x] Phase 1: [Spalten](https://github.com/atharvesting/spalten-linalg-library) (Foundational Linear Algebra Library)
- [x] Phase 2: [NNFS-Extreme](#nnfs-extreme) (Neural Network trained on the MNIST dataset) -> Built on top of Spalten
- [ ] Phase 3: [MNIST-CV (WIP)](#mnist-cv) (CV layer that classifies digits drawn using gestures) -> NNFS-E as the engine
- [ ] Phase 4: ESP32-MNIST (The entire pipeline running on an ESP32-S3 Sense) -> Using all of the above

---

## NNFS-Extreme

> NNFS from scratch. Actually.

This is an educational project (MVP completed) aimed at building a neural network by literally starting with the dot product function :)
and building my way from the bottom up towards a performant NN that can classify MNIST digit pictures. However, this engine can generally work on for any dataset with some light tweaking.

This project is heavily inspired by [Michael Nielsen](https://michaelnielsen.org/)'s [book on Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) ,
as I have not only used this book to learn about deep learning, but also translated its implementation to blazing fast C++. As this is the 
first time I was learning about neural networks, I am perfectly happy with this approach. If you feel curious about my journey of 
improving training speed and inference, see the notes I kept on performance (I started tracking this too late `sad_emoji`).

After successfully achieving 95% accuracy in digit classification, as an experiment, I swapped the input and outputs of the training data and got the network (with only 1 hidden layer containing 30 neurons!) to output some really nice images ([visualized using Python](scripts/pixelarray_to_image.py)) of digits!

<table width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="assets/output_6.png" width="90%" alt="Neural net generates a 6!">
    </td>
    <td align="center" width="50%">
      <img src="assets/output_9.png" width="90%" alt="Neural net generates a 9!">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Definitely not an upside down 9</b>
    </td>
    <td align="center">
      <b>Definitely a 9</b>
    </td>
  </tr>
</table>

### Standard Example
```cpp
int main() 
{
  int epochs = 20, mini_batch_size = 32;
  float learning_rate = 2.0F;

	auto training_data = MNIST_loader::load_training_data("data/mnist_train_images.bin", "data/mnist_train_labels.bin", 50000,  false);	// inverted_data = false
	
	auto test_data = MNIST_loader::load_test_data("data/mnist_test_images.bin", "data/mnist_test_labels.bin", 10000);

	auto nn = Network(std::vector<int>{ 784, 30, 10 });

	nn.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data);
}
```

### Spalten: Backbone of the Neural Network

The foundational math libary used for this project was my very own linear algebra library Spalten, built for the very purpose of using it to write neural networks. The main offering of that library is the Matrix class template, that is optimised and accelerated for great (i hope) performance using fast algorithms and advanced C++ features. Check the library (still WIP) out [here](https://github.com/atharvesting/spalten-linalg-library). Spalten and NNFS-Extreme are written with zero external dependencies.

The pickled and zipped data can be found inside Nielsen's own [repository](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz).

---

## MNIST-CV

This project puts the above Neural Network to test by making it classify digits from a stream of real time images. The images are sourced from a live camera feed that uses OpenCV and MediaPipe to track hand landmarks and draw the digits on an air canvas. The feed must be dealt with such that the individual digit drawings are detected, processed (this word is doing some heavy lifting) and sent to the network for classification. The primary language used is Python. I intend to use python bindings to handle digit image data transmission to my C++ Neural Network and back.

The idea for this project stemmed from my interest in Computer Vision as well as the desire to build upon the neural network in a way that expanded my technological know-how further.

---

I plan to write about this project in my blog which you can find on [Medium](https://atharvesting.medium.com/) or [Substack](https://atharvesting.substack.com/).


### References

- [MediaPipe Hand Landmarker Python Docs](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [Hand Landmarker Example Notebook](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
- [C++ Thread Pool Implementation](https://stackoverflow.com/a/32593825)
- [Cropping an Image](https://learnopencv.com/cropping-an-image-using-opencv/)
- [Replacing elements in NumPy](https://stackoverflow.com/questions/19666626/replace-all-elements-of-numpy-array-that-are-greater-than-some-value)
- [Pixelating an Image using OpenCV](https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python)
- [Blurring an Image using OpenCV](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_filtering/py_filtering.html)
- [Time Python Module docs](https://docs.python.org/3/library/time.html)

---
### AI use disclosure

I would like to be absolutely transparent with the fact that I used AI tools for debugging and helping me understand how Python translates 
optimally to C++. I also used it to build the scripts that extracted the zipped MNIST data and prepare it to be used by the NN. However, I 
made sure to understand each line of code and comment it throughout to document my understanding. I regularly post blogs and to put this 
understanding into my own words for everyone.

---

- Started: 2026-07-29
- Languages: C++, Python
