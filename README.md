# NNFS Extreme

> NNFS from scratch. Actually.

This is an educational project that aimed at building a neural network starting by literally writing the dot product function :)
and building my way from the bottom up towards a performant NN that can classify MNIST digit pictures. The foundational math libary used for 
this project was my very own linear algebra library Spalten, built for the very purpose of using it to write neural networks. The main offering
of this library is the Matrix class template, that is optimised and accelerated for great (i hope) performance using fast algorithms and advanced 
C++ features. Check the library out [here](https://github.com/atharvesting/spalten-linalg-library).

This project is heavily inspired by Michael Nielsen's [book](http://neuralnetworksanddeeplearning.com/) on Neural Networks and Deep Learning,
as I have not only used this book to learn about deep learning, but also translated its implementation to blazing fast C++. As this is the 
first time I was learning about neural networks, I am perfectly happy with this approach.

The project is part of a mult-layered and multi-faceted progression, which was conceptualized by me to take a journey through low-level C++, 
ML mathematics, Concurrency, Computer Vision, all the way to resource-efficient-aggresive TinyML.

- [x] Phase 1: Spalten (Foundational Linear Algebra Library)
- [ ] Phase 2: NNFS-Extreme (Neural Network trained on the MNIST dataset) -> Built on top of Spalten
- [ ] Phase 3: MNIST-CV (CV layer that classifies digits drawn using gestures) -> NNFS-E as the engine
- [ ] Phase 4: ESP32-MNIST (The entire pipeline running on an ESP32-S3 Sense) -> Using all of the above

More about this to be documented soon!

---

I would like to be absolutely transparent with the fact that I used AI tools for debugging and helping me understand how Python translates 
optimally to C++. I also used it to build the scripts that extracted the zipped MNIST data and prepare it to be used by the NN. However, I 
made sure to understand each line of code and comment it throughout to document my understanding. I regularly post blogs and try to put this 
understanding into my own words for everyone.