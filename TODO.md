# TODO.md for NNFS-Extreme (started 00:48 24/07/2026)

### In Progress


### Major
- [ ] Train and export a 42(10+32Noise)-128-256-512-784 topology model with LeakyRELU.
- [ ] Save 3 models for access on GitHub (10-30-784, 10-128-256-512-784, 42(10+32Noise)-128-256-512-784)

### Minor
- [ ] Implement variable stroke width to make digit images more consistent irrespective of sketch size.

### Documentation
- [ ] (README.md) Add the ESP32-MNIST Section.
- [ ] (README.md) Quickstart Guide.

### Backlog
- [ ] Matmul research for gemm: Block tiling, Right Matrix Transpose, Combinations etc.

### Completed ✓
- [x] (MNIST_CV/frame_source.py) Fix the ESP32 CAM feed compatibility with MNIST_CV. (2026-08-07)
- [x] (NNFS_Extreme/NN.cpp) Fix the import NN constructor to properly accomodate uint16_t data. (2026-08-07)
- [x] (zPERF_NOTES.md) Fix the accuracy issue with training the standard neural network. (2026-08-06)
- [x] (MNIST_CV/frame_source.py) Update MNIST_CV to support modularity concerning video input: CV2 and ESP32-S3 CAM (2026-08-06)
- [x] (ESP32_MNIST) Set up the ESP32 subproject using platformio. (2026-08-06)
- [x] (NNFS_Extreme/NN.cpp) Change export datatype of parameter matrix dimensions from size_t to uint16t. (2026-08-06)
- [x] (MNIST_CV/looping.py) Add ordered predictions in the Webcam UI. (2026-08-02)
- [x] (README.md) Architecture diagrams and directory structure. (2026-08-02)
- [x] (MNIST_CV/looping.py) Integrate classification inference in the live loop. (2026-08-02)
- [x] Working binding between NNFS_E and MNIST_CV along with successful inference from drawn digits. (2026-08-01)
- [x] (MNIST_CV/image.py) Implement the image processing pipeline to make drawings MNIST-ready. (2026-08-01)
- [x] Add support for independent polylines, bounding boxes, and performance improvements. (2026-07-31)
- [x] (README.md) Short example codeblock showcasing API. (2026-07-30)
- [x] (examples/export_model.hpp) Examples: Model binary Import/Export workflow. (2026-07-29)
- [x] (README.md) Major refactor to include MNIST-CV Extension. Use links for navigation. (2026-07-29)
- [x] Parallelize the training loop using threads. (2026-07-28)
- [x] (third_party/Spalten/) Implement GEMM operation (Task for Spalten) (2026-07-25)
- [x] (examples/inverted_mnist.hpp) Examples: Inverted data (digit2image) MNIST training and eval workflow. (2026-07-24)
- [x] (examples/standard_mnist.hpp) Examples: Standard data (image2digit) MNIST training and eval workflow. (2026-07-24)
- [x] (MNIST_CV/) Write foundational OpenCV code for hand detection and drawing. Live Video + Hand Landmarks + Drawing. (2026-07-24)
- [x] (NNFS_Extreme/NN.cpp) Set a file naming standard for model exports - `[data]_[topology-order]_ep[epochs]_lr[learning-rate]_l[loss(optional)].bin`. (2026-07-24)