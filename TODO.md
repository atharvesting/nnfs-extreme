# TODO.md for NNFS-Extreme (started 00:48 24/07/2026)

### In Progress

### Major
- [ ] Train and export a 42(10+32Noise)-128-256-512-784 topology model with LeakyRELU.
- [ ] Save 3 models for access on GitHub (10-30-784, 10-128-256-512-784, 42(10+32Noise)-128-256-512-784)

### Minor
- [ ] (NNFS_Extreme/NN.cpp) Explore multi-threading to handle multiple mini-batches concurrently.
- [ ] (NNFS_Extreme/NN.cpp) Change export datatype of parameter matrix dimensions from size_t to uint64_t or uint32_t

### Documentation
- [ ] (examples/export_model.hpp) Examples: Model binary Import/Export workflow.
- [ ] (README.md) Short example codeblock showcasing API.
- [ ] (README.md) Quickstart Guide.

### Backlog
- [ ] Additional storage for weights and biases for the best performing parameters.
- [ ] Matmul research for gemm: Block tiling, Right Matrix Transpose, Combinations etc.

### Completed ✓
- [x] (README.md) Major refactor to include MNIST-CV Extension. Use links for navigation. (2026-7-29)
- [x] Parallelize the training loop using threads. (2026-07-28)
- [x] (third_party/Spalten/) Implement GEMM operation (Task for Spalten) (2026-07-25)
- [x] (examples/inverted_mnist.hpp) Examples: Inverted data (digit2image) MNIST training and eval workflow. (2026-07-24)
- [x] (examples/standard_mnist.hpp) Examples: Standard data (image2digit) MNIST training and eval workflow. (2026-07-24)
- [x] (MNIST_CV/) Write foundational OpenCV code for hand detection and drawing. Live Video + Hand Landmarks + Drawing. (2026-07-24)
- [x] (NNFS_Extreme/NN.cpp) Set a file naming standard for model exports - `[data]_[topology-order]_ep[epochs]_lr[learning-rate]_l[loss(optional)].bin`. (2026-07-24)