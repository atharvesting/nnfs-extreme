# TODO.md for NNFS-Extreme (started 00:48 24/07/2026)

### In Progress

### Major
- [ ] Train and export a 42(10+32Noise)-128-256-512-784 topology model with LeakyRELU.
- [ ] Save 3 models for access on GitHub (10-30-784, 10-128-256-512-784, 42(10+32Noise)-128-256-512-784)

### Minor
- [ ] (NNFS_Extreme/NN.cpp) Change export datatype of parameter matrix dimensions from size_t to uint64_t or uint32_t

### Documentation
- [ ] (examples/standard_mnist.hpp) Examples: Standard (image-digit) MNIST training and eval workflow.
- [ ] (examples/standard_mnist.hpp) Examples: Inverted data (digit-image) MNIST training and eval workflow.
- [ ] (examples/export_model.hpp) Examples: Model binary Import/Export workflow.
- [ ] (README.md) Short example codeblock showcasing API.

### Backlog
- [ ] Additional storage for weights and biases for the best performing parameters.

### Completed ✓
- [x] (NNFS_Extreme/NN.cpp) Set a file naming standard for model exports - `[data]_[topology-order]_ep[epochs]_lr[learning-rate]_l[loss(optional)].bin` (2026-07-24)