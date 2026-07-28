# Notes on Performance (started 2026-07-25 01:15)

2026-08-28
---
- The results and changes in each epoch are dependent on the one before it. The same goes for each mini-batch. This is why
multithreading could only be implemented on the level of the training samples in each mini-batch, where the samples where 
distributed among threads.
- A particular and difficult challenge that arose with the idea of concurrency was managing activation, z's, and nabla (w&b) buffers.
Since each group of training samples would accumulate their own gradient, each thread needed to have access to their
own group of these 4 buffers. This effectively (thread_count)x'd the memory requirements (in my case, about 16x), which can scale significantly with heavier topologies, but this was a classic memory space/time tradeoff I was willing to make due to the potential for extreme training time improvements. The fact that this extra memory was designed to be reused throughout the entire training process softens the blow.
- This specific application of a Neural network, where the each individual job of training through samples has a relatively short runtime
is what thread pooling is best for. Thread pooling eliminates the significant spawning overhead by initializing a group of threads upfront
and throwing tasks at them continuously.
- Given the sequential epoch and mini-batch dependency, it is also important to ensure that all threads are done with working on their
set of training samples before moving on to the next mini-batch.
- Using a custom (see NNFS_Extreme\utils.hpp for code source) `ThreadPool` class, pooling of threads could be initialized and set up 
outside the epoch loop itself to be used throughout the training process, effectively eliminating the primary setback associated with
concurrent programs.
- The results weren't expected...

```Output
Epoch 0: 7948 / 10000 in 2.0100036 seconds/epoch
Epoch 1: 8281 / 10000 in 1.951465 seconds/epoch
Epoch 2: 8378 / 10000 in 1.9267082 seconds/epoch
Epoch 3: 9001 / 10000 in 1.92471 seconds/epoch
Epoch 4: 9095 / 10000 in 1.9181187 seconds/epoch
Epoch 5: 9124 / 10000 in 1.9151405 seconds/epoch
Epoch 6: 9182 / 10000 in 1.9521091 seconds/epoch
Epoch 7: 9247 / 10000 in 1.9548612 seconds/epoch
Epoch 8: 9275 / 10000 in 1.9587994 seconds/epoch
Epoch 9: 9289 / 10000 in 1.9639766 seconds/epoch
Epoch 10: 9300 / 10000 in 1.9599723 seconds/epoch
Epoch 11: 9314 / 10000 in 1.957583 seconds/epoch
Epoch 12: 9339 / 10000 in 1.9530449 seconds/epoch
Epoch 13: 9351 / 10000 in 1.9528309 seconds/epoch
Epoch 14: 9354 / 10000 in 1.9539362 seconds/epoch
Epoch 15: 9357 / 10000 in 1.9579301 seconds/epoch
Epoch 16: 9377 / 10000 in 1.9591134 seconds/epoch
Epoch 17: 9380 / 10000 in 1.9638623 seconds/epoch
Epoch 18: 9392 / 10000 in 1.9624003 seconds/epoch
Epoch 19: 9401 / 10000 in 1.9638792 seconds/epoch
```

2026-07-25
---
- Using a reusable buffer for storing Mini-batches (no improvements)
    - Previously, a container supposed to hold a batch of training samples was being declared in every epoch loop along with memory reservation.
    - Since the size of the mini-batches is known at the time of function call, this container should easily be able to be declared and memory could be reserved outside the main epoch loop.

2026-07-25 (Midnight)
---
Matmul:
- Switching from debug to release mode led to a 7.5x performance increase or approx. 86% reduction in training time per epoch. 
This improvement is unsurprising but worth noting. (01:15)
- Using the ikj matmul form over ijk led to almost 1.8x performance or approx. 45% reduction in training time per epoch. (01:17)
- TODO: Naive operation combinations (z = w * a + b) Vs gemm function (z = gemm(alpha, w, a, beta, b, z)). (01:18)

#### As of 2026-07-25 01:21, with the following configuration:
- Fixed:
    - i5-12500H Laptop CPU (Silent Mode)
    - 3200 MT/s Primary Memory
    - MNIST dataset
    - Topology = 784-Input -> 30-Hidden -> 10-Output
    - Release mode
- Variable:
    - Epochs = 30
    - Mini-batch size = 10
    - matmul using naive ikj
    - gemm() function for inplace
    - Reusable buffers for nabla, activations and zs

...the average training time per epoch is approx. 4.25s, leading to a total training time of 4.25 * 30 = 128 seconds.
Considering Nielsen (see README.md) 
[estimated](http://neuralnetworksanddeeplearning.com/chap1.html#:~:text=Note%20that%20if%20you%27re%20running%20the%20code%20as%20you%20read%20along%2C%20it%20will%20take%20some%20time%20to%20execute%20%2D%20for%20a%20typical%20machine%20%28as%20of%202015%29%20it%20will%20likely%20take%20a%20few%20minutes%20to%20run%2E)
the training time to be around a few minutes (I will assume 3 minutes) to execute SGD() on a 2015 machine with normal specs as well a
Python-Numpy stack, I should be able to target a 2x performance improvement for my future iterations.