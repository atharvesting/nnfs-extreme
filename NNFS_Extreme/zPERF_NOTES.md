# Notes on Performance (started 2026-07-25 01:15)

2026-09-06
---

- Noticed that training using the following hyperparameters (epochs=30, batch_size=32, eta=3.0) led to a maximum of 85%
accuracy on the test dataset. This change was unexpected because the network had already demonstrated over 95% accuracy
in previous iterations.
- I noticed that my previous iterations used a batch size = 10 and eta = 3.0 easily led to 94%+ accuracy (this was before
the threadpool update) and upon trying the same hyperparameters for the new version, I got back to the 94-95% accuracy region.
The problem was clearly not the program, but the amount of additional iterations the 32-size version required to reach similar
levels. For now, the steps towards valleys were quite small. Step size = 3.0 (eta) / 32 = 0.09375 compared to 
step size = 3.0 / 10 = 0.3, 
which is a 3.2x larger step size, leading to decisive and impactful steps towards the minimum-loss regions. 
- If distance travelled towards hyperplane minima is approximated by epochs * step size, where step size = eta / batch size,
then the solution could be reached from two different angles - multiplying either the epochs or eta by ~3.2. Theoretically,
both ways will lead to favourable accuracy by either giving the small steps enough iterations to accumulate towards minima or
making bigger steps in the first place.
- Right now, batch size = 32 results in sub-2 second epochs as a result of the threadpool (about 4 seconds when batch size = 10
due to inefficient thread use) and increasing epochs by 3x would essentially nullify its benefits. scaling the eta is the superior
option due to its essentially zero additional cost.
- The results speak for themselves:

| Metric | (1) Batch Size = 32 ($\eta = 3.0$) | (2) Batch Size = 10 ($\eta = 3.0$) | (3) Batch Size = 32 ($\eta = 9.0$) | Improvement |	
| --- |	--- | --- | --- | --- |
Max Accuracy | ~85% | 94.9% | 95.3% | +12% Accuracy comp. to (1) |
Training Time | 80 seconds | 182 seconds	| 80 seconds | 2.27x Faster (-56%) comp. to (2) |

- The minimal accuracy improvement is definitely a matter of chance. I did not benchmark the two versions properly to draw
conclusions about the accuracy delta. But the training time is another matter entirely.
- Going from bs = 10 to bs = 32 also significantly reduces the number of updates made to the weights and biases after each
batch, about 3.2x fewer times. this contributes to decreasing cpu overhead.


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