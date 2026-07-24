# Notes on Performance (started 2026-07-25 01:15)

2026-07-25
---
Matmul:
- Switching from debug to release mode led to a 7.5x performance increase or approx. 86% reduction in training time per epoch. 
This improvement is unsurprising but worth noting. (01:15)
- Using the ikj matmul form over ijk led to almost 1.8x performance or approx. 45% reduction in training time per epoch. (01:17)
- TODO: Naive operation combinations (z = w * a + b) Vs gemm function (z = gemm(alpha, w, a, beta, b, z)). (01:18)

#### As of 2026-07-25 01:21, with the following configuration:
- Fixed:
    - i5-12500H Laptop CPU
    - 3200 MT/s Primary Memory
    - Mini-batch size = 10
    - MNIST dataset
    - Topology = 784-Input -> 30-Hidden -> 10-Output
    - Release mode
- Variable:
    - matmul using naive ikj
    - gemm() function for inplace
    - Reusable buffers for nabla, activations and zs

...the average training time per epoch is approx. 4.25s, leading to a total training time of 4.25 * 30 = 128 seconds.
Considering Nielsen (see README.md) 
[estimated](http://neuralnetworksanddeeplearning.com/chap1.html#:~:text=Note%20that%20if%20you%27re%20running%20the%20code%20as%20you%20read%20along%2C%20it%20will%20take%20some%20time%20to%20execute%20%2D%20for%20a%20typical%20machine%20%28as%20of%202015%29%20it%20will%20likely%20take%20a%20few%20minutes%20to%20run%2E)
the training time to be around a few minutes (I will assume 3 minutes) to execute SGD() on a 2015 machine with normal specs as well a
Python-Numpy stack, I should be able to target a 5x performance improvement for my future iterations.

