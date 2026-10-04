# Notes on Performance (started 2026-07-25 01:15)

2026-10-4
---
- Reached 98.20% accuracy on a configuration I tried on impulse before running my big bad benchmark.
    ```output
    Epoch 0: 8611 / 10000 in 4.435637283 seconds/epoch
    Epoch 5: 9597 / 10000 in 4.4374465125 seconds/epoch
    Epoch 10: 9712 / 10000 in 4.436275765090909 seconds/epoch
    Epoch 15: 9798 / 10000 in 4.4490212530625 seconds/epoch
    Epoch 20: 9820 / 10000 in 4.44792334747619 seconds/epoch
    Epoch 25: 9789 / 10000 in 4.454295043923077 seconds/epoch
    Epoch 29: 9815 / 10000 in 4.4549883968 seconds/epoch
    ====== Benchmark Results ======
    Configuration
    Epochs: 30
    Mini-batch size: 32
    Learning rate (eta): 9
    Topology: 784 -> 512 -> 128 -> 30 -> 10
    Threading: Single
    Thread count: 16
    Results (seconds unless noted)
    Total time: 133.622
    Total training time: 124.207
    Total testing time: 9.415
    Average training time/epoch: 4.140
    Average testing time/epoch: 0.314
    Maximum accuracy: 98.20%
    Final accuracy: 98.15%
    Accuracy threshold: 94.00%
    Epochs to threshold: 3
    Time to threshold: 13.175 s
    ```
- This configuration will definitely go in my benchmark harness and replace {784, 512, 512, 10} on account of mediocre
performance in most runs.
- I ran the benchmark overnight and got to configuration number 646/1152 after 9 hours. I terminated it there
because I needed to work on the laptop. The way the odometer/cartesian product generator works, I can simply plug in
the last indices state (for eg. {2, 1, 2, 0, 2, 0, 1}) and have it resume from there. The key realization is that I only
reached configuration no. 136 last time with my totally flawed evaluation code.

    ```
    Config no = 638, Time = 31411.2
    {2, 1, 2, 0, 0, 0, 1}
    Config no = 639, Time = 31524.1
    {2, 1, 2, 0, 0, 1, 0}
    Config no = 640, Time = 31641.1
    {2, 1, 2, 0, 0, 1, 1}
    Config no = 641, Time = 31754.9
    {2, 1, 2, 0, 1, 0, 0}
    Config no = 642, Time = 31874.2
    {2, 1, 2, 0, 1, 0, 1}
    Config no = 643, Time = 31987.3
    {2, 1, 2, 0, 1, 1, 0}
    Config no = 644, Time = 32104
    {2, 1, 2, 0, 1, 1, 1}
    Config no = 645, Time = 32216.8
    {2, 1, 2, 0, 2, 0, 0}
    Config no = 646, Time = 32333.4
    {2, 1, 2, 0, 2, 0, 1}
    ```

2026-10-03
---
- The overnight benchmark made the testing bottleneck impossible to ignore. After about 8 hours, the harness was
still at config no. 136. With the heavier 784-512-512-10 topology, each config was taking about 600 seconds for
30 epochs, with testing alone taking about 374 seconds compared to 223 seconds of training. Testing was eating
about 63% of the total runtime. The execution policy changes from September helped, but didn't address the
actual structure of evaluation.
- The problem was that `evaluate()` fed each of the 10,000 test images through the network individually after
every epoch. That is 300,000 separate forward passes per config. Every layer multiplied its weights by a
single-column activation matrix, allocated intermediate matrices and applied the activation function, only to
repeat all of this for the next image. The innermost column loop in the ikj/gemm implementation had just one
column to work with, which wasted the opportunity for contiguous operations and weight reuse across samples.
- Simply changing execution profiles couldn't solve this. `SGD()` kept the matrix backend on `Execution::Single`,
and even if that were changed, these individual matrix-vector operations were below Spalten's 4-million-operation
parallel dispatch threshold. The largest one was only 512 * 784 * 1 = 401,408.
- The fix was to batch evaluation into groups of 128 images, stacking each image into a column of a 784x128
input matrix. My existing `feedforward()` already supported multiple columns and bias broadcasting, so no new
inference implementation was required. The resulting output is 10x128, where each column gets its own argmax
and label comparison. The final batch uses its actual size (16 images for MNIST), and predictions are counted
directly instead of being stored in another vector for a second pass. The NaN/infinity checks were retained.
- This brings the forward-pass count down from 10,000 to 79 per epoch. The mathematical workload still exists,
but each weight is now reused across multiple images and the inner column loop has enough contiguous work to
benefit from vectorization. This improvement was achieved with sequential evaluation itself.
- The isolated Release check on 10,000 MNIST images using the 784-512-512-10 topology gave:

| Metric | Individual Evaluation | Batched Evaluation (128) | Improvement |
| --- | --- | --- | --- |
| Evaluation Time | 12.809 seconds | 0.438 seconds | ~29.3x Faster (-96.6%) |

- This check used the same initialized weights for both paths. Predictions were also compared across all four
activations on 137 images (including a partial batch), and the full 10,000-image check matched accuracy for the
largest topology.
- A subsequent full 30-epoch run with `Config(30, 32, 0.1F, {784, 512, 512, 10})` gave:

    ```output
    Total time: 240.892
    Total training time: 227.858
    Total testing time: 13.034
    Average training time/epoch: 7.595
    Average testing time/epoch: 0.434
    Maximum accuracy: 93.17%
    Final accuracy: 93.17%
    ```

- Compared to the roughly 600-second configs observed overnight, this is about a 60% reduction in total
training + testing time. This isn't a controlled accuracy comparison because the learning rate/configs differ.
The isolated evaluation check is the stronger evidence for the speedup. Training itself wasn't accelerated by
this fix; removing most of the testing cost is what drastically reduced total time. New records carry
`evaluation_method = "batched_128_v1"` to distinguish their testing times from the old baseline.

2026-09-26
---
- New 98% Milestone using the same 784-128-30-10 topology:
    ```Output
    Epoch 0: 9272 / 10000 in 1.228516 seconds/epoch
    Epoch 5: 9701 / 10000 in 3.196187 seconds/epoch
    Epoch 10: 9736 / 10000 in 3.4197307 seconds/epoch
    Epoch 15: 9777 / 10000 in 3.4835584 seconds/epoch
    Epoch 20: 9779 / 10000 in 3.500495 seconds/epoch
    Epoch 25: 9800 / 10000 in 3.4994333 seconds/epoch
    Epoch 29: 9799 / 10000 in 3.5373962 seconds/epoch
    ```
- An important bottleneck I was able to identify using the observability extension was the unnaturally high
testing time. With the standard configuration (which has the topology 784-128-30-10 now), testing time took
about 70% more time than training. Some light research was enough to classify this as an anomaly.
- Using `perf`, it was found out that despite being on single-threaded mode, threads were being spawned from
an unlikely source being `activation_functions.hpp`, which used the `std::execution::par_unseq` policy to apply
any activation function for an output matrix of any size.
- Simply switching to `std::execution::seq` shaved training time by 5% and testing time by a cool 24%. For the
standard config, that is a total time of 109s to 93s (17% reduction). The clear lesson is to use execution policies
carefully and prove its need before settling. In our case, the matrix sizes involved didn't justify spawning a dozen
threads that just ended up stuck on wait.

    ```output
    Epoch 0: 9261 / 10000 in 3.1482703130000003 seconds/epoch
    Epoch 5: 9736 / 10000 in 3.1054950236666667 seconds/epoch
    Epoch 10: 9742 / 10000 in 3.1281526302727274 seconds/epoch
    Epoch 15: 9767 / 10000 in 3.105608385 seconds/epoch
    Epoch 20: 9791 / 10000 in 3.1034742635238097 seconds/epoch
    Epoch 25: 9797 / 10000 in 3.096594653923077 seconds/epoch
    Epoch 29: 9792 / 10000 in 3.0955425474333333 seconds/epoch
    ====== Benchmark Results ======
    Configuration
        Epochs: 30
        Mini-batch size: 32
        Learning rate (eta): 9
        Topology: 784 -> 128 -> 30 -> 10
        Threading: Single
        Thread count: 0
    Results (seconds unless noted)
        Total time: 92.789
        Total training time: 34.408
        Total testing time: 58.381
        Average training time/epoch: 1.147
        Average testing time/epoch: 1.946
        Maximum accuracy: 98.05%
        Final accuracy: 97.92%
        Accuracy threshold: 94.00%
        Epochs to threshold: 3
        Time to threshold: 9.288 s
    ```

2026-09-22
---
- Here are the results of the NN with a 784-128-30-10 topology:
    ```Output
    Epoch 0: 9207 / 10000 in 3.400761 seconds/epoch
    Epoch 5: 9692 / 10000 in 3.4902685 seconds/epoch
    Epoch 10: 9741 / 10000 in 3.5094476 seconds/epoch
    Epoch 15: 9746 / 10000 in 3.5133014 seconds/epoch
    Epoch 20: 9763 / 10000 in 3.5174725 seconds/epoch
    Epoch 25: 9752 / 10000 in 3.5140538 seconds/epoch
    Epoch 29: 9789 / 10000 in 3.5280585 seconds/epoch
    Total Training (+ Testing) Time = 165.714
    ```
- The much higher parameter count (~100k vs ~22k) led to noticeable
improvement in classification accuracy to 97.89% (+2.3% of 728-30-10). The reason
can likely be attributed to the denser net being able to track more complex
patterns in the digits, leading to a higher success rate with edge cases.
- The time performance cost is real, but expected. A delta of +2.9s per epoch.


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
