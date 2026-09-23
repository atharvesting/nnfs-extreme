## Naive

- In the case of NxN matrix multiplication, accessing elements in the columns of the right matrix can consistently lead
to cache misses, as each column element is spaced N elements apart in memory.
- We access one element from each row in the right matrix at a time, which automatically loads the entire cache line on 
which that element is sitting. However, due to the nature of our access pattern, the rest of the elements loaded in the 
line are almost never used, leading to a significant cache miss rate.

## Blocked

- Blocking optimisation deals with the problem of inefficient cache line loading.