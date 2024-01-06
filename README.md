# cudaTemp
temp cuda code
## Linear Multi-head Attention Kernel
 - shared memory: load data from global memory to shared memory collectively
 - double buffering: double shared memory to hide the latency of data loading
 - free atomic add: warp-level sync reduction sum
 - pack: pack 4 float data and unroll for-loop
