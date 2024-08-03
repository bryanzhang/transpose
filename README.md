# transpose
矩阵转置优化


base:

Iters: 1000, Elapsed time: 6.20065 s


1_tilling （分为8x8分块处理，便于向量化加载和写入，但转置仍使用swap操作）:

Iters: 1000, Elapsed time: 3.9367 s


2_prefetch （分块处理前进行数据预取）:

Iters: 1000, Elapsed time: 3.90402 s


3_reduce_instructions （使用unpack和permute进行8x8小块的转置处理大幅减少指令数量）:

Iters: 1000, Elapsed time: 2.63604 s


prefetch不如预想的优化大，可能因为地址计算使用的指令较多。


后续优化：

1.使用avx512进行向量优化，并把分配内存对齐到缓存行；（某些AMD服务器不支持AVX512）

2.尽量使用指针而非i、j变量减少实际指令数量；

3.针对更大矩阵，使用huge page减少tlb miss；

4.直接使用汇编。
