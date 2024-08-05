#include <random>
#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>

constexpr int n = 1024;  // 矩阵大小, 需要是32的倍数，否则需要padding.
constexpr int digits = 10;
int matrix[n * n + 8 * 8];  // 8 * 8是为了避免预取失败.

void transpose() {
  // prefetch (0, 0)
  __builtin_prefetch(matrix, 0, 3);
  __builtin_prefetch(matrix + (1 << digits), 0, 3);
  __builtin_prefetch(matrix + (2 << digits), 0, 3);
  __builtin_prefetch(matrix +(3 << digits), 0, 3);
  __builtin_prefetch(matrix +(4 << digits), 0, 3);
  __builtin_prefetch(matrix +(5 << digits), 0, 3);
  __builtin_prefetch(matrix +(6 << digits), 0, 3);
  __builtin_prefetch(matrix +(7 << digits), 0, 3);

  for (int i = 0; i < n; i += 8) {
    // 两个矩阵互相做转置
    for (int j = 0; j < i; j += 8) {

      __m256i xvec0 = _mm256_load_si256((__m256i*)(matrix + (j << digits) + i));
      __m256i xvec1 = _mm256_load_si256((__m256i*)(matrix + ((j + 1) << digits) + i));
      __m256i xvec2 = _mm256_load_si256((__m256i*)(matrix + ((j + 2) << digits) + i));
      __m256i xvec3 = _mm256_load_si256((__m256i*)(matrix + ((j + 3) << digits) + i));
      __m256i xvec4 = _mm256_load_si256((__m256i*)(matrix + ((j + 4) << digits) + i));
      __m256i xvec5 = _mm256_load_si256((__m256i*)(matrix + ((j + 5) << digits) + i));
      __m256i xvec6 = _mm256_load_si256((__m256i*)(matrix + ((j + 6) << digits) + i));
      __m256i xvec7 = _mm256_load_si256((__m256i*)(matrix + ((j + 7) << digits) + i));

      __m256i t0 = _mm256_unpacklo_epi32(xvec0, xvec4);
      __m256i t1 = _mm256_unpackhi_epi32(xvec0, xvec4);
      __m256i t2 = _mm256_unpacklo_epi32(xvec1, xvec5);
      __m256i t3 = _mm256_unpackhi_epi32(xvec1, xvec5);
      __m256i t4 = _mm256_unpacklo_epi32(xvec2, xvec6);
      __m256i t5 = _mm256_unpackhi_epi32(xvec2, xvec6);
      __m256i t6 = _mm256_unpacklo_epi32(xvec3, xvec7);
      __m256i t7 = _mm256_unpackhi_epi32(xvec3, xvec7);

      __m256i w0 = _mm256_unpacklo_epi32(t0, t4);
      __m256i w1 = _mm256_unpackhi_epi32(t0, t4);
      __m256i w2 = _mm256_unpacklo_epi32(t1, t5);
      __m256i w3 = _mm256_unpackhi_epi32(t1, t5);
      __m256i w4 = _mm256_unpacklo_epi32(t2, t6);
      __m256i w5 = _mm256_unpackhi_epi32(t2, t6);
      __m256i w6 = _mm256_unpacklo_epi32(t3, t7);
      __m256i w7 = _mm256_unpackhi_epi32(t3, t7);

      __m256i x0 = _mm256_unpacklo_epi32(w0, w4);
      __m256i x1 = _mm256_unpackhi_epi32(w0, w4);
      __m256i x2 = _mm256_unpacklo_epi32(w1, w5);
      __m256i x3 = _mm256_unpackhi_epi32(w1, w5);
      __m256i x4 = _mm256_unpacklo_epi32(w2, w6);
      __m256i x5 = _mm256_unpackhi_epi32(w2, w6);
      __m256i x6 = _mm256_unpacklo_epi32(w3, w7);
      __m256i x7 = _mm256_unpackhi_epi32(w3, w7);

      __m256i y0 = _mm256_permute2x128_si256(x0, x1, 0x20);
      __m256i y1 = _mm256_permute2x128_si256(x2, x3, 0x20);
      __m256i y2 = _mm256_permute2x128_si256(x4, x5, 0x20);
      __m256i y3 = _mm256_permute2x128_si256(x6, x7, 0x20);
      __m256i y4 = _mm256_permute2x128_si256(x0, x1, 0x31);
      __m256i y5 = _mm256_permute2x128_si256(x2, x3, 0x31);
      __m256i y6 = _mm256_permute2x128_si256(x4, x5, 0x31);
      __m256i y7 = _mm256_permute2x128_si256(x6, x7, 0x31);

      __m256i yvec0 = _mm256_load_si256((__m256i*)(matrix + (i << digits) + j));
      __m256i yvec1 = _mm256_load_si256((__m256i*)(matrix + ((i + 1) << digits) + j));
      __m256i yvec2 = _mm256_load_si256((__m256i*)(matrix + ((i + 2) << digits) + j));
      __m256i yvec3 = _mm256_load_si256((__m256i*)(matrix + ((i + 3) << digits) + j));
      __m256i yvec4 = _mm256_load_si256((__m256i*)(matrix + ((i + 4) << digits) + j));
      __m256i yvec5 = _mm256_load_si256((__m256i*)(matrix + ((i + 5) << digits) + j));
      __m256i yvec6 = _mm256_load_si256((__m256i*)(matrix + ((i + 6) << digits) + j));
      __m256i yvec7 = _mm256_load_si256((__m256i*)(matrix + ((i + 7) << digits) + j));

      // store & prefetch ((j + 8), i)
      _mm256_store_si256((__m256i*)(matrix + (i << digits) + j), y0);
      __builtin_prefetch(matrix + ((j + 8) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 1) << digits) + j), y1);
      __builtin_prefetch(matrix + ((j + 9) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 2) << digits) + j), y2);
      __builtin_prefetch(matrix + ((j + 10) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 3) << digits) + j), y3);
      __builtin_prefetch(matrix + ((j + 11) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 4) << digits) + j), y4);
      __builtin_prefetch(matrix + ((j + 12) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 5) << digits) + j), y5);;
      __builtin_prefetch(matrix + ((j + 13) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 6) << digits) + j), y6);
      __builtin_prefetch(matrix + ((j + 14) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 7) << digits) + j), y7);
      __builtin_prefetch(matrix + ((j + 15) << digits) + i, 0, 3);

      __m256i tt0 = _mm256_unpacklo_epi32(yvec0, yvec4);
      __m256i tt1 = _mm256_unpackhi_epi32(yvec0, yvec4);
      __m256i tt2 = _mm256_unpacklo_epi32(yvec1, yvec5);
      __m256i tt3 = _mm256_unpackhi_epi32(yvec1, yvec5);
      __m256i tt4 = _mm256_unpacklo_epi32(yvec2, yvec6);
      __m256i tt5 = _mm256_unpackhi_epi32(yvec2, yvec6);
      __m256i tt6 = _mm256_unpacklo_epi32(yvec3, yvec7);
      __m256i tt7 = _mm256_unpackhi_epi32(yvec3, yvec7);

      __m256i ww0 = _mm256_unpacklo_epi32(tt0, tt4);
      __m256i ww1 = _mm256_unpackhi_epi32(tt0, tt4);
      __m256i ww2 = _mm256_unpacklo_epi32(tt1, tt5);
      __m256i ww3 = _mm256_unpackhi_epi32(tt1, tt5);
      __m256i ww4 = _mm256_unpacklo_epi32(tt2, tt6);
      __m256i ww5 = _mm256_unpackhi_epi32(tt2, tt6);
      __m256i ww6 = _mm256_unpacklo_epi32(tt3, tt7);
      __m256i ww7 = _mm256_unpackhi_epi32(tt3, tt7);

      __m256i xx0 = _mm256_unpacklo_epi32(ww0, ww4);
      __m256i xx1 = _mm256_unpackhi_epi32(ww0, ww4);
      __m256i xx2 = _mm256_unpacklo_epi32(ww1, ww5);
      __m256i xx3 = _mm256_unpackhi_epi32(ww1, ww5);
      __m256i xx4 = _mm256_unpacklo_epi32(ww2, ww6);
      __m256i xx5 = _mm256_unpackhi_epi32(ww2, ww6);
      __m256i xx6 = _mm256_unpacklo_epi32(ww3, ww7);
      __m256i xx7 = _mm256_unpackhi_epi32(ww3, ww7);

      __m256i yy0 = _mm256_permute2x128_si256(xx0, xx1, 0x20);
      __m256i yy1 = _mm256_permute2x128_si256(xx2, xx3, 0x20);
      __m256i yy2 = _mm256_permute2x128_si256(xx4, xx5, 0x20);
      __m256i yy3 = _mm256_permute2x128_si256(xx6, xx7, 0x20);
      __m256i yy4 = _mm256_permute2x128_si256(xx0, xx1, 0x31);
      __m256i yy5 = _mm256_permute2x128_si256(xx2, xx3, 0x31);
      __m256i yy6 = _mm256_permute2x128_si256(xx4, xx5, 0x31);
      __m256i yy7 = _mm256_permute2x128_si256(xx6, xx7, 0x31);

      // store && prefetch (i, (j + 8))
      _mm256_store_si256((__m256i*)(matrix + (j << digits) + i), yy0);
      __builtin_prefetch(matrix + (i << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 1) << digits) + i), yy1);
      __builtin_prefetch(matrix + ((i + 1) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 2) << digits) + i), yy2);
      __builtin_prefetch(matrix + ((i + 2) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 3) << digits) + i), yy3);
      __builtin_prefetch(matrix + ((i + 3) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 4) << digits) + i), yy4);
      __builtin_prefetch(matrix + ((i + 4) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 5) << digits) + i), yy5);
      __builtin_prefetch(matrix + ((i + 5) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 6) << digits) + i), yy6);
      __builtin_prefetch(matrix + ((i + 6) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 7) << digits) + i), yy7);
      __builtin_prefetch(matrix + ((i + 7) << digits) + j + 8, 0, 3);
    }

    // prefetch (0, i + 8)
    __builtin_prefetch(matrix + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (1 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (2 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (3 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (4 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (5 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (6 << digits) + (i + 8), 0, 3);
    __builtin_prefetch(matrix + (7 << digits) + (i + 8), 0, 3);

    // prefetch (i + 8, 0)
    __builtin_prefetch(matrix + ((i + 8) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 9) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 10) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 11) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 12) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 13) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 14) << digits), 0, 3);
    __builtin_prefetch(matrix + ((i + 15) << digits), 0, 3);

    // 相同矩阵内的处理.
    __m256i vec0 = _mm256_load_si256((__m256i*)(matrix + (i << digits) + i));
    __m256i vec1 = _mm256_load_si256((__m256i*)(matrix + ((i + 1) << digits) + i));
    __m256i vec2 = _mm256_load_si256((__m256i*)(matrix + ((i + 2) << digits) + i));
    __m256i vec3 = _mm256_load_si256((__m256i*)(matrix + ((i + 3) << digits) + i));
    __m256i vec4 = _mm256_load_si256((__m256i*)(matrix + ((i + 4) << digits) + i));
    __m256i vec5 = _mm256_load_si256((__m256i*)(matrix + ((i + 5) << digits) + i));
    __m256i vec6 = _mm256_load_si256((__m256i*)(matrix + ((i + 6) << digits) + i));
    __m256i vec7 = _mm256_load_si256((__m256i*)(matrix + ((i + 7) << digits) + i));

    __m256i t0 = _mm256_unpacklo_epi32(vec0, vec4);
    __m256i t1 = _mm256_unpackhi_epi32(vec0, vec4);
    __m256i t2 = _mm256_unpacklo_epi32(vec1, vec5);
    __m256i t3 = _mm256_unpackhi_epi32(vec1, vec5);
    __m256i t4 = _mm256_unpacklo_epi32(vec2, vec6);
    __m256i t5 = _mm256_unpackhi_epi32(vec2, vec6);
    __m256i t6 = _mm256_unpacklo_epi32(vec3, vec7);
    __m256i t7 = _mm256_unpackhi_epi32(vec3, vec7);

    __m256i w0 = _mm256_unpacklo_epi32(t0, t4);
    __m256i w1 = _mm256_unpackhi_epi32(t0, t4);
    __m256i w2 = _mm256_unpacklo_epi32(t1, t5);
    __m256i w3 = _mm256_unpackhi_epi32(t1, t5);
    __m256i w4 = _mm256_unpacklo_epi32(t2, t6);
    __m256i w5 = _mm256_unpackhi_epi32(t2, t6);
    __m256i w6 = _mm256_unpacklo_epi32(t3, t7);
    __m256i w7 = _mm256_unpackhi_epi32(t3, t7);

    __m256i x0 = _mm256_unpacklo_epi32(w0, w4);
    __m256i x1 = _mm256_unpackhi_epi32(w0, w4);
    __m256i x2 = _mm256_unpacklo_epi32(w1, w5);
    __m256i x3 = _mm256_unpackhi_epi32(w1, w5);
    __m256i x4 = _mm256_unpacklo_epi32(w2, w6);
    __m256i x5 = _mm256_unpackhi_epi32(w2, w6);
    __m256i x6 = _mm256_unpacklo_epi32(w3, w7);
    __m256i x7 = _mm256_unpackhi_epi32(w3, w7);

    __m256i y0 = _mm256_permute2x128_si256(x0, x1, 0x20);
    __m256i y1 = _mm256_permute2x128_si256(x2, x3, 0x20);
    __m256i y2 = _mm256_permute2x128_si256(x4, x5, 0x20);
    __m256i y3 = _mm256_permute2x128_si256(x6, x7, 0x20);
    __m256i y4 = _mm256_permute2x128_si256(x0, x1, 0x31);
    __m256i y5 = _mm256_permute2x128_si256(x2, x3, 0x31);
    __m256i y6 = _mm256_permute2x128_si256(x4, x5, 0x31);
    __m256i y7 = _mm256_permute2x128_si256(x6, x7, 0x31);


    // 向量化回写
    _mm256_store_si256((__m256i*)(matrix + (i << digits) + i), y0);
    _mm256_store_si256((__m256i*)(matrix + ((i + 1) << digits) + i), y1);
    _mm256_store_si256((__m256i*)(matrix + ((i + 2) << digits) + i), y2);
    _mm256_store_si256((__m256i*)(matrix + ((i + 3) << digits) + i), y3);
    _mm256_store_si256((__m256i*)(matrix + ((i + 4) << digits) + i), y4);
    _mm256_store_si256((__m256i*)(matrix + ((i + 5) << digits) + i), y5);;
    _mm256_store_si256((__m256i*)(matrix + ((i + 6) << digits) + i), y6);
    _mm256_store_si256((__m256i*)(matrix + ((i + 7) << digits) + i), y7);
  }
}

int main() {
  const int iters = 1000;

  void* p = (void*)matrix;
  std::cout << "Is aligned to cache line: " << (((size_t)p & 63L) == 0) << ", p=" << p << std::endl;
  double total = 0.0;
  // 生成并填充数组
  for(int i = 0; i < n * n; ++i) {
    matrix[i] = n * n - i; // 生成一个均匀分布的随机数
  }
  for (int it = 0; it < iters; ++it) {
    for (int i = 0; i < n * n; ++i) {
      matrix[i] += i;
    }
    auto start = std::chrono::high_resolution_clock::now();
    transpose();
    auto finish = std::chrono::high_resolution_clock::now();
    total += (finish - start).count();
  }
  std::cout << "\nIters: " << iters << ", Elapsed time: " << total / 1000000000.0 << " s\n";   // 输出耗时
  return 0;
}
