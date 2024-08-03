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

      __m256i yvec0 = _mm256_load_si256((__m256i*)(matrix + (i << digits) + j));
      __m256i yvec1 = _mm256_load_si256((__m256i*)(matrix + ((i + 1) << digits) + j));
      __m256i yvec2 = _mm256_load_si256((__m256i*)(matrix + ((i + 2) << digits) + j));
      __m256i yvec3 = _mm256_load_si256((__m256i*)(matrix + ((i + 3) << digits) + j));
      __m256i yvec4 = _mm256_load_si256((__m256i*)(matrix + ((i + 4) << digits) + j));
      __m256i yvec5 = _mm256_load_si256((__m256i*)(matrix + ((i + 5) << digits) + j));
      __m256i yvec6 = _mm256_load_si256((__m256i*)(matrix + ((i + 6) << digits) + j));
      __m256i yvec7 = _mm256_load_si256((__m256i*)(matrix + ((i + 7) << digits) + j));

      int tmp1, tmp2;

      // xvec0
      tmp1  = _mm256_extract_epi32(xvec0, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 0);
      _mm256_insert_epi32(xvec0, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 0);
      _mm256_insert_epi32(xvec0, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 0);
      _mm256_insert_epi32(xvec0, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 0);
      _mm256_insert_epi32(xvec0, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 0);
      _mm256_insert_epi32(xvec0, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 0);
      _mm256_insert_epi32(xvec0, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 6);
      tmp2 = _mm256_extract_epi32(yvec6, 0);
      _mm256_insert_epi32(xvec0, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec0, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 0);
      _mm256_insert_epi32(xvec0, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 0);

      // xvec1
      tmp1  = _mm256_extract_epi32(xvec1, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 1);
      _mm256_insert_epi32(xvec1, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 1);
      _mm256_insert_epi32(xvec1, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 0);

      tmp1 = _mm256_extract_epi32(xvec1, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 1);
      _mm256_insert_epi32(xvec1, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 1);
      _mm256_insert_epi32(xvec1, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 1);
      _mm256_insert_epi32(xvec1, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 1);
      _mm256_insert_epi32(xvec1, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 1);
      _mm256_insert_epi32(xvec1, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 1);

      tmp1 = _mm256_extract_epi32(xvec1, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 1);
      _mm256_insert_epi32(xvec1, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 1);

      // xvec2
      tmp1  = _mm256_extract_epi32(xvec2, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 2);
      _mm256_insert_epi32(xvec2, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 2);
      _mm256_insert_epi32(xvec2, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 2);
      _mm256_insert_epi32(xvec2, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 2);
      _mm256_insert_epi32(xvec2, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 2);
      _mm256_insert_epi32(xvec2, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 2);
      _mm256_insert_epi32(xvec2, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 2);
      _mm256_insert_epi32(xvec2, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 2);

      tmp1 = _mm256_extract_epi32(xvec2, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 2);
      _mm256_insert_epi32(xvec2, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 2);

      // xvec3
      tmp1  = _mm256_extract_epi32(xvec3, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 3);
      _mm256_insert_epi32(xvec3, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 3);
      _mm256_insert_epi32(xvec3, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 3);
      _mm256_insert_epi32(xvec3, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 3);
      _mm256_insert_epi32(xvec3, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 3);
      _mm256_insert_epi32(xvec3, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 3);
      _mm256_insert_epi32(xvec3, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 3);
      _mm256_insert_epi32(xvec3, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 3);

      tmp1 = _mm256_extract_epi32(xvec3, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 3);
      _mm256_insert_epi32(xvec3, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 3);


      // xvec4
      tmp1  = _mm256_extract_epi32(xvec4, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 4);
      _mm256_insert_epi32(xvec4, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 4);
      _mm256_insert_epi32(xvec4, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 4);
      _mm256_insert_epi32(xvec4, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 4);
      _mm256_insert_epi32(xvec4, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 4);
      _mm256_insert_epi32(xvec4, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 4);
      _mm256_insert_epi32(xvec4, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 4);
      _mm256_insert_epi32(xvec4, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 4);

      tmp1 = _mm256_extract_epi32(xvec4, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 4);
      _mm256_insert_epi32(xvec4, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 4);


      // xvec6
      tmp1  = _mm256_extract_epi32(xvec6, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 5);
      _mm256_insert_epi32(xvec6, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 5);
      _mm256_insert_epi32(xvec6, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 5);
      _mm256_insert_epi32(xvec6, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 5);
      _mm256_insert_epi32(xvec6, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 5);
      _mm256_insert_epi32(xvec6, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 5);
      _mm256_insert_epi32(xvec6, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 5);
      _mm256_insert_epi32(xvec6, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 5);

      tmp1 = _mm256_extract_epi32(xvec6, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 5);
      _mm256_insert_epi32(xvec6, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 5);

      // xvec7
      tmp1  = _mm256_extract_epi32(xvec7, 0);
      tmp2 = _mm256_extract_epi32(yvec0, 7);
      _mm256_insert_epi32(xvec7, tmp2, 0);
      _mm256_insert_epi32(yvec0, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 1);
      tmp2 = _mm256_extract_epi32(yvec1, 7);
      _mm256_insert_epi32(xvec7, tmp2, 1);
      _mm256_insert_epi32(yvec1, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 2);
      tmp2 = _mm256_extract_epi32(yvec2, 7);
      _mm256_insert_epi32(xvec7, tmp2, 2);
      _mm256_insert_epi32(yvec2, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 3);
      tmp2 = _mm256_extract_epi32(yvec3, 7);
      _mm256_insert_epi32(xvec7, tmp2, 3);
      _mm256_insert_epi32(yvec3, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 4);
      tmp2 = _mm256_extract_epi32(yvec4, 7);
      _mm256_insert_epi32(xvec7, tmp2, 4);
      _mm256_insert_epi32(yvec4, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 5);
      tmp2 = _mm256_extract_epi32(yvec5, 7);
      _mm256_insert_epi32(xvec7, tmp2, 5);
      _mm256_insert_epi32(yvec5, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7,  6);
      tmp2 = _mm256_extract_epi32(yvec6, 7);
      _mm256_insert_epi32(xvec7, tmp2, 6);
      _mm256_insert_epi32(yvec6, tmp1, 7);

      tmp1 = _mm256_extract_epi32(xvec7, 7);
      tmp2 = _mm256_extract_epi32(yvec7, 7);
      _mm256_insert_epi32(xvec7, tmp2, 7);
      _mm256_insert_epi32(yvec7, tmp1, 7);

      // prefetch ((j + 8), i)
      _mm256_store_si256((__m256i*)(matrix + (j << digits) + i), xvec0);
      __builtin_prefetch(matrix + ((j + 8) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 1) << digits) + i), xvec1);
      __builtin_prefetch(matrix + ((j + 9) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 2) << digits) + i), xvec2);
      __builtin_prefetch(matrix + ((j + 10) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 3) << digits) + i), xvec3);
      __builtin_prefetch(matrix + ((j + 11) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 4) << digits) + i), xvec4);
      __builtin_prefetch(matrix + ((j + 12) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 5) << digits) + i), xvec5);;
      __builtin_prefetch(matrix + ((j + 13) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 6) << digits) + i), xvec6);
      __builtin_prefetch(matrix + ((j + 14) << digits) + i, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((j + 7) << digits) + i), xvec7);
      __builtin_prefetch(matrix + ((j + 15) << digits) + i, 0, 3);

      // prefetch (i, (j + 8))
      _mm256_store_si256((__m256i*)(matrix + (i << digits)+ j), yvec0);
      __builtin_prefetch(matrix + (i << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 1) << digits) + j), yvec1);
      __builtin_prefetch(matrix + ((i + 1) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 2) << digits) + j), yvec2);
      __builtin_prefetch(matrix + ((i + 2) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 3) << digits) + j), yvec3);
      __builtin_prefetch(matrix + ((i + 3) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 4) << digits) + j), yvec4);
      __builtin_prefetch(matrix + ((i + 4) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 5) << digits) + j), yvec5);
      __builtin_prefetch(matrix + ((i + 5) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 6) << digits) + j), yvec6);
      __builtin_prefetch(matrix + ((i + 6) << digits) + j + 8, 0, 3);
      _mm256_store_si256((__m256i*)(matrix + ((i + 7) << digits) + j), yvec7);
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

    int tmp1, tmp2;

    // vec0
    tmp1 = _mm256_extract_epi32(vec0, 1);
    tmp2 = _mm256_extract_epi32(vec1, 0);
    _mm256_insert_epi32(vec0, tmp2, 1);
    _mm256_insert_epi32(vec1, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 2);
    tmp2 = _mm256_extract_epi32(vec2, 0);
    _mm256_insert_epi32(vec0, tmp2, 2);
    _mm256_insert_epi32(vec2, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 3);
    tmp2 = _mm256_extract_epi32(vec3, 0);
    _mm256_insert_epi32(vec0, tmp2, 3);
    _mm256_insert_epi32(vec3, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 4);
    tmp2 = _mm256_extract_epi32(vec4, 0);
    _mm256_insert_epi32(vec0, tmp2, 4);
    _mm256_insert_epi32(vec4, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 5);
    tmp2 = _mm256_extract_epi32(vec5, 0);
    _mm256_insert_epi32(vec0, tmp2, 5);
    _mm256_insert_epi32(vec5, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 6);
    tmp2 = _mm256_extract_epi32(vec6, 0);
    _mm256_insert_epi32(vec0, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 0);

    tmp1 = _mm256_extract_epi32(vec0, 7);
    tmp2 = _mm256_extract_epi32(vec7, 0);
    _mm256_insert_epi32(vec0, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 0);

    // vec 1
    tmp1 = _mm256_extract_epi32(vec1, 2);
    tmp2 = _mm256_extract_epi32(vec2, 1);
    _mm256_insert_epi32(vec1, tmp2, 2);
    _mm256_insert_epi32(vec2, tmp1, 1);

    tmp1 = _mm256_extract_epi32(vec1, 3);
    tmp2 = _mm256_extract_epi32(vec3, 1);
    _mm256_insert_epi32(vec1, tmp2, 3);
    _mm256_insert_epi32(vec3, tmp1, 1);

    tmp1 = _mm256_extract_epi32(vec1, 4);
    tmp2 = _mm256_extract_epi32(vec4, 1);
    _mm256_insert_epi32(vec1, tmp2, 4);
    _mm256_insert_epi32(vec4, tmp1, 1);

    tmp1 = _mm256_extract_epi32(vec1, 5);
    tmp2 = _mm256_extract_epi32(vec4, 1);
    _mm256_insert_epi32(vec1, tmp2, 5);
    _mm256_insert_epi32(vec5, tmp1, 1);

    tmp1 = _mm256_extract_epi32(vec1, 6);
    tmp2 = _mm256_extract_epi32(vec6, 1);
    _mm256_insert_epi32(vec1, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 1);

    tmp1 = _mm256_extract_epi32(vec1, 7);
    tmp2 = _mm256_extract_epi32(vec7, 1);
    _mm256_insert_epi32(vec1, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 1);

    // vec2
    tmp1 = _mm256_extract_epi32(vec2, 3);
    tmp2 = _mm256_extract_epi32(vec3, 2);
    _mm256_insert_epi32(vec2, tmp2, 3);
    _mm256_insert_epi32(vec3, tmp1, 2);

    tmp1 = _mm256_extract_epi32(vec2, 4);
    tmp2 = _mm256_extract_epi32(vec4, 2);
    _mm256_insert_epi32(vec2, tmp2, 4);
    _mm256_insert_epi32(vec4, tmp1, 2);

    tmp1 = _mm256_extract_epi32(vec2, 5);
    tmp2 = _mm256_extract_epi32(vec4, 2);
    _mm256_insert_epi32(vec2, tmp2, 5);
    _mm256_insert_epi32(vec5, tmp1, 2);

    tmp1 = _mm256_extract_epi32(vec2, 6);
    tmp2 = _mm256_extract_epi32(vec6, 2);
    _mm256_insert_epi32(vec2, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 2);

    tmp1 = _mm256_extract_epi32(vec2, 7);
    tmp2 = _mm256_extract_epi32(vec7, 2);
    _mm256_insert_epi32(vec2, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 2);

    // vec3
    tmp1 = _mm256_extract_epi32(vec3, 4);
    tmp2 = _mm256_extract_epi32(vec4, 3);
    _mm256_insert_epi32(vec3, tmp2, 4);
    _mm256_insert_epi32(vec4, tmp1, 4);

    tmp1 = _mm256_extract_epi32(vec3, 5);
    tmp2 = _mm256_extract_epi32(vec5, 3);
    _mm256_insert_epi32(vec3, tmp2, 5);
    _mm256_insert_epi32(vec5, tmp1, 3);

    tmp1 = _mm256_extract_epi32(vec3, 6);
    tmp2 = _mm256_extract_epi32(vec6, 3);
    _mm256_insert_epi32(vec3, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 3);

    tmp1 = _mm256_extract_epi32(vec3, 7);
    tmp2 = _mm256_extract_epi32(vec7, 3);
    _mm256_insert_epi32(vec3, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 3);;

    // vec4
    tmp1 = _mm256_extract_epi32(vec4, 5);
    tmp2 = _mm256_extract_epi32(vec5, 4);
    _mm256_insert_epi32(vec4, tmp2, 5);
    _mm256_insert_epi32(vec5, tmp1, 4);

    tmp1 = _mm256_extract_epi32(vec4, 6);
    tmp2 = _mm256_extract_epi32(vec6, 4);
    _mm256_insert_epi32(vec4, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 4);

    tmp1 = _mm256_extract_epi32(vec4, 7);
    tmp2 = _mm256_extract_epi32(vec7, 4);
    _mm256_insert_epi32(vec4, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 4);

    // vec5
    tmp1 = _mm256_extract_epi32(vec5, 6);
    tmp2 = _mm256_extract_epi32(vec6, 5);
    _mm256_insert_epi32(vec5, tmp2, 6);
    _mm256_insert_epi32(vec6, tmp1, 5);

    tmp1 = _mm256_extract_epi32(vec5, 7);
    tmp2 = _mm256_extract_epi32(vec7, 5);
    _mm256_insert_epi32(vec5, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 5);

   // vec6
    tmp1 = _mm256_extract_epi32(vec6, 7);
    tmp2 = _mm256_extract_epi32(vec7, 6);
    _mm256_insert_epi32(vec6, tmp2, 7);
    _mm256_insert_epi32(vec7, tmp1, 6);

    // 向量化回写
    _mm256_store_si256((__m256i*)(matrix + (i << digits) + i), vec0);
    _mm256_store_si256((__m256i*)(matrix + ((i + 1) << digits) + i), vec1);
    _mm256_store_si256((__m256i*)(matrix + ((i + 2) << digits) + i), vec2);
    _mm256_store_si256((__m256i*)(matrix + ((i + 3) << digits) + i), vec3);
    _mm256_store_si256((__m256i*)(matrix + ((i + 4) << digits) + i), vec4);
    _mm256_store_si256((__m256i*)(matrix + ((i + 5) << digits) + i), vec5);;
    _mm256_store_si256((__m256i*)(matrix + ((i + 6) << digits) + i), vec6);
    _mm256_store_si256((__m256i*)(matrix + ((i + 7) << digits) + i), vec7);
  }
}

int main() {
  const int iters = 1000;

  void* p = (void*)matrix;
  std::cout << "Is aligned to cache line: " << (((size_t)p & 63L) == 0) << ", p=" << p << std::endl;
  double total = 0.0;
  for(int i = 0; i < n * n; ++i) {
    matrix[i] = n * n - i; // 生成一个均匀分布的随机数
  }
  for (int it = 0; it < iters; ++it) {
    // 生成并填充数组
    for(int i = 0; i < n * n; ++i) {
      matrix[i] += i; // 生成一个均匀分布的随机数
    }
    auto start = std::chrono::high_resolution_clock::now();
    transpose();
    auto finish = std::chrono::high_resolution_clock::now();
    total += (finish - start).count();
  }
  std::cout << "\nIters: " << iters << ", Elapsed time: " << total / 1000000000.0 << " s\n";   // 输出耗时
  return 0;
}
