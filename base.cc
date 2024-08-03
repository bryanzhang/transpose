#include <random>
#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>

constexpr int n = 1024;
int matrix[n * n]; // 声明数组

void transpose() {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      int tmp = matrix[i * n + j];
      matrix[i * n + j] = matrix[j * n + i];
      matrix[j * n + i] = tmp;
    }
  }
}

int main() {
  const int iters = 1000;

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
