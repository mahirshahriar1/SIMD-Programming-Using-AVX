#include <iostream>
#include <immintrin.h>  // Header for AVX intrinsics

// Function to compute element-wise maximum using AVX-256
void avx256_max(const double* a, const double* b, double* result, int length) {
    for (int i = 0; i < length; i += 4) {
        // Load 4 elements from both arrays
        __m256d vecA = _mm256_loadu_pd(a + i);
        __m256d vecB = _mm256_loadu_pd(b + i);

        // Compute the element-wise maximum
        __m256d maxVec = _mm256_max_pd(vecA, vecB);

        // Store the result
        _mm256_storeu_pd(result + i, maxVec);
    }
}

int main() {
    // Two arrays of 8 doubles (AVX processes 4 doubles at a time)
    double a[8] = { 1.0, 5.0, 3.0, 9.0, 6.0, 2.0, 8.0, 7.0 };
    double b[8] = { 4.0, 3.0, 8.0, 1.0, 5.0, 6.0, 7.0, 10.0 };
    double result[8];

    // Perform element-wise maximum using AVX-256
    avx256_max(a, b, result, 8);

    // Output the result
    std::cout << "Element-wise maximum result: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << result[i] << " ";  // Should output 4.0 5.0 8.0 9.0 6.0 6.0 8.0 10.0
    }
    std::cout << std::endl;

    return 0;
}

/*
Explanation:
This program uses the intrinsic _mm256_max_pd to compute the maximum of corresponding elements in two vectors.
It processes 4 doubles at a time, which is much faster than a standard loop.
*/