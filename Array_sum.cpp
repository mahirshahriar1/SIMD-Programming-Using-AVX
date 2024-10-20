#include <iostream>
#include <immintrin.h>  // Header for AVX intrinsics

// Function to compute the sum of an array using AVX-256 and loop unrolling
double avx256_sum_array(const double* arr, int length) {
    __m256d sumVec1 = _mm256_setzero_pd();  // Initialize sum vectors
    __m256d sumVec2 = _mm256_setzero_pd();

    // Loop unrolling: Process 8 elements in one iteration
    for (int i = 0; i < length; i += 8) {
        __m256d vec1 = _mm256_loadu_pd(&arr[i]);      // Load first 4 elements
        __m256d vec2 = _mm256_loadu_pd(&arr[i + 4]);  // Load next 4 elements

        sumVec1 = _mm256_add_pd(sumVec1, vec1);  // Accumulate in first sum vector
        sumVec2 = _mm256_add_pd(sumVec2, vec2);  // Accumulate in second sum vector
    }

    // Combine the two sum vectors
    __m256d totalSum = _mm256_add_pd(sumVec1, sumVec2);

    // Horizontal sum of totalSum vector
    double sumArr[4];
    _mm256_storeu_pd(sumArr, totalSum);
    double finalSum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];

    return finalSum;
}

int main() {
    // Example array of 16 doubles
    double arr[16] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };

    // Compute sum using AVX-256
    double result = avx256_sum_array(arr, 16);

    // Output the result
    std::cout << "Sum of array: " << result << std::endl;  // Should output 136

    return 0;
}

/*
Optimizations:
Loop Unrolling: By processing 8 elements per iteration, we reduce loop overhead.
Multiple Accumulators: We use two accumulators (sumVec1 and sumVec2) to further parallelize the computation.
Horizontal Reduction: After summing the vectors, we perform horizontal addition to get the final result.
*/