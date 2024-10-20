#include <iostream>
#include <immintrin.h>  // Header for AVX intrinsics

// Function to compute the dot product using AVX-256
double avx256_dot_product(const double* a, const double* b, int length) {
    // Accumulate the result in an AVX register
    __m256d result = _mm256_setzero_pd();  // Initialize result vector to zero

    // Loop through the arrays in chunks of 4 doubles (256 bits)
    for (int i = 0; i < length; i += 4) {
        // Load 4 elements from each array
        __m256d vecA = _mm256_loadu_pd(a + i);  // Load 4 elements from 'a'
        __m256d vecB = _mm256_loadu_pd(b + i);  // Load 4 elements from 'b'

        // Multiply the vectors element-wise
        __m256d mulResult = _mm256_mul_pd(vecA, vecB);

        // Accumulate the result
        result = _mm256_add_pd(result, mulResult);
    }

    // Horizontal addition: sum all elements in the AVX register
    // AVX-256 does not support horizontal sum directly, so we manually sum:
    double sum[4];
    _mm256_storeu_pd(sum, result);  // Store the result back into an array

    // Return the total dot product sum
    return sum[0] + sum[1] + sum[2] + sum[3];
}

int main() {
    // Two arrays of 8 doubles (length should be a multiple of 4 for AVX-256)
    double a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double b[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    // Compute dot product using AVX-256
    double dotProduct = avx256_dot_product(a, b, 8);

    // Output the result
    std::cout << "Dot Product: " << dotProduct << std::endl;  // Should output 204

    return 0;
}

/* 

Benefits of Using AVX-256:
Parallelism:

Without AVX, you would compute each multiplication and addition one at a time in a loop. 
With AVX-256, you compute 4 multiplications at once, significantly speeding up the operation for large vectors.

Reduced Loop Iterations:

Normally, for an array of 8 elements, you'd need 8 iterations of the loop. 
With AVX-256, you're reducing the loop count by a factor of 4 (only 2 iterations here), reducing overhead and improving performance.

Improved Performance for Large Arrays:
The larger the array, the more performance benefit you'll gain. 
Modern CPUs can handle these SIMD instructions very efficiently, 
and you'll notice a huge speedup when dealing with vectors of length in the thousands or millions.

Memory Bandwidth Efficiency:
AVX loads and operates on 256 bits of data at a time, 
making better use of the memory bandwidth when loading from memory compared to loading individual elements.

Performance Consideration:
For large datasets, AVX-256 can provide 3-4x performance improvements over scalar code. 
It's particularly beneficial for high-performance computing tasks like machine learning, signal processing, 
scientific simulations, and graphics rendering where vector operations are frequent.

Horizontal Operations in AVX:

One challenge with AVX-256 is that it lacks efficient horizontal operations
(e.g., summing all elements in a vector). This is usually worked around by storing the data in memory and summing manually,
as seen in the example. AVX-512 (if supported) has more direct support for such operations.

*/