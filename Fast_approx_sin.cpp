#include <iostream>
#include <immintrin.h>  // Header for AVX intrinsics

// Fast approximation of sin(x) using AVX-256 and a simple Taylor expansion
void avx256_approx_sin(const double* x, double* result, int length) {
    // Constants for Taylor series approximation (up to x^5 term)
    const double c1 = -1.0 / 6.0;    // Factor for x^3
    const double c2 = 1.0 / 120.0;   // Factor for x^5

    for (int i = 0; i < length; i += 4) {
        __m256d vecX = _mm256_loadu_pd(&x[i]);  // Load 4 elements from input array

        // Compute x^2 and x^3
        __m256d vecX2 = _mm256_mul_pd(vecX, vecX);      // x^2
        __m256d vecX3 = _mm256_mul_pd(vecX2, vecX);     // x^3

        // Compute Taylor series approximation for sin(x): x - x^3/6 + x^5/120
        __m256d resultVec = _mm256_fmadd_pd(vecX3, _mm256_set1_pd(c1), vecX);  // x - x^3/6
        resultVec = _mm256_fmadd_pd(_mm256_mul_pd(vecX3, vecX2), _mm256_set1_pd(c2), resultVec);  // + x^5/120

        // Store the result
        _mm256_storeu_pd(&result[i], resultVec);
    }
}

int main() {
    // Example input array of angles (in radians)
    double x[8] = { 0.0, 0.5235987756, 1.0471975512, 1.5707963268,  // 0, 30, 60, 90 degrees
                    2.0943951024, 2.6179938780, 3.1415926536, 3.6651914292 };  // 120, 150, 180, 210 degrees
    double result[8];

    // Compute sin(x) using AVX-256
    avx256_approx_sin(x, result, 8);

    // Output the results
    std::cout << "Approximated sin values: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Optimizations:
Polynomial Approximation: Using a simple Taylor expansion for sin(x) improves performance over the standard sin function.
Fused Multiply-Add (FMA): The intrinsic _mm256_fmadd_pd computes (a * b) + c in one step, reducing both latency and instruction count.
*/