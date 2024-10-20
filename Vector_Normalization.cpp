#include <iostream>
#include <immintrin.h>  // Header for AVX intrinsics
#include <cmath>        // For sqrt

// Function to normalize a vector using AVX-256
void avx256_normalize(double* vec, int length) {
    __m256d sumVec = _mm256_setzero_pd();  // Initialize sum vector to zero

    // Step 1: Compute the sum of squares of the vector elements
    for (int i = 0; i < length; i += 4) {
        __m256d vecPart = _mm256_loadu_pd(&vec[i]);
        __m256d sqVec = _mm256_mul_pd(vecPart, vecPart);
        sumVec = _mm256_add_pd(sumVec, sqVec);  // Accumulate squares
    }

    // Horizontal addition to sum up all elements in sumVec
    double sumArr[4];
    _mm256_storeu_pd(sumArr, sumVec);
    double sumSquares = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];

    // Step 2: Compute the magnitude of the vector (sqrt of sum of squares)
    double magnitude = sqrt(sumSquares);

    // Step 3: Normalize the vector by dividing each element by the magnitude
    __m256d magVec = _mm256_set1_pd(magnitude);  // Set all elements to magnitude
    for (int i = 0; i < length; i += 4) {
        __m256d vecPart = _mm256_loadu_pd(&vec[i]);
        __m256d normVec = _mm256_div_pd(vecPart, magVec);  // Divide by magnitude
        _mm256_storeu_pd(&vec[i], normVec);  // Store normalized values
    }
}

int main() {
    // Define a vector of 8 doubles
    double vec[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    // Normalize the vector using AVX-256
    avx256_normalize(vec, 8);

    // Output the normalized vector
    std::cout << "Normalized vector: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Explanation:
Sum of Squares: We use AVX to calculate the sum of squares of the vector elements (_mm256_mul_pd), processing 4 elements at a time.
Horizontal Sum: After the sum of squares is computed, we sum the values in the AVX register manually.
Division by Magnitude: Finally, we divide each vector element by the magnitude to normalize the vector using _mm256_div_pd.
*/