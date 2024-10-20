#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cassert>

void avx_safe_matrix_mult(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int N = A.size(); // Assuming square matrices
    assert(A.size() == B.size() && B.size() == C.size()); // Ensure matrices are compatible

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {
            __m256i c_vec = _mm256_setzero_si256(); // Initialize result vector
            for (int k = 0; k < N; k++) {
                __m256i a_vec = _mm256_set1_epi32(A[i][k]); // Broadcast A[i][k]

                // Load 8 elements from B[k][j] if within bounds
                if (j + 7 < N) {
                    __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k][j])); // Load 8 elements
                    c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec, b_vec)); // Multiply and accumulate
                }
                else {
                    // Handle edge case for the remaining elements (j to N)
                    alignas(32) int tempB[8] = { 0 }; // Temporary buffer for loading remaining elements safely
                    for (int rem = j; rem < N; ++rem) {
                        tempB[rem - j] = B[k][rem];
                    }
                    __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tempB));
                    c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec, b_vec));
                }
            }
            if (j + 7 < N) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[i][j]), c_vec); // Store result back
            }
            else {
                // Store remaining elements manually (for the edge case)
                alignas(32) int tempC[8];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tempC), c_vec);
                for (int rem = j; rem < N; ++rem) {
                    C[i][rem] = tempC[rem - j];
                }
            }
        }
    }
}

int main() {
    const int N = 5; // Size of the matrix (not a multiple of 8)
    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4, 5},
        {5, 6, 7, 8, 9},
        {9, 10, 11, 12, 13},
        {13, 14, 15, 16, 17},
        {18, 19, 20, 21, 22}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0, 1, 1},
        {0, 1, 1, 0, 0},
        {1, 1, 0, 0, 1},
        {0, 0, 1, 1, 0},
        {1, 0, 1, 1, 0}
    };
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    avx_safe_matrix_mult(A, B, C);

    // Print result
    for (const auto& row : C) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

/*
Performance Gains: AVX processes 8 elements at a time, significantly speeding up matrix multiplication 
by leveraging SIMD (Single Instruction, Multiple Data) parallelism.

Handling Non-Optimal Sizes: It safely handles matrices of any size by checking bounds and using scalar 
fallback for remaining elements when dimensions aren't multiples of 8. This avoids out-of-bounds memory access errors.

Efficient CPU Utilization: By vectorizing most operations, the solution makes better use of the 
CPU's SIMD capabilities, reducing the total number of operations and improving cache utilization.

Scalability: It works for various matrix sizes and can be extended to AVX-512 for even more performance on newer CPUs.
*/