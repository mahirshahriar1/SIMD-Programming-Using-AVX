
/*
Loop unrolling improves performance by:

Reducing Loop Overhead: 
It decreases the number of loop control instructions (like incrementing counters and checking conditions), making the code more efficient.

Increasing Instruction-Level Parallelism: 
By unrolling, more independent operations are exposed, allowing the CPU to execute more instructions in parallel.

Better Resource Utilization: 
It reduces the number of branches and increases the workload per iteration, which can lead to better cache usage and fewer branch mispredictions.

Improving SIMD Usage: 
Unrolling helps in fully utilizing AVX registers, allowing the processor to perform more vector operations per iteration.

*/
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cassert>

void avx_unrolled_matrix_mult(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int N = A.size(); // Assuming square matrices
    assert(A.size() == B.size() && B.size() == C.size()); // Ensure matrices are compatible

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {
            __m256i c_vec = _mm256_setzero_si256(); // Initialize result vector
            for (int k = 0; k < N; k += 4) { // Unroll by 4
                __m256i a_vec0 = _mm256_set1_epi32(A[i][k]);
                __m256i a_vec1 = _mm256_set1_epi32(k + 1 < N ? A[i][k + 1] : 0);
                __m256i a_vec2 = _mm256_set1_epi32(k + 2 < N ? A[i][k + 2] : 0);
                __m256i a_vec3 = _mm256_set1_epi32(k + 3 < N ? A[i][k + 3] : 0);

                __m256i b_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k][j]));
                __m256i b_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 1 < N ? &B[k + 1][j] : &B[k][j])); // Handle edge cases
                __m256i b_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 2 < N ? &B[k + 2][j] : &B[k][j]));
                __m256i b_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 3 < N ? &B[k + 3][j] : &B[k][j]));

                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec0, b_vec0));
                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec1, b_vec1));
                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec2, b_vec2));
                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec3, b_vec3));
            }
            
            // Handle case where j + 7 is out of bounds
            if (j + 7 < N) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[i][j]), c_vec);
            } else {
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

    avx_unrolled_matrix_mult(A, B, C);

    // Print result
    for (const auto& row : C) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
