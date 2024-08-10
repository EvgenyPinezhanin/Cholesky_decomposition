#include <cholesky_decomposition.h>

#include <cmath>

#include <omp.h>

inline void set_zero(double *A, int n) {
#pragma omp parallel for schedule(static) \
        firstprivate(n)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
}

inline void cholesky(double *A, double *L, int left_index, int right_index, int n) {
    for (int i = left_index; i < right_index; ++i) {
        double l = A[i * n + i];
        for (int j = left_index; j < i; ++j) {
            l -= L[i * n + j] * L[i * n + j];
        }
        L[i * n + i] = std::sqrt(l);

    #pragma omp parallel for schedule(dynamic) \
            private(l) firstprivate(left_index, right_index, i, n)
        for (int j = i + 1; j < right_index; ++j) {
            l = A[j * n + i];
            for (int k = left_index; k < i; ++k) {
                l -= L[i * n + k] * L[j * n + k];
            }
            L[j * n + i] = l / L[i * n + i];
        }
    }
};

inline void system_equations(double *A, double *L, int iter, int block_size, int loop_main, int n) {
#pragma omp parallel for schedule(static) \
        firstprivate(iter, block_size, n, loop_main)
    for (int iter_sq = iter + 1; iter_sq < loop_main + 1; ++iter_sq) {
        for (int i = iter_sq * block_size; i < std::min((iter_sq + 1) * block_size, n); ++i) {
            for (int j = iter * block_size; j < (iter + 1) * block_size; ++j) {
                double l = A[i * n + j];
                for (int k = iter * block_size; k < j; ++k) {
                    l -= L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = l / L[j * n + j];
            }
        }
    }
};

inline void matrix_mul(double *A, double *L, int iter, int block_size, int block_size_matrix, int coeff, int loop_matrix, int n) {
#pragma omp parallel for schedule(dynamic) \
        firstprivate(iter, coeff, block_size, block_size_matrix, n)
    for (int iter_mm_i = (iter + 1) * coeff; iter_mm_i < loop_matrix + 1; ++iter_mm_i) {
        for (int iter_mm_j = (iter + 1) * coeff; iter_mm_j <= iter_mm_i; ++iter_mm_j) {
            for (int i = iter_mm_i * block_size_matrix; i < std::min((iter_mm_i + 1) * block_size_matrix, n); ++i) {
                for (int j = iter_mm_j * block_size_matrix; j <= std::min((iter_mm_j + 1) * block_size_matrix - 1, i); ++j) {
                    double l = 0.0;
                    for (int k = iter * block_size; k < (iter + 1) * block_size; ++k) {
                        l += L[i * n + k] * L[j * n + k];
                    }
                    A[i * n + j] -= l;
                }
            }
        }
    }
};

void Cholesky_Decomposition(double * A, double * L, int n) {
    const int coeff = 2;
    const int block_size_main = 256;
    const int loop_main = n / block_size_main;
    const int block_size_matrix = block_size_main / coeff;
    const int loop_matrix = n / block_size_matrix;
    const int tail = n % block_size_main;

    set_zero(L, n);

    for (int iter = 0; iter < loop_main; ++iter) {
        cholesky(A, L, iter * block_size_main, (iter + 1) * block_size_main, n);
        system_equations(A, L, iter, block_size_main, loop_main, n);
        matrix_mul(A, L, iter, block_size_main, block_size_matrix, coeff, loop_matrix, n);
    }

    if (tail != 0) {
        cholesky(A, L, n - tail, n, n);
    }
}
