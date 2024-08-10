#include <iostream>
#include <random>

#include <cholesky_decomposition.h>
#include <omp.h>

void print_matrix(double *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

void copy_matrix(double *A, double *A_copy, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_copy[i * n + j] = A[i * n + j];
        }
    }
}

int main() {
    int n = 1000;
    double *B = new double[n * n];
    double *A = new double[n * n];
    double *A_copy = new double[n * n];
    double *L = new double[n * n];

    std::mt19937_64 gen(30032001);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B[i * n + j] = ((double)(gen() - gen.min()) / (gen.max() - gen.min()) - 0.5) * 1000.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                A[i * n + j] += B[i * n + k] * B[j * n + k];
            }
        }
    }

    int num_threads[3] = { 1, 2, 4 };
    double start_time, end_time;
    double time[3];

    for (int i = 0; i < 3; ++i) {
        omp_set_num_threads(num_threads[i]);

        copy_matrix(A, A_copy, n);

        start_time = omp_get_wtime();
        Cholesky_Decomposition(A_copy, L, n);
        end_time = omp_get_wtime();
        time[i] = end_time - start_time;

        std::cout << "Number of threads: " << num_threads[i]
                  << ", Time: " << time[i]
                  << ", Boost: " << time[0] / time[i] << "\n";
    }

    delete [] B;
    delete [] A;
    delete [] A_copy;
    delete [] L;

    return 0;
}
