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

void set_zero(double *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
}

int main() {
    int n = 11;
    double A[n * n] = { 
        334,	158,	205,	177,	228,	213,	247,	251,	177,	213,	211,
        158,	214,	126,	166,	157,	162,	207,	132,	148,	137,	204,
        205,	126,	258,	186,	175,	239,	179,	177,	133,	148,	165,
        177,	166,	186,	279,	111,	229,	233,	254,	215,	197,	209,
        228,	157,	175,	111,	258,	177,	179,	148,	 79,	169,	159,
        213,	162,	239,	229,	177,	285,	231,	229,	169,	164,	203,
        247,	207,	179,	233,	179,	231,	339,	248,	216,	216,	226,
        251,	132,	177,	254,	148,	229,	248,	348,	228,	240,	241,
        177,	148,	133,	215,	 79,	169,	216,	228,	223,	183,	224,
        213,	137,	148,	197,	169,	164,	216,	240,	183,	271,	179,
        211,	204,	165,	209,	159,	203,	226,	241,	224,	179,	313,
    };
    double A_copy[n * n];
    double L[n * n];

    int num_threads[3] = { 1, 2, 4 };
    double start_time, end_time;
    double time[3];

    set_zero(L, n);

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

        std::cout << "L = \n";
        print_matrix(L, n);
        std::cout << "\n";
    }

    return 0;
}
