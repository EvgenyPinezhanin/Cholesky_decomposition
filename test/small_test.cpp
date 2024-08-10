#include <iostream>

#include <cholesky_decomposition.h>

void print_matrix(double *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    int n = 5;
    double *A = new double[n * n];
    double *L = new double[n * n];

    A[0]  = 9.0; A[1]  = 4.0; A[2]  = 5.0; A[3]  = 3.0; A[4]  = 4.0;
    A[5]  = 4.0; A[6]  = 8.0; A[7]  = 1.0; A[8]  = 4.0; A[9]  = 3.0;
    A[10] = 5.0; A[11] = 1.0; A[12] = 9.0; A[13] = 1.0; A[14] = 5.0;
    A[15] = 3.0; A[16] = 4.0; A[17] = 1.0; A[18] = 8.0; A[19] = 0.0;
    A[20] = 4.0; A[21] = 3.0; A[22] = 5.0; A[23] = 0.0; A[24] = 8.0;

    std::cout << "A = \n";
    print_matrix(A, n);
    std::cout << "\n";

    Cholesky_Decomposition(A, L, n);

    std::cout << "L = \n";
    print_matrix(L, n);
    std::cout << "\n";

    delete [] A;
    delete [] L;

    return 0;
}
