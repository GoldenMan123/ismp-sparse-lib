#include "cgSolver/cgSolver.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>

using std::cerr;
using std::cout;
using std::cin;
using std::endl;

csr_matrix<double> read_matrix() {
    csr_matrix<double> B;
    cin >> B.n_rows >> B.n_cols >> B.n_nz;
    B.row_ptr.resize(B.n_rows + 1);
    B.cols.resize(B.n_nz);
    B.elms.resize(B.n_nz);
    int row, col;
    double val;
    B.row_ptr[0] = 0;
    vector<int> rcount(B.n_rows);
    for (int i = 0; i < rcount.size(); ++i) {
        rcount[i] = 0;
    }
    for (int i = 0; i < B.n_nz; ++i) {
        cin >> col >> row >> val;
        ++rcount[row];
        B.cols[i] = col;
        B.elms[i] = val;
    }
    for (int i = 0; i < B.n_rows; ++i) {
        B.row_ptr[i + 1] = B.row_ptr[i] + rcount[i];
    }
    return B;
}

vector<double> random_vector(int size) {
    vector<double> r(size);
    for (int i = 0; i < size; ++i) {
        r[i] = rand() / (RAND_MAX + 1.0);
    }
    return r;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <matrix>" << endl;
        return 0;
    }

    srand(time(NULL));

    if (!freopen(argv[1], "r", stdin)) {
        cerr << "IO error" << endl;
        return -1;
    }

    csr_matrix<double> A = read_matrix();
    vector<double> b = random_vector(A.n_rows);
    vector<double> x(A.n_cols);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = 0;
    }

    cgSolver(x, A, b, 1e-6);

    return 0;
}
