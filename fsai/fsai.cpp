#include "fsai/fsai.h"

#include <omp.h>
#include <cassert>

template<typename real>
csr_matrix<real> csr_transpose(const csr_matrix<real> &A) {
    /// Init matrix
    csr_matrix<real> B;
    B.n_rows = A.n_cols;
    B.n_cols = A.n_rows;
    B.row_ptr.clear();
    B.row_ptr.resize(B.n_rows + 1);
    B.cols.clear();
    B.elms.clear();

    /// Allocate temp arrays
    int *rcount = new int[B.n_rows];
    int *rindex = new int[B.n_rows];

    for (int i = 0; i < B.n_rows; ++i) {
        rcount[i] = 0;
    }

    /// Calculate row count
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            ++rcount[A.cols[j]];
        }
    }

    /// Calculare row_ptr
    rindex[0] = 0;
    for (int i = 1; i < B.n_rows; ++i) {
        rindex[i] = rindex[i - 1] + rcount[i - 1];
    }

    for (int i = 0; i < B.n_rows; ++i) {
        B.row_ptr[i] = rindex[i];
    }
    B.n_nz = B.row_ptr[B.n_rows] = rindex[B.n_rows - 1] + rcount[B.n_rows - 1];

    /// Fill cols & elms
    B.cols.resize(B.n_nz);
    B.elms.resize(B.n_nz);

    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            B.cols[rindex[A.cols[j]]] = i;
            B.elms[rindex[A.cols[j]]] = A.elms[j];
            ++rindex[A.cols[j]];
        }
    }

    /// Free temp arrays
    delete[] rcount;
    delete[] rindex;

    /// Return result
    return B;
}

template<typename in_real, typename out_real>
csr_matrix<out_real> generate_pattern(const csr_matrix<in_real> &A) {
    /// Init matrix
    csr_matrix<out_real> B;
    B.n_rows = A.n_rows;
    B.n_cols = A.n_cols;
    B.row_ptr.clear();
    B.row_ptr.resize(B.n_rows + 1);
    B.cols.clear();
    B.elms.clear();

    /// Allocate temp arrays
    int *rcount = new int[B.n_rows];
    int *rindex = new int[B.n_rows];

    for (int i = 0; i < B.n_rows; ++i) {
        rcount[i] = 1;
    }

    /// Calculate row count
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.cols[j] >= i) {
                break;
            }
            ++rcount[i];
        }
    }

    /// Calculare row_ptr
    rindex[0] = 0;
    for (int i = 1; i < B.n_rows; ++i) {
        rindex[i] = rindex[i - 1] + rcount[i - 1];
    }

    for (int i = 0; i < B.n_rows; ++i) {
        B.row_ptr[i] = rindex[i];
    }
    B.n_nz = B.row_ptr[B.n_rows] = rindex[B.n_rows - 1] + rcount[B.n_rows - 1];

    /// Allocate cols & elms
    B.cols.resize(B.n_nz);
    B.elms.resize(B.n_nz);

    /// Fill cols
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            B.cols[rindex[i]] = A.cols[j];
            ++rindex[i];
        }
        B.cols[rindex[i]] = i;
    }

    /// Free temp arrays
    delete[] rcount;
    delete[] rindex;

    /// Return result
    return B;
}

template<typename real>
real real_abs(real x) {
    return x > 0 ? x : -x;
}

template<typename T>
void swap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
}

template<>
void swap(int &a, int &b) {
    a ^= b;
    b ^= a;
    a ^= b;
}

template<typename in_real, typename out_real>
vector<out_real> solve(dense_matrix<in_real> &A) {
    /// Check system
    assert(A.n_rows + 1 == A.n_cols);

    /// Define system size N
    int N = A.n_rows;

    /// Init X
    vector<out_real> X;
    X.resize(N);

    /// Init swaps array
    int *swp = new int[N];
    for (int i = 0; i < N; ++i) {
        swp[i] = i;
    }

    /// Gauss iterations
    for (int i = 0; i < N; ++i) {
        /// Find max element
        in_real max = real_abs(A.data[i][i]);
        int max_i = i;
        for (int j = i + 1; j < N; ++j) {
            if (real_abs(A.data[i][j]) > max) {
                max = real_abs(A.data[i][j]);
                max_i = j;
            }
        }

        /// Swap cols
        if (i != max_i) {
            swap(swp[i], swp[max_i]);
            for (int j = 0; j < N; ++j) {
                swap(A.data[j][i], A.data[j][max_i]);
            }
        }

        /// Subtract rows
        in_real tmp = A.data[i][i];
        for (int j = i; j < N; ++j) {
            A.data[i][j] /= tmp;
        }
        A.data[i][N] /= tmp;
        for (int j = i + 1; j < N; ++j) {
            tmp = A.data[j][i];
            for (int k = i; k <= N; ++k) {
                A.data[j][k] -= tmp * A.data[i][k];
            }
        }
    }

    /// Calculate answer
    for (int i = N; i > 0; --i) {
        X[swp[i - 1]] = A.data[i - 1][N];
        for (int j = i; j < N; ++j) {
            X[swp[i - 1]] -= A.data[i - 1][j] * X[swp[j]];
        }
    }

    /// Free temp array
    delete[] swp;

    /// Return X
    return X;
}

template<typename in_real, typename out_real>
void calculate_factor(csr_matrix<out_real> &F, const csr_matrix<in_real> &A) {
    /// Calculare F's rows
    #pragma omp parallel for
    for (int row = 0; row < F.n_rows; ++row) {
        /// Init matrix for small system
        dense_matrix<in_real> S;
        S.n_rows = F.row_ptr[row + 1] - F.row_ptr[row];
        S.n_cols = S.n_rows + 1;
        S.data.resize(S.n_rows);
        for (int i = 0; i < S.n_rows; ++i) {
            S.data[i].resize(S.n_cols);
        }

        /// Generate small system
        for (int i = 0; i < S.n_rows; ++i) {
            int a_row = F.cols[F.row_ptr[row] + i];
            int f1 = F.row_ptr[row];
            int f1e = F.row_ptr[row + 1];
            int f2 = A.row_ptr[a_row];
            int f2e = A.row_ptr[a_row + 1];
            int j = 0;
            while ((f1 < f1e) && (f2 < f2e)) {
                if (F.cols[f1] == A.cols[f2]) {
                    S.data[i][j] = A.elms[f2];
                    ++f1;
                    ++f2;
                    ++j;
                } else {
                    if (F.cols[f1] < A.cols[f2]) {
                        S.data[i][j] = 0;
                        ++f1;
                        ++j;
                    } else {
                        ++f2;
                    }
                }
            }
            while (j < S.n_cols - 1) {
                S.data[i][j] = 0;
                ++j;
            }
            S.data[i][S.n_cols - 1] = ((i == S.n_rows - 1) ? 1 : 0);
        }

        /// Solve small system
        vector<out_real> X = solve<in_real, out_real>(S);
        for (int i = 0; i < X.size(); ++i) {
            F.elms[F.row_ptr[row] + i] = X[i];
        }
    }
}

template<typename real>
csr_matrix<real> out_transform(const csr_matrix<real> &A) {
    /// Init matrix
    csr_matrix<real> B;
    B.n_rows = A.n_rows;
    B.n_cols = A.n_cols;
    B.row_ptr.clear();
    B.row_ptr.resize(B.n_rows + 1);
    B.cols.clear();
    B.elms.clear();

    /// Allocate temp arrays
    int *rcount = new int[B.n_rows];
    int *rindex = new int[B.n_rows];

    /// Calculate row count
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            ++rcount[A.n_rows - 1 - i];
        }
    }

    /// Calculare row_ptr
    rindex[0] = 0;
    for (int i = 1; i < B.n_rows; ++i) {
        rindex[i] = rindex[i - 1] + rcount[i - 1];
    }

    for (int i = 0; i < B.n_rows; ++i) {
        B.row_ptr[i] = rindex[i];
    }
    B.n_nz = B.row_ptr[B.n_rows] = rindex[B.n_rows - 1] + rcount[B.n_rows - 1];

    /// Fill cols & elms
    B.cols.resize(B.n_nz);
    B.elms.resize(B.n_nz);

    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            B.cols[rindex[A.n_rows - 1 - i]] = A.cols[j];
            B.elms[rindex[A.n_rows - 1 - i]] = A.elms[j];
            ++rindex[A.n_rows - 1 - i];
        }
    }

    /// Free temp arrays
    delete[] rcount;
    delete[] rindex;

    /// Return result
    return B;
}

template<typename in_real, typename out_real>
void fsai_impl(csr_matrix<out_real> &Ainv1,
    csr_matrix<out_real> &Ainv2,
    const csr_matrix<in_real> &A) {

    /// Calculate transposed A
    csr_matrix<in_real> AT = csr_transpose(A);

    /// Generate pattern for Ainv1
    Ainv1 = generate_pattern<in_real, out_real>(A);

    /// Calculate Ainv1
    calculate_factor(Ainv1, AT);

    /// Transform Ainv1
    Ainv1 = out_transform(Ainv1);

    /// Calculate transposed Ainv2
    Ainv2 = csr_transpose(Ainv1);

}

void fsai(csr_matrix<float> &Ainv1,
    csr_matrix<float> &Ainv2,
    const csr_matrix<double> &A) {
    fsai_impl<double, float>(Ainv1, Ainv2, A);
}

void fsai(csr_matrix<double> &Ainv1,
    csr_matrix<double> &Ainv2,
    const csr_matrix<double> &A) {
    fsai_impl<double, double>(Ainv1, Ainv2, A);
}
