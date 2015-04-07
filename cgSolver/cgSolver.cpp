#include "cgSolver/cgSolver.h"

#include <iostream>

template<typename T>
T abs(const T &x) {
    return x > 0 ? x : -x;
}

template<typename T>
const T &max(const T &a, const T &b) {
    return a > b ? a : b;
}

double sum_mag(const vector<double> &v) {
    double r = 0;
    for (int i = 0; i < v.size(); ++i) {
        r += abs(v[i]);
    }
    return r;
}

double sum_prod(const vector<double> &a, const vector<double> &b) {
    double r = 0;
    for (int i = 0; i < a.size(); ++i) {
        r += a[i] * b[i];
    }
    return r;
}

void cgSolver(vector<double> &x, const csr_matrix<double> &A, const vector<double> &b, double tol) {
    vector<double> wA(b.size());
    vector<double> rA(b.size());

    /// Calculate initial residual
    wA = spvm(A, x);

    /// Calc matrix great
    double great = 0.0;
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            great = max(great, abs(A.elms[j]));
        }
    }

    /// Calc b great
    double b_great = 0.0;
    for (int i = 0; i < b.size(); ++i) {
        b_great = max(b_great, abs(b[i]));
    }

    double normFactor = great;

    for (int i = 0; i < rA.size(); ++i) {
        rA[i] = b[i] - wA[i];
    }

    double res = sum_mag(rA) / normFactor;

    std::cerr << "Residual: " << res << std::endl;

    int nIters = 0;

    /// cg iterations
    if (res > tol) {
        double rho = great;
        double rhoOld = rho;

        double alpha, beta, wApA;

        vector<double> pA(b.size());
        for (int i = 0; i < pA.size(); ++i) {
            pA[i] = 0;
        }

        do
        {
            ++nIters;
            rhoOld = rho;

            /// Execute preconditioning
            wA = rA;
            /** TODO **/

            /// Update search directions
            rho = sum_prod(wA, rA);
            beta = rho / rhoOld;
            for (int i = 0; i < pA.size(); ++i) {
                pA[i] = wA[i] + beta * pA[i];
            }

            /// Update preconditioned residual
            wA = spvm(A, pA);
            wApA = sum_prod(wA, pA);

            /// Update solution and residual
            alpha = rho / wApA;
            for (int i = 0; i < x.size(); ++i) {
                x[i] += alpha * pA[i];
            }
            for (int i = 0; i < rA.size(); ++i) {
                rA[i] -= alpha * wA[i];
            }
            res = sum_mag(rA) / normFactor;

            std::cerr << "Residual: " << res << std::endl;
        } while (res > tol);
    }

    std::cerr << "nIters: " << nIters << std::endl;
}
