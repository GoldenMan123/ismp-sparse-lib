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
    vector<double> wA(x.size());
    vector<double> rA(x.size());

    /// Calculate initial residual
    wA = spvm(A, x);

    double normFactor = 1; /** TODO **/

    for (int i = 0; i < rA.size(); ++i) {
        rA[i] = b[i] - wA[i];
    }

    double res = sum_mag(rA) / normFactor;

    /// Calc matrix great
    double great = 0.0;
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            great = max(great, abs(A.elms[j]));
        }
    }

    std::cerr << "Residual: " << res << std::endl;

    /// cg iterations
    if (res > tol) {
        double rho = great;
        double rhoOld = rho;

        double alpha, beta, wApA;

        vector<double> pA(x.size());
        for (int i = 0; i < pA.size(); ++i) {
            pA[i] = 0;
        }

        do
        {
            rhoOld = rho;

            /// Execute preconditioning
            //preconPtr_->precondition(wA, rA, cmpt);
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
}
