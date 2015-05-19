#include "cgSolver/cgSolver.h"
#include "fsai/fsai.h"

#include <iostream>
#include <sys/time.h>

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

double get_time() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void cgSolver(vector<double> &x, const csr_matrix<double> &A, const vector<double> &b, double tol) {

    std::cerr << "System size: " << b.size() << std::endl;

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

    std::cerr << "great: " << great << std::endl;

    /// Calc b great
    double b_great = 0.0;
    for (int i = 0; i < b.size(); ++i) {
        b_great = max(b_great, abs(b[i]));
    }

    std::cerr << "b_great: " << b_great << std::endl;

    double normFactor = great;

    for (int i = 0; i < rA.size(); ++i) {
        rA[i] = b[i] - wA[i];
    }

    double res = sum_mag(rA) / normFactor;

    std::cerr << "Start residual: " << res << std::endl;

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

        /// Init FSAI preconditioner
        csr_matrix<double> Ainv1, Ainv2;
        double old_time = get_time();
        fsai(Ainv1, Ainv2, A);
        std::cerr << "Preconditioner construction time: " << std::setprecision(3) << std::fixed << 1000.0 * (get_time() - old_time) << "ms" << std::endl;
        double solve_start = old_time = get_time();

        do
        {
            ++nIters;
            rhoOld = rho;

            /// Execute preconditioning
            wA = spvm(Ainv2, spvm(Ainv1, rA));

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

            if (get_time() - old_time > 0.1) {
                std::cerr << "Iteration " << nIters << " residual: " << std::setprecision(3) << std::scientific << res << std::endl;
                old_time = get_time();
            }
        } while (res > tol);

        res = sum_mag(rA) / normFactor;
        std::cerr << "Iteration " << nIters << " residual: " << std::setprecision(3) << std::scientific << res << std::endl;

        std::cerr << "Solve time: " << std::setprecision(3) << std::fixed << 1000.0 * (get_time() - solve_start) << "ms" << std::endl;
    }

    wA = spvm(A, x);
    for (int i = 0; i < x.size(); ++i) {
        rA[i] = b[i] - wA[i];
    }

    std::cerr << "final residual: " << std::setprecision(3) << std::scientific << sum_mag(rA) / normFactor << std::endl;
    std::cerr << "nIters: " << nIters << std::endl;
}
