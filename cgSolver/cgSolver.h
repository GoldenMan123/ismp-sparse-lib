#ifndef cgSolver_H
#define cgSolver_H

#include "formats/formats.h"

void cgSolver(vector<double> &x, const csr_matrix<double> &A, const vector<double> &b, double tol);

#endif
