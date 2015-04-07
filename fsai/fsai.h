#ifndef FSAI_H
#define FSAI_H

#include "formats/formats.h"

void fsai(csr_matrix<float> &Ainv1,
    csr_matrix<float> &Ainv2,
    const csr_matrix<double> &A);

void fsai(csr_matrix<double> &Ainv1,
    csr_matrix<double> &Ainv2,
    const csr_matrix<double> &A);

#endif
