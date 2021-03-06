#ifndef MATRIX_FORMATS_H
#define MATRIX_FORMATS_H

#include "util/cuda/vector.h"

#include <iostream>
#include <iomanip>

struct matrix_shape
{
  int n_rows, n_cols, n_nz;
};

template<class T>
struct dense_matrix: matrix_shape
{
    vector<vector<T, host_memory_space_tag>, host_memory_space_tag> data;
};

template<class T>
struct coo_matrix: matrix_shape
{
  vector<int> rows, cols;
  vector<T> elms;
};

template<class T>
struct csr_matrix: matrix_shape
{
  vector<int, host_memory_space_tag> row_ptr, cols;
  vector<T, host_memory_space_tag> elms;
  void pad(int k)
    {
      k--;
      int n_padded_rows = (n_rows + k) & ~k;
      int last = row_ptr.back();
      for (; n_rows < n_padded_rows; n_rows++)
	row_ptr.push_back(last);
    }

  unsigned spmv_bytes() const
  {
    return (sizeof(unsigned) * (row_ptr.size() + cols.size())
	    + sizeof(T) * (elms.size() + 2 * n_rows));
  }
  void clear() {
    n_rows = n_cols = n_nz = 0;
    row_ptr.clear();
    cols.clear();
    elms.clear();
  }
};

template<class T>
vector<T> spvm(const csr_matrix<T> &A, const vector<T> &X) {
    vector<T> R(A.n_rows);
    #ifdef openmp
    #pragma omp parallel for
    #endif
    for (int i = 0; i < R.size(); ++i) {
        R[i] = 0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            R[i] += A.elms[j] * X[A.cols[j]];
        }
    }
    return R;
}

struct slell_params
{
  unsigned S, H, V, D;
};

template<typename E, typename memory_space>
struct slell_matrix: matrix_shape
{
  vector<unsigned, memory_space> slice_ptr;
  vector<unsigned, memory_space> cols;
  vector<E, memory_space> elms;
  unsigned slice_height;
  unsigned n_slices;
  unsigned hblock;
  bool var_height;
  bool diags;

  slell_matrix(const slell_matrix<E, host_memory_space_tag> &m)
   : matrix_shape(m),
     slice_ptr(m.slice_ptr),
     cols(m.cols),
     elms(m.elms),
     slice_height(m.slice_height),
     n_slices(m.n_slices),
     hblock(m.hblock),
     var_height(m.var_height),
     diags(m.diags)
  {}

  slell_matrix(const matrix_shape &m): matrix_shape(m) {}

  slell_matrix() {}

  unsigned spmv_bytes(bool effective = false, int vec_elt_size = sizeof(E))
  {
    return (sizeof(unsigned) * slice_ptr.size()
            + sizeof(unsigned) * (effective ? n_nz : cols.size())
            + sizeof(E)        * (effective ? n_nz : elms.size())
            + sizeof(vec_elt_size) * 2 * n_rows);
  }

  E *slice_elt_ptr(unsigned row, unsigned &slice_hint)
  {
    int slice_ents = slice_ptr.size() / (n_slices + 1);
    int slice;
    if (!var_height)
    {
      slice = row / slice_height;
      if (row % slice_height)
	return NULL;
    }
    else
    {
      for (slice = slice_hint;
	   slice <= n_slices && slice_ptr[slice * slice_ents + 1] != row;
	   slice++)
	{}
      if (slice > n_slices)
	return NULL;
      slice_hint = slice;
    }
    return &elms[slice_ptr[slice * slice_ents]];
  }
};

template<typename real>
void print_matrix(const csr_matrix<real> &A) {
    for (int i = 0; i < A.n_rows; ++i) {
        int j = 0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            while (j < A.cols[k]) {
                std::cout << std::setprecision(3) << std::fixed << std::setw(16) << 0.0 << " ";
                ++j;
            }
            if (j == A.cols[k]) {
                std::cout << std::setprecision(3) << std::fixed << std::setw(16) << A.elms[k] << " ";
                ++j;
            }
        }
        while (j < A.n_cols) {
            std::cout << std::setprecision(3) << std::fixed << std::setw(16) << 0.0 << " ";
            ++j;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename real>
void print_matrix(const dense_matrix<real> &A) {
    for (int i = 0; i < A.n_rows; ++i) {
        for (int j = 0; j < A.n_cols; ++j) {
            std::cout << std::setprecision(3) << std::fixed << std::setw(16) << A.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif
