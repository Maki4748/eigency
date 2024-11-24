cimport numpy as np


ctypedef fused array_types_t:
    double
    long double
    double complex
    long double complex
    float
    float complex
    long
    unsigned long
    long long
    unsigned long long
    int
    unsigned int
    short
    unsigned short
    signed char
    unsigned char


# Array with limit 2D
cdef api np.ndarray[array_types_t, ndim=2] ddarray(const array_types_t *data)
cdef api np.ndarray[array_types_t, ndim=2] ddarray_C(array_types_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_types_t, ndim=2] ddarray_F(array_types_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_types_t, ndim=2] ddarray_copy_C(const array_types_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_types_t, ndim=2] ddarray_copy_F(const array_types_t *data, long rows, long cols, long outer_stride, long inner_stride)
