cimport cython
cimport numpy as np

import numpy as np
from numpy.lib.stride_tricks import as_strided


# Array with limit 2D
# While data isn't used it is what causes cython to create a function for every type in array_types_t and gives us a defined data type to use with the numpy array
@cython.boundscheck(False)
cdef np.ndarray[array_types_t, ndim=2] ddarray(const array_types_t *data):
    cdef array_types_t tmp # Yes, this line is necessary, otherwise we can't initialize the cython array
    return np.empty_like(<array_types_t[:1]>(&tmp), shape=(0, 0))

@cython.boundscheck(False)
cdef np.ndarray[array_types_t, ndim=2] ddarray_C(array_types_t *data, long rows, long cols, long row_stride, long col_stride):
    return as_strided(np.asarray(<array_types_t[:rows,:cols]>data, order="C"), strides=[row_stride, col_stride])

@cython.boundscheck(False)
cdef np.ndarray[array_types_t, ndim=2] ddarray_F(array_types_t *data, long rows, long cols, long row_stride, long col_stride):
    return as_strided(np.asarray(<array_types_t[:rows:1,:cols]>data, order="F"), strides=[row_stride, col_stride])

@cython.boundscheck(False)
cdef np.ndarray[array_types_t, ndim=2] ddarray_copy_C(const array_types_t *data, long rows, long cols, long row_stride, long col_stride):
    return np.copy(as_strided(np.asarray(<array_types_t[:rows,:cols]>data, order="C"), strides=[row_stride, col_stride]))

@cython.boundscheck(False)
cdef np.ndarray[array_types_t, ndim=2] ddarray_copy_F(const array_types_t *data, long rows, long cols, long row_stride, long col_stride):
    return np.copy(as_strided(np.asarray(<array_types_t[:rows:1,:cols]>data, order="F"), strides=[row_stride, col_stride]))
