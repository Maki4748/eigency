# distutils: language = c++
# distutils: sources = eigency_tests/eigency_tests_cpp.cpp
import numpy as np

from libc.stdint cimport (int8_t, int16_t, int64_t, uint8_t, uint16_t,
                          uint32_t, uint64_t)

from eigency.core cimport *

# cimport eigency.conversions
# from eigency_tests.eigency cimport *


# import eigency
# include "../eigency.pyx"

cdef extern from "eigency_tests/eigency_tests_cpp.h":

     cdef long _function_w_vec_arg "function_w_vec_arg"(Map[VectorXd] &)

     cdef long _function_w_1darr_arg "function_w_1darr_arg"(Map[ArrayXi] &)

     cdef void _function_w_vec_arg_no_map1 "function_w_vec_arg_no_map1"(Map[VectorXd])

     cdef void _function_w_vec_arg_no_map2 "function_w_vec_arg_no_map2"(Map[VectorXd] &)

     cdef void _function_w_mat_arg "function_w_mat_arg"(Map[MatrixXd] &)

     cdef void _function_w_ld_mat_arg "function_w_ld_mat_arg"(Map[MatrixXld] &)

     cdef void _function_w_complex_mat_arg "function_w_complex_mat_arg"(Map[MatrixXcd] &)

     cdef void _function_w_complex_ld_mat_arg "function_w_complex_ld_mat_arg"(Map[MatrixXcld] &)

     cdef void _function_w_fullspec_arg "function_w_fullspec_arg" (FlattenedMap[Array, double, Dynamic, _1] &)

     cdef VectorXd _function_w_vec_retval "function_w_vec_retval" ()

     cdef Matrix3d _function_w_mat_retval "function_w_mat_retval" ()

     cdef MatrixXd _function_w_empty_mat_retval "function_w_empty_mat_retval" ()

     cdef PlainObjectBase _function_w_mat_retval_full_spec "function_w_mat_retval_full_spec" ()

     cdef Map[ArrayXXd] &_function_filter1 "function_filter1" (Map[ArrayXXd] &)

     cdef PlainObjectBase _function_filter2 "function_filter2" (FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor] &)

     cdef PlainObjectBase _function_filter3 "function_filter3" (FlattenedMapWithStride[Array, double, Dynamic, Dynamic, ColMajor, Unaligned, _1, Dynamic] &)

     cdef PlainObjectBase _function_type_float "function_type_float" (Map[ArrayXXf] &)
     cdef PlainObjectBase _function_type_double "function_type_double" (Map[ArrayXXd] &)
     cdef PlainObjectBase _function_type_int8 "function_type_int8" (FlattenedMap[Array, int8_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_uint8 "function_type_uint8" (FlattenedMap[Array, uint8_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_int16 "function_type_int16" (FlattenedMap[Array, int16_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_uint16 "function_type_uint16" (FlattenedMap[Array, uint16_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_int32 "function_type_int32" (Map[ArrayXXi] &)
     cdef PlainObjectBase _function_type_uint32 "function_type_uint32" (FlattenedMap[Array, uint32_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_int64 "function_type_int64" (FlattenedMap[Array, int64_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_uint64 "function_type_uint64" (FlattenedMap[Array, uint64_t, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_complex_float "function_type_complex_float"(Map[ArrayXXcf] &)
     cdef PlainObjectBase _function_type_complex_double "function_type_complex_double"(Map[ArrayXXcd] &)

     cdef PlainObjectBase _function_type_longlong "function_type_longlong"(FlattenedMap[Array, long long, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_ulonglong "function_type_ulonglong"(FlattenedMap[Array, unsigned long long, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_clongdouble "function_type_clongdouble"(FlattenedMap[Array, long double complex, Dynamic, Dynamic] &)

     cdef PlainObjectBase _function_single_col_matrix "function_single_col_matrix" (Map[ArrayXXd] &)

     cdef cppclass _FixedMatrixClass "FixedMatrixClass":
         _FixedMatrixClass () except +
         Matrix3d &get_matrix()
         const Matrix3d &get_const_matrix()

     cdef cppclass _DynamicArrayClass "DynamicArrayClass":
         _DynamicArrayClass (Map[ArrayXXd] &) except +
         ArrayXXd &get_array()
         ArrayXXd get_array_copy()

     cdef cppclass _DynamicRowMajorArrayClass "DynamicRowMajorArrayClass":
         _DynamicRowMajorArrayClass (FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor] &) except +
         PlainObjectBase &get_array()
         PlainObjectBase get_array_copy()

# Function with vector argument.
def function_w_vec_arg(np.ndarray[np.float64_t] array):
    return _function_w_vec_arg(Map[VectorXd](array))

# Function with vector argument.
def function_w_1darr_arg(np.ndarray[np.int32_t] array):
    return _function_w_1darr_arg(Map[ArrayXi](array))

# Function with vector argument - no map - no ref.
def function_w_vec_arg_no_map1(np.ndarray[np.float64_t] array):
    return _function_w_vec_arg_no_map1(Map[VectorXd](array))

# Function with vector argument - no map.
def function_w_vec_arg_no_map2(np.ndarray[np.float64_t] array):
    return _function_w_vec_arg_no_map2(Map[VectorXd](array))

# Function with matrix argument.
def function_w_mat_arg(np.ndarray[np.float64_t, ndim=2] array):
    return _function_w_mat_arg(Map[MatrixXd](array))

# Function with long double matrix argument.
def function_w_ld_mat_arg(np.ndarray[np.longdouble_t, ndim=2] array):
    return _function_w_ld_mat_arg(Map[MatrixXld](array))

# Function with complex matrix argument.
def function_w_complex_mat_arg(np.ndarray[np.complex128_t, ndim=2] array):
    return _function_w_complex_mat_arg(Map[MatrixXcd](array))

# Function with complex long double matrix argument.
def function_w_complex_ld_mat_arg(np.ndarray[np.clongdouble_t, ndim=2] array):
    return _function_w_complex_ld_mat_arg(Map[MatrixXcld](array))

# Function using a full Map specification, rather than the convenience typedefs
# Note that since cython does not support nested fused types, the Map has been
# flattened to include all arguments at once
def function_w_fullspec_arg(np.ndarray[np.float64_t] array):
    return _function_w_fullspec_arg(FlattenedMap[Array, double, Dynamic, _1](array))

# Function returning vector (copy is made)
def function_w_vec_retval():
    return ndarray(_function_w_vec_retval())

# Function returning matrix (copy is made)
def function_w_mat_retval():
    return ndarray(_function_w_mat_retval())

# Function returning empty matrix (copy is made)
def function_w_empty_mat_retval():
    return ndarray(_function_w_empty_mat_retval())

# Function returning matrix (copy is made)
def function_w_mat_retval_full_spec():
    return ndarray(_function_w_mat_retval_full_spec())

# Function both taking array as argument and returning it
def function_filter1(np.ndarray array):
    return ndarray(_function_filter1(Map[ArrayXXd](array)))

# Function both taking array as argument and returning it - RowMajor order
def function_filter2(np.ndarray[np.float64_t, ndim=2] array):
    return ndarray(_function_filter2(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor](array)))

# Function both taking array as argument and returning it - RowMajor stride
def function_filter3(np.ndarray[np.float64_t, ndim=2] array):
    return ndarray(_function_filter3(FlattenedMapWithStride[Array, double, Dynamic, Dynamic, ColMajor, Unaligned, _1, Dynamic](array)))

# Functions with different matrix types: float32
def function_type_float32(np.ndarray[np.float32_t, ndim=2] array):
    return ndarray(_function_type_float(Map[ArrayXXf](array)))

# Functions with different matrix types: float64
def function_type_float64(np.ndarray[np.float64_t, ndim=2] array):
    return ndarray(_function_type_double(Map[ArrayXXd](array)))

# Functions with different matrix types: signed char
def function_type_int8(np.ndarray[np.int8_t, ndim=2] array):
    return ndarray(_function_type_int8(FlattenedMap[Array, int8_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: unsigned char
def function_type_uint8(np.ndarray[np.uint8_t, ndim=2] array):
    return ndarray(_function_type_uint8(FlattenedMap[Array, uint8_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: short/int
def function_type_int16(np.ndarray[np.int16_t, ndim=2] array):
    return ndarray(_function_type_int16(FlattenedMap[Array, int16_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: unsigned short/unsigned int
def function_type_uint16(np.ndarray[np.uint16_t, ndim=2] array):
    return ndarray(_function_type_uint16(FlattenedMap[Array, uint16_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: long
def function_type_int32(np.ndarray[np.int32_t, ndim=2] array):
    return ndarray(_function_type_int32(Map[ArrayXXi](array)))

# Functions with different matrix types: unsigned long
def function_type_uint32(np.ndarray[np.uint32_t, ndim=2] array):
    return ndarray(_function_type_uint32(FlattenedMap[Array, uint32_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: long long
def function_type_int64(np.ndarray[np.int64_t, ndim=2] array):
    return ndarray(_function_type_int64(FlattenedMap[Array, int64_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: unsigned long long
def function_type_uint64(np.ndarray[np.uint64_t, ndim=2] array):
    return ndarray(_function_type_uint64(FlattenedMap[Array, uint64_t, Dynamic, Dynamic](array)))

# Functions with different matrix types: complex64
def function_type_complex64(np.ndarray[np.complex64_t, ndim=2] array):
    return ndarray(_function_type_complex_float(Map[ArrayXXcf](array)))

# Functions with different matrix types: complex128
def function_type_complex128(np.ndarray[np.complex128_t, ndim=2] array):
    return ndarray(_function_type_complex_double(Map[ArrayXXcd](array)))

# Functions with different matrix types: long long
def function_type_longlong(np.ndarray[np.longlong_t, ndim=2] array):
    return ndarray(_function_type_longlong(FlattenedMap[Array, long_long, Dynamic, Dynamic](array)))

# Functions with different matrix types: unsigned long long
def function_type_ulonglong(np.ndarray[np.ulonglong_t, ndim=2] array):
    return ndarray(_function_type_ulonglong(FlattenedMap[Array, u_long_long, Dynamic, Dynamic](array)))

# Functions with different matrix types: long double complex
def function_type_clongdouble(np.ndarray[np.clongdouble_t, ndim=2] array):
    return ndarray(_function_type_clongdouble(FlattenedMap[Array, c_long_double, Dynamic, Dynamic](array)))

# Functions testing a matrix with only one column
def function_single_col_matrix(np.ndarray[np.float64_t, ndim=2] array):
    return ndarray(_function_single_col_matrix(Map[ArrayXXd](array)))

# Functions testing that map properly holds a reference to python objects.
def function_map_holds_reference(np.ndarray[np.float64_t, ndim=2] array):
    # Hold a reference to a copy of an array.
    cdef Map[ArrayXXd] eigency_map = Map[ArrayXXd](array.copy(order="K"))

    # Do some nontrivial operation so that array_copy might be clobbered.
    array_doubled = 2 * array

    # Use the reference to the copy held by eigency_map.
    return ndarray(_function_type_double(eigency_map))


cdef class FixedMatrixClass:
    cdef _FixedMatrixClass *thisptr;
    def __cinit__(self):
        self.thisptr = new _FixedMatrixClass()
    def __dealloc__(self):
        del self.thisptr
    def get_matrix(self):
        return ndarray(self.thisptr.get_matrix())
    def get_const_matrix(self):
        return ndarray(self.thisptr.get_const_matrix())
    def get_const_matrix_force_view(self):
        return ndarray_view(self.thisptr.get_const_matrix())

cdef class DynamicArrayClass:
    cdef _DynamicArrayClass *thisptr;
    def __cinit__(self, np.ndarray[np.float64_t, ndim=2] array):
        self.thisptr = new _DynamicArrayClass(Map[ArrayXXd](array))
    def __dealloc__(self):
        del self.thisptr
    def get_array(self):
        return ndarray(self.thisptr.get_array())

cdef class DynamicRowMajorArrayClass:
    cdef _DynamicRowMajorArrayClass *thisptr;
    def __cinit__(self, np.ndarray[np.float64_t, ndim=2] array):
        self.thisptr = new _DynamicRowMajorArrayClass(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor](array))
    def __dealloc__(self):
        del self.thisptr
    def get_array(self):
        return ndarray(self.thisptr.get_array())
    def get_array_copy(self):
        return ndarray(self.thisptr.get_array_copy())
