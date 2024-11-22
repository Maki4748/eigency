#include <iostream>
#include <stdexcept>
#include <complex>

#ifndef EIGENCY_CPP_H
#define EIGENCY_CPP_H

#include <Eigen/Core>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include "eigency.h"

typedef ::std::complex< double > __pyx_t_double_complex;
typedef ::std::complex< float > __pyx_t_float_complex;
typedef ::std::complex< long double > __pyx_t_long_double_complex;

#include "conversions_api.h"

namespace eigency {

template<typename Scalar>
inline PyArrayObject* _ddarray_view(Scalar *, long rows, long cols, bool is_row_major, long outer_stride=0, long inner_stride=0);
template<typename Scalar>
inline PyArrayObject* _ddarray_copy(const Scalar *, long rows, long cols, bool is_row_major, long outer_stride=0, long inner_stride=0);

// Strides:
// Eigen and numpy differ in their way of dealing with strides. Eigen has the concept of outer and
// inner strides, which are dependent on whether the array/matrix is row-major of column-major:
//     Inner stride: denotes the offset between succeeding elements in each row (row-major) or column (column-major).
//     Outer stride: denotes the offset between succeeding rows (row-major) or succeeding columns (column-major).
// In contrast, numpy's stride is simply a measure of how fast each dimension should be incremented.
// Consequently, a switch in numpy storage order from row-major to column-major involves a switch
// in strides, while it does not affect the stride in Eigen.

#define _DDAV(TYPE, FUNC_NAME_E, FUNC_NAME_C, FUNC_NAME_F) template<>                                                                       \
inline PyArrayObject* _ddarray_view< TYPE >(TYPE *data, long rows, long cols, bool is_row_major, long outer_stride, long inner_stride) {    \
    if (data == nullptr) {                                                                                                                  \
        return FUNC_NAME_E();                                                                                                               \
    } else if (is_row_major) {                                                                                                              \
        /* Eigen row-major mode: row_stride=outer_stride, and col_stride=inner_stride */                                                    \
        /* If no stride is given, the row_stride is set to the number of columns. */                                                        \
        return FUNC_NAME_C(data, rows, cols, outer_stride>0?outer_stride:cols, inner_stride>0?inner_stride:1);                              \
    } else {                                                                                                                                \
        /* Eigen column-major mode: row_stride=outer_stride, and col_stride=inner_stride */                                                 \
        /* If no stride is given, the cow_stride is set to the number of rows. */                                                           \
        return FUNC_NAME_F(data, rows, cols, inner_stride>0?inner_stride:1, outer_stride>0?outer_stride:rows);                              \
    }                                                                                                                                       \
}

#define _DDAC(TYPE, FUNC_NAME_E, FUNC_NAME_C, FUNC_NAME_F) template<>                                                                           \
inline PyArrayObject* _ddarray_copy< TYPE >(const TYPE *data, long rows, long cols, bool is_row_major, long outer_stride, long inner_stride) {  \
    if (data == nullptr) {                                                                                                                      \
        return FUNC_NAME_E();                                                                                                                   \
    } else if (is_row_major) {                                                                                                                  \
        return FUNC_NAME_C(data, rows, cols, outer_stride>0?outer_stride:cols, inner_stride>0?inner_stride:1);                                  \
    } else {                                                                                                                                    \
        return FUNC_NAME_F(data, rows, cols, inner_stride>0?inner_stride:1, outer_stride>0?outer_stride:rows);                                  \
    }                                                                                                                                           \
}

_DDAV(long double, ddarray_long_double, ddarray_long_double_C, ddarray_long_double_F)
_DDAC(long double, ddarray_long_double, ddarray_copy_long_double_C, ddarray_copy_long_double_F)
_DDAV(double, ddarray_double, ddarray_double_C, ddarray_double_F)
_DDAC(double, ddarray_double, ddarray_copy_double_C, ddarray_copy_double_F)
_DDAV(float, ddarray_float, ddarray_float_C, ddarray_float_F)
_DDAC(float, ddarray_float, ddarray_copy_float_C, ddarray_copy_float_F)
_DDAV(long, ddarray_long, ddarray_long_C, ddarray_long_F)
_DDAC(long, ddarray_long, ddarray_copy_long_C, ddarray_copy_long_F)
_DDAV(unsigned long, ddarray_ulong, ddarray_ulong_C, ddarray_ulong_F)
_DDAC(unsigned long, ddarray_ulong, ddarray_copy_ulong_C, ddarray_copy_ulong_F)
_DDAV(int, ddarray_int, ddarray_int_C, ddarray_int_F)
_DDAC(int, ddarray_int, ddarray_copy_int_C, ddarray_copy_int_F)
_DDAV(unsigned int, ddarray_uint, ddarray_uint_C, ddarray_uint_F)
_DDAC(unsigned int, ddarray_uint, ddarray_copy_uint_C, ddarray_copy_uint_F)
_DDAV(short, ddarray_short, ddarray_short_C, ddarray_short_F)
_DDAC(short, ddarray_short, ddarray_copy_short_C, ddarray_copy_short_F)
_DDAV(unsigned short, ddarray_ushort, ddarray_ushort_C, ddarray_ushort_F)
_DDAC(unsigned short, ddarray_ushort, ddarray_copy_ushort_C, ddarray_copy_ushort_F)
_DDAV(signed char, ddarray_schar, ddarray_schar_C, ddarray_schar_F)
_DDAC(signed char, ddarray_schar, ddarray_copy_schar_C, ddarray_copy_schar_F)
_DDAV(unsigned char, ddarray_uchar, ddarray_uchar_C, ddarray_uchar_F)
_DDAC(unsigned char, ddarray_uchar, ddarray_copy_uchar_C, ddarray_copy_uchar_F)
_DDAV(std::complex<long double>, ddarray_complex_long_double, ddarray_complex_long_double_C, ddarray_complex_long_double_F)
_DDAC(std::complex<long double>, ddarray_complex_long_double, ddarray_copy_complex_long_double_C, ddarray_copy_complex_long_double_F)
_DDAV(std::complex<double>, ddarray_complex_double, ddarray_complex_double_C, ddarray_complex_double_F)
_DDAC(std::complex<double>, ddarray_complex_double, ddarray_copy_complex_double_C, ddarray_copy_complex_double_F)
_DDAV(std::complex<float>, ddarray_complex_float, ddarray_complex_float_C, ddarray_complex_float_F)
_DDAC(std::complex<float>, ddarray_complex_float, ddarray_copy_complex_float_C, ddarray_copy_complex_float_F)

#undef _DDAV
#undef _DDAC

//
// Constructors
//

template <typename Derived>
inline PyArrayObject *ddarray(Eigen::PlainObjectBase<Derived> &m) {
    import_eigency__conversions();
    return _ddarray_view(m.data(), m.rows(), m.cols(), m.IsRowMajor);
}
// If C++11 is available, check if m is an r-value reference, in
// which case a copy should always be made
#if __cplusplus >= 201103L
template <typename Derived>
inline PyArrayObject *ddarray(Eigen::PlainObjectBase<Derived> &&m) {
    import_eigency__conversions();
    return _ddarray_copy(m.data(), m.rows(), m.cols(), m.IsRowMajor);
}
#endif
template <typename Derived>
inline PyArrayObject *ddarray(const Eigen::PlainObjectBase<Derived> &m) {
    import_eigency__conversions();
    return _ddarray_copy(m.data(), m.rows(), m.cols(), m.IsRowMajor);
}
template <typename Derived>
inline PyArrayObject *ddarray_view(Eigen::PlainObjectBase<Derived> &m) {
    import_eigency__conversions();
    return _ddarray_view(m.data(), m.rows(), m.cols(), m.IsRowMajor);
}
template <typename Derived>
inline PyArrayObject *ddarray_view(const Eigen::PlainObjectBase<Derived> &m) {
    import_eigency__conversions();
    return _ddarray_view(const_cast<typename Derived::Scalar*>(m.data()), m.rows(), m.cols(), m.IsRowMajor);
}
template <typename Derived>
inline PyArrayObject *ddarray_copy(const Eigen::PlainObjectBase<Derived> &m) {
    import_eigency__conversions();
    return _ddarray_copy(m.data(), m.rows(), m.cols(), m.IsRowMajor);
}

template <typename Derived, int MapOptions, typename Stride>
inline PyArrayObject *ddarray(Eigen::Map<Derived, MapOptions, Stride> &m) {
    import_eigency__conversions();
    return _ddarray_view(m.data(), m.rows(), m.cols(), m.IsRowMajor, m.outerStride(), m.innerStride());
}
template <typename Derived, int MapOptions, typename Stride>
inline PyArrayObject *ddarray(const Eigen::Map<Derived, MapOptions, Stride> &m) {
    import_eigency__conversions();
    // Since this is a map, we assume that ownership is correctly taken care
    // of, and we avoid taking a copy
    return _ddarray_view(const_cast<typename Derived::Scalar*>(m.data()), m.rows(), m.cols(), m.IsRowMajor, m.outerStride(), m.innerStride());
}
template <typename Derived, int MapOptions, typename Stride>
inline PyArrayObject *ddarray_view(Eigen::Map<Derived, MapOptions, Stride> &m) {
    import_eigency__conversions();
    return _ddarray_view(m.data(), m.rows(), m.cols(), m.IsRowMajor, m.outerStride(), m.innerStride());
}
template <typename Derived, int MapOptions, typename Stride>
inline PyArrayObject *ddarray_view(const Eigen::Map<Derived, MapOptions, Stride> &m) {
    import_eigency__conversions();
    return _ddarray_view(const_cast<typename Derived::Scalar*>(m.data()), m.rows(), m.cols(), m.IsRowMajor, m.outerStride(), m.innerStride());
}
template <typename Derived, int MapOptions, typename Stride>
inline PyArrayObject *ddarray_copy(const Eigen::Map<Derived, MapOptions, Stride> &m) {
    import_eigency__conversions();
    return _ddarray_copy(m.data(), m.rows(), m.cols(), m.IsRowMajor, m.outerStride(), m.innerStride());
}


template <typename MatrixType,
          int _MapOptions = Eigen::Unaligned,
          typename _StrideType=Eigen::Stride<0,0> >
class MapBase: public Eigen::Map<MatrixType, _MapOptions, _StrideType> {
public:
    typedef Eigen::Map<MatrixType, _MapOptions, _StrideType> Base;
    typedef typename Base::Scalar Scalar;

    MapBase(Scalar* data,
            long rows,
            long cols,
            _StrideType stride=_StrideType())
        : Base(data,
               // If both dimensions are dynamic or dimensions match, accept dimensions as they are
               ((Base::RowsAtCompileTime==Eigen::Dynamic && Base::ColsAtCompileTime==Eigen::Dynamic) ||
                (Base::RowsAtCompileTime==rows && Base::ColsAtCompileTime==cols))
               ? rows
               // otherwise, test if swapping them makes them fit
               : ((Base::RowsAtCompileTime==cols || Base::ColsAtCompileTime==rows)
                  ? cols
                  : rows),
               ((Base::RowsAtCompileTime==Eigen::Dynamic && Base::ColsAtCompileTime==Eigen::Dynamic) ||
                (Base::RowsAtCompileTime==rows && Base::ColsAtCompileTime==cols))
               ? cols
               : ((Base::RowsAtCompileTime==cols || Base::ColsAtCompileTime==rows)
                  ? rows
                  : cols),
               stride
            )  {}

    MapBase &operator=(const MatrixType &other) {
        Base::operator=(other);
        return *this;
    }

    virtual ~MapBase() { }
};


template <template<class,int,int,int,int,int> class EigencyDenseBase,
          typename Scalar,
          int _Rows, int _Cols,
          int _Options = Eigen::AutoAlign |
#if defined(__GNUC__) && __GNUC__==3 && __GNUC_MINOR__==4
    // workaround a bug in at least gcc 3.4.6
    // the innermost ?: ternary operator is misparsed. We write it slightly
    // differently and this makes gcc 3.4.6 happy, but it's ugly.
    // The error would only show up with EIGEN_DEFAULT_TO_ROW_MAJOR is defined
    // (when EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION is RowMajor)
                          ( (_Rows==1 && _Cols!=1) ? Eigen::RowMajor
// EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION contains explicit namespace since Eigen 3.1.19
#if EIGEN_VERSION_AT_LEAST(3,2,90)
                          : !(_Cols==1 && _Rows!=1) ? EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION
#else
                          : !(_Cols==1 && _Rows!=1) ? Eigen::EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION
#endif
                          : ColMajor ),
#else
                          ( (_Rows==1 && _Cols!=1) ? Eigen::RowMajor
                          : (_Cols==1 && _Rows!=1) ? Eigen::ColMajor
// EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION contains explicit namespace since Eigen 3.1.19
#if EIGEN_VERSION_AT_LEAST(3,2,90)
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
#else
                          : Eigen::EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
#endif
#endif
          int _MapOptions = Eigen::Unaligned,
          int _StrideOuter=0, int _StrideInner=0,
          int _MaxRows = _Rows,
          int _MaxCols = _Cols>
class FlattenedMap: public MapBase<EigencyDenseBase<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, _MapOptions, Eigen::Stride<_StrideOuter, _StrideInner> >  {
public:
    typedef MapBase<EigencyDenseBase<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, _MapOptions, Eigen::Stride<_StrideOuter, _StrideInner> > Base;

    FlattenedMap()
        : Base(NULL,
               _Rows == Eigen::Dynamic ? 0 : _Rows,
               _Cols == Eigen::Dynamic ? 0 : _Cols),
          object_(NULL) {}

    FlattenedMap(Scalar *data, long rows, long cols, long outer_stride=0, long inner_stride=0)
        : Base(data, rows, cols,
               Eigen::Stride<_StrideOuter, _StrideInner>(outer_stride, inner_stride)),
          object_(NULL) {
    }

    FlattenedMap(PyArrayObject *object)
        : Base((Scalar *) PyArray_DATA(((PyArrayObject*)object)),
        // : Base(_from_numpy<Scalar>((PyArrayObject*)object),
               (PyArray_NDIM((PyArrayObject*)object) == 2) ? PyArray_DIMS((PyArrayObject*)object)[0] : 1,
               (PyArray_NDIM((PyArrayObject*)object) == 2) ? PyArray_DIMS((PyArrayObject*)object)[1] : PyArray_DIMS((PyArrayObject*)object)[0],
               Eigen::Stride<_StrideOuter, _StrideInner>(_StrideOuter != Eigen::Dynamic ? _StrideOuter : (PyArray_NDIM((PyArrayObject*)object) == 2) ? PyArray_DIMS((PyArrayObject*)object)[0] : 1,
                                                         _StrideInner != Eigen::Dynamic ? _StrideInner : (PyArray_NDIM((PyArrayObject*)object) == 2) ? PyArray_DIMS((PyArrayObject*)object)[1] : PyArray_DIMS((PyArrayObject*)object)[0])),
          object_(object) {

        if (((PyObject*)object != Py_None) && !PyArray_ISONESEGMENT(object))
            throw std::invalid_argument("Numpy array must be a in one contiguous segment to be able to be transferred to a Eigen Map.");

        Py_XINCREF(object_);
    }
    FlattenedMap &operator=(const FlattenedMap &other) {
        if (this == &other){
            return *this;
        }

        Py_XDECREF(object_);
        if (other.object_) {
            new (this) FlattenedMap(other.object_);
        } else {
            // Replace the memory that we point to (not a memory allocation)
            new (this) FlattenedMap(const_cast<Scalar*>(other.data()),
                                    other.rows(),
                                    other.cols(),
                                    other.outerStride(),
                                    other.innerStride());
        }

        return *this;
    }

    operator Base() const {
        return static_cast<Base>(*this);
    }

    operator Base&() const {
        return static_cast<Base&>(*this);
    }

    operator EigencyDenseBase<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>() const {
        return EigencyDenseBase<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>(static_cast<Base>(*this));
    }

    virtual ~FlattenedMap() {
        Py_XDECREF(object_);
    }

private:
    PyArrayObject * const object_;
};


template <typename MatrixType>
class Map: public MapBase<MatrixType> {
public:
    typedef MapBase<MatrixType> Base;
    typedef typename MatrixType::Scalar Scalar;

    enum {
        RowsAtCompileTime = Base::Base::RowsAtCompileTime,
        ColsAtCompileTime = Base::Base::ColsAtCompileTime
    };

    Map()
        : Base(NULL,
               (RowsAtCompileTime == Eigen::Dynamic) ? 0 : RowsAtCompileTime,
               (ColsAtCompileTime == Eigen::Dynamic) ? 0 : ColsAtCompileTime),
          object_(NULL) {
    }

    Map(Scalar *data, long rows, long cols)
        : Base(data, rows, cols),
          object_(NULL) {}

    Map(PyArrayObject *object)
        : Base((PyObject*)object == Py_None? NULL: (Scalar *)PyArray_DATA(object),
               // ROW: If array is in row-major order, transpose (see README)
               (PyObject*)object == Py_None? 0 :
               (!PyArray_IS_F_CONTIGUOUS(object)
                ? ((PyArray_NDIM(object) == 1)
                   ? 1  // ROW: If 1D row-major numpy array, set to 1 (row vector)
                   : PyArray_DIMS(object)[1])
                : PyArray_DIMS(object)[0]),
               // COLUMN: If array is in row-major order: transpose (see README)
               (PyObject*)object == Py_None? 0 :
               (!PyArray_IS_F_CONTIGUOUS(object)
                ? PyArray_DIMS(object)[0]
                : ((PyArray_NDIM(object) == 1)
                   ? 1  // COLUMN: If 1D col-major numpy array, set to length (column vector)
                   : PyArray_DIMS(object)[1]))),
          object_(object) {

        if (((PyObject*)object != Py_None) && !PyArray_ISONESEGMENT(object))
            throw std::invalid_argument("Numpy array must be a in one contiguous segment to be able to be transferred to a Eigen Map.");
        Py_XINCREF(object_);
    }

    Map &operator=(const Map &other) {
        if (this == &other){
            return *this;
        }

        Py_XDECREF(object_);
        if (other.object_) {
            new (this) Map(other.object_);
        } else {
            // Replace the memory that we point to (not a memory allocation)
            new (this) Map(const_cast<Scalar*>(other.data()),
                          other.rows(),
                          other.cols());
        }

        return *this;
    }

    Map &operator=(const MatrixType &other) {
        MapBase<MatrixType>::operator=(other);
        return *this;
    }

    operator Base() const {
        return static_cast<Base>(*this);
    }

    operator Base&() const {
        return static_cast<Base&>(*this);
    }

    operator MatrixType() const {
        return MatrixType(static_cast<Base>(*this));
    }

    virtual ~Map() {
        Py_XDECREF(object_);
    }

private:
    PyArrayObject * const object_;
};


}

#endif