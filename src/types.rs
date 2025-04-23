//! Type aliases for external dependencies
//! 
//! This module provides type aliases for types from external dependencies,
//! allowing users to not worry about specific versions of these dependencies.

pub use ndarray;
use crate::error::XGBResult;
use crate::dmatrix::DMatrix;

/// 1-dimensional array of f32 values
pub type Array1F = ndarray::Array1<f32>;

/// 2-dimensional array of f32 values
pub type Array2F = ndarray::Array2<f32>;

/// 2-dimensional array view of f32 values
pub type ArrayView2F<'a> = ndarray::ArrayView2<'a, f32>;

/// 3-dimensional array of f32 values
pub type Array3F = ndarray::Array3<f32>; 

/// 3-dimensional array view of f32 values
pub type ArrayView3F<'a> = ndarray::ArrayView3<'a, f32>; 

/// 2-dimensional array of any type
pub type Array2<T> = ndarray::Array2<T>;
/// 2-dimensional array view of any type
pub type ArrayView2<'a, T> = ndarray::ArrayView2<'a, T>;

/// 3-dimensional array of any type
pub type Array3<T> = ndarray::Array3<T>;

/// 3-dimensional array view of any type
pub type ArrayView3<'a, T> = ndarray::ArrayView3<'a, T>;

/// 1-dimensional array of any type
pub type Array1<T> = ndarray::Array1<T>;

/// 1-dimensional array view of any type
pub type ArrayView1<'a, T> = ndarray::ArrayView1<'a, T>;

/// Trait for types that can be converted into a DMatrix
pub trait IntoDMatrix {
    /// Convert the implementor into a DMatrix
    fn into_dmatrix(self) -> XGBResult<DMatrix>;
}

// Implement the IntoDMatrix trait for floats and ints
impl<'a, T> IntoDMatrix for Array2<T>
where
    T: Copy,
    T: Into<f64>, // Numeric type constraint
{
    fn into_dmatrix(self) -> XGBResult<DMatrix> {
        // Convert to Array2F using as cast
        let data_f32: Array2F = self.mapv(|x| x.into() as f32);

        if data_f32.is_standard_layout() {
            if let Some(slice) = data_f32.as_slice() {
                DMatrix::from_dense(slice, data_f32.nrows())
            } else {
                let data: Vec<f32> = data_f32.iter().copied().collect();
                DMatrix::from_dense(&data, data_f32.nrows())
            }
        } else {
            let data: Vec<f32> = data_f32.iter().copied().collect();
            DMatrix::from_dense(&data, data_f32.nrows())
        }
    }
}

impl<'a, T> IntoDMatrix for ArrayView2<'a, T>
where
    T: Copy + 'static,
    T: Into<f64>,
{
    fn into_dmatrix(self) -> XGBResult<DMatrix> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let data_f32: ArrayView2F<'a> = unsafe { std::mem::transmute(self) };
            DMatrix::from_array_view(&data_f32)
        } else {
            let data_f32: Array2F = self.to_owned().mapv(|x| x.into() as f32);
            DMatrix::from_array_view(&data_f32.view())
        }
    }
} 