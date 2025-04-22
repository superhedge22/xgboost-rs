//! Type aliases for external dependencies
//! 
//! This module provides type aliases for types from external dependencies,
//! allowing users to not worry about specific versions of these dependencies.

pub use ndarray;

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