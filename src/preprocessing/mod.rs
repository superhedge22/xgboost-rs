use ndarray::{Array2, ArrayView2};

use crate::error::PreprocessingError;

pub mod scaler;
pub mod imputer;
pub mod encoder;
pub mod transform;
pub mod pipeline;



// Transformer trait for all preprocessing objects
pub trait Transformer {
    fn fit(&mut self, x: &ArrayView2<f32>) -> Result<(), PreprocessingError>;
    fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError>;
    fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError>;
}


// Helper function to extract specific columns from a 2D array
fn extract_columns(x: &ArrayView2<f32>, columns: &[usize]) -> Result<Array2<f32>, PreprocessingError> {
    // Check for out-of-bounds columns
    if let Some(&max_col) = columns.iter().max() {
        if max_col >= x.ncols() {
            return Err(PreprocessingError::ColumnIndexOutOfBounds(max_col));
        }
    }
    
    // Create a new array with selected columns
    let mut result = Array2::zeros((x.nrows(), columns.len()));
    
    // Fast copy using ndarray's column operations
    for (i, &col_idx) in columns.iter().enumerate() {
        let column = x.column(col_idx).to_owned();
        for row in 0..x.nrows() {
            result[[row, i]] = column[row];
        }
    }
    
    Ok(result)
}

// Helper function to concatenate arrays horizontally
fn horizontal_concat(arrays: &[Array2<f32>], nrows: usize) -> Result<Array2<f32>, PreprocessingError> {
    if arrays.is_empty() {
        return Err(PreprocessingError::EmptyArray);
    }
    
    let total_cols = arrays.iter().map(|a| a.ncols()).sum();
    let mut result = Array2::zeros((nrows, total_cols));
    
    let mut col_offset = 0;
    for arr in arrays {
        for i in 0..arr.nrows() {
            for j in 0..arr.ncols() {
                result[[i, col_offset + j]] = arr[[i, j]];
            }
        }
        col_offset += arr.ncols();
    }
    
    Ok(result)
}