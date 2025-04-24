use encoder::OneHotEncoder;
use imputer::SimpleImputer;
use pipeline::Pipeline;
use scaler::StandardScaler;
use serde_json::Value;
use transform::ColumnTransformer;
use std::any::Any;

use crate::error::PreprocessingError;
use crate::types::{Array2, ArrayView2};
pub mod scaler;
pub mod imputer;
pub mod encoder;
pub mod transform;
pub mod pipeline;

#[derive(Debug, Clone, PartialEq)]
pub enum TransformerType {
    Imputer(SimpleImputer),
    StandardScaler(StandardScaler),
    OneHotEncoder(OneHotEncoder),
    Pipeline(Pipeline),
    ColumnTransformer(ColumnTransformer),
}

impl Transformer for TransformerType {
    fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<(), PreprocessingError> {
        match self {
            TransformerType::Imputer(imputer) => {
                imputer.fit(x, None)?;
                Ok(())
            }
            TransformerType::StandardScaler(scaler) => {
                scaler.fit(x)?;
                Ok(())
            }
            TransformerType::OneHotEncoder(encoder) => {
                encoder.fit(x, None)?;
                Ok(())
            }
            TransformerType::Pipeline(pipeline) => {
                pipeline.fit(x)?;
                Ok(())
            }
            TransformerType::ColumnTransformer(column_transformer) => {
                column_transformer.fit(x)?;
                Ok(())
            }
        }
    }

    fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        match self {
            TransformerType::Imputer(imputer) => imputer.transform(x),
            TransformerType::StandardScaler(scaler) => scaler.transform(x),
            TransformerType::OneHotEncoder(encoder) => encoder.transform(x),
            TransformerType::Pipeline(pipeline) => pipeline.transform(x),
            TransformerType::ColumnTransformer(column_transformer) => column_transformer.transform(x),
        }
    }

    fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        match self {
            TransformerType::Imputer(imputer) => imputer.fit_transform(x, None),
            TransformerType::StandardScaler(scaler) => scaler.fit_transform(x),
            TransformerType::OneHotEncoder(encoder) => encoder.fit_transform(x, None),
            TransformerType::Pipeline(pipeline) => pipeline.fit_transform(x),
            TransformerType::ColumnTransformer(column_transformer) => column_transformer.fit_transform(x),
        }
    }

    fn to_json_opt(&self) -> Option<Value> {
        match self {
            TransformerType::Imputer(imputer) => imputer.to_json_opt(),
            TransformerType::StandardScaler(scaler) => scaler.to_json_opt(),
            TransformerType::OneHotEncoder(encoder) => encoder.to_json_opt(),
            TransformerType::Pipeline(pipeline) => pipeline.to_json_opt(),
            TransformerType::ColumnTransformer(column_transformer) => column_transformer.to_json_opt(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Transformer trait for all preprocessing objects
pub trait Transformer: Any + Sized {
    fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<(), PreprocessingError>;
    fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError>;
    fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError>;
    
    /// Optional method to serialize the transformer to JSON.
    /// Default implementation returns None, which means the transformer doesn't support serialization.
    fn to_json_opt(&self) -> Option<Value> {
        None
    }
    
    /// Helper method to downcast to concrete type
    fn as_any(&self) -> &dyn Any;
    
    /// Helper method to downcast to concrete type (mutable)
    fn as_any_mut(&mut self) -> &mut dyn Any;
}


// Helper function to extract specific columns from a 2D array
fn extract_columns<T: Copy + 'static + Into<f64>>(x: &ArrayView2<T>, columns: &[usize]) -> Result<Array2<f64>, PreprocessingError> {
    // Check for out-of-bounds columns
    if let Some(&max_col) = columns.iter().max() {
        if max_col >= x.ncols() {
            return Err(PreprocessingError::ColumnIndexOutOfBounds(max_col));
        }
    }
    
    // Create a new array with selected columns
    let mut result: Array2<f64> = Array2::zeros((x.nrows(), columns.len()));
    
    // Fast copy using ndarray's column operations
    for (i, &col_idx) in columns.iter().enumerate() {
        let column = x.column(col_idx).to_owned();
        for row in 0..x.nrows() {
            result[[row, i]] = column[row].into();
        }
    }
    
    Ok(result)
}

// Helper function to concatenate arrays horizontally
fn horizontal_concat<T: Copy + 'static + Into<f64>>(arrays: &[Array2<T>], nrows: usize) -> Result<Array2<f64>, PreprocessingError> {
    if arrays.is_empty() {
        return Err(PreprocessingError::EmptyArray);
    }
    
    let total_cols = arrays.iter().map(|a| a.ncols()).sum();
    let mut result: Array2<f64> = Array2::zeros((nrows, total_cols));
    
    let mut col_offset = 0;
    for arr in arrays {
        for i in 0..arr.nrows() {
            for j in 0..arr.ncols() {
                result[[i, col_offset + j]] = arr[[i, j]].into();
            }
        }
        col_offset += arr.ncols();
    }
    
    Ok(result)
}