use ndarray::{Array2, ArrayView2};

use crate::error::PreprocessingError;

use super::{extract_columns, horizontal_concat, Transformer};

/// ColumnTransformer applies transformers to columns of an array.
///
/// This structure allows applying different transformers to specific subsets of columns
/// in a dataset. Each transformer operates independently on its designated columns, and
/// the results are concatenated horizontally to form the final output.
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use xgboostrs::preprocessing::transform::ColumnTransformer;
/// use xgboostrs::preprocessing::scaler::StandardScaler;
/// use xgboostrs::preprocessing::Transformer;
///
/// // Create transformers for different column subsets
/// let transformers = vec![
///     ("scale_first".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0]),
///     ("scale_last".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![1]),
/// ];
///
/// let mut transformer = ColumnTransformer::new(transformers);
///
/// // Sample data with 2 columns
/// let data = array![
///     [1.0, 2.0],
///     [3.0, 4.0],
///     [5.0, 6.0],
/// ];
///
/// // Fit and transform the data
/// let transformed = transformer.fit_transform(&data.view()).unwrap();
/// ```
pub struct ColumnTransformer {
    transformers: Vec<(String, Box<dyn Transformer>, Vec<usize>)>,
    fitted: bool,
}

impl ColumnTransformer {
    /// Creates a new ColumnTransformer with the specified transformers.
    ///
    /// # Parameters
    ///
    /// * `transformers` - A vector of (name, transformer, columns) triplets that define
    ///   which transformer to apply to which columns.
    ///
    /// # Returns
    ///
    /// A new unfitted ColumnTransformer with the provided transformers.
    ///
    /// # Examples
    ///
    /// ```
    /// use xgboostrs::preprocessing::transform::ColumnTransformer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create transformers for different column subsets
    /// let transformers = vec![
    ///     ("scale_first".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0]),
    ///     ("scale_second".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![1]),
    /// ];
    ///
    /// let transformer = ColumnTransformer::new(transformers);
    /// ```
    pub fn new(transformers: Vec<(String, Box<dyn Transformer>, Vec<usize>)>) -> Self {
        ColumnTransformer {
            transformers,
            fitted: false,
        }
    }
    
    /// Fits all transformers on their respective columns of the input data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit the transformers on
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, PreprocessingError>` - The fitted transformer on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::ColumnIndexOutOfBounds` - If a column index is out of bounds
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::transform::ColumnTransformer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create transformers
    /// let transformers = vec![
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0, 1]),
    /// ];
    ///
    /// let mut transformer = ColumnTransformer::new(transformers);
    ///
    /// // Sample data
    /// let data = array![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ];
    ///
    /// // Fit the transformer
    /// transformer.fit(&data.view()).unwrap();
    /// ```
    fn fit(&mut self, x: &ArrayView2<f32>) -> Result<&mut Self, PreprocessingError> {
        for (_, transformer, columns) in &mut self.transformers {
            // Extract subset of columns
            let x_subset = extract_columns(x, columns)?;
            
            // Fit the transformer
            transformer.fit(&x_subset.view())?;
        }
        
        self.fitted = true;
        Ok(self)
    }
    
    /// Applies transformations to their respective columns and concatenates results.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The transformed data on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::NotFitted` - If the transformer has not been fitted
    /// * `PreprocessingError::ColumnIndexOutOfBounds` - If a column index is out of bounds
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::transform::ColumnTransformer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create and fit a transformer
    /// let transformers = vec![
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0, 1]),
    /// ];
    ///
    /// let mut transformer = ColumnTransformer::new(transformers);
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0]];
    /// transformer.fit(&train_data.view()).unwrap();
    ///
    /// // Transform new data
    /// let test_data = array![[2.0, 3.0], [4.0, 5.0]];
    /// let transformed = transformer.transform(&test_data.view()).unwrap();
    /// ```
    fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        
        let mut transformed_arrays = Vec::new();
        
        for (_, transformer, columns) in &self.transformers {
            // Extract subset of columns
            let x_subset = extract_columns(x, columns)?;
            
            // Transform the subset
            let x_transformed = transformer.transform(&x_subset.view())?;
            
            transformed_arrays.push(x_transformed);
        }
        
        // Concatenate the transformed arrays horizontally
        horizontal_concat(&transformed_arrays, x.nrows())
    }
    
    /// Fits all transformers to the data and then transforms it in one operation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The transformed data on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::ColumnIndexOutOfBounds` - If a column index is out of bounds
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::transform::ColumnTransformer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create transformers for different columns
    /// let transformers = vec![
    ///     ("scale_first".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0]),
    ///     ("scale_second".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![1]),
    /// ];
    ///
    /// let mut transformer = ColumnTransformer::new(transformers);
    ///
    /// // Sample data
    /// let data = array![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    ///     [5.0, 6.0],
    /// ];
    ///
    /// // Fit and transform in one step
    /// let transformed = transformer.fit_transform(&data.view()).unwrap();
    /// ```
    fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.fit(x)?;
        self.transform(&x.view())
    }
}

impl Transformer for ColumnTransformer {
    fn fit(&mut self, x: &ArrayView2<f32>) -> Result<(), PreprocessingError> {
        self.fit(x)?;
        Ok(())
    }

    fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.transform(x)
    }

    fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.fit_transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    use crate::preprocessing::scaler::StandardScaler;
    use std::fmt::Debug;

    // A simple mock transformer that multiplies values by a constant factor
    struct MockTransformer {
        fitted: bool,
        factor: f32,
    }

    impl MockTransformer {
        fn new(factor: f32) -> Self {
            MockTransformer {
                fitted: false,
                factor,
            }
        }
    }

    impl Transformer for MockTransformer {
        fn fit(&mut self, _: &ArrayView2<f32>) -> Result<(), PreprocessingError> {
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
            if !self.fitted {
                return Err(PreprocessingError::NotFitted);
            }
            let mut result = x.to_owned();
            result.mapv_inplace(|v| v * self.factor);
            Ok(result)
        }

        fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
            self.fit(x)?;
            self.transform(x)
        }
    }

    impl Debug for MockTransformer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockTransformer {{ factor: {} }}", self.factor)
        }
    }

    #[test]
    fn test_column_transformer_new() {
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![0, 1]),
        ];
        
        let transformer = ColumnTransformer::new(transformers);
        assert!(!transformer.fitted);
    }

    #[test]
    fn test_column_transformer_fit() {
        // Create transformers for different column subsets
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![0]),
            ("t2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>, vec![1, 2]),
        ];
        
        let mut transformer = ColumnTransformer::new(transformers);
        
        // Sample data with 3 columns
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        
        // Fit the transformer
        transformer.fit(&x.view()).unwrap();
        
        // Check that it's now fitted
        assert!(transformer.fitted);
    }

    #[test]
    fn test_column_transformer_transform() {
        // Create transformers for different column subsets
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![0]),
            ("t2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>, vec![1, 2]),
        ];
        
        let mut transformer = ColumnTransformer::new(transformers);
        
        // Sample data with 3 columns
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        
        // Fit the transformer
        transformer.fit(&x.view()).unwrap();
        
        // Transform the data
        let result = transformer.transform(&x.view()).unwrap();
        
        // Expected result:
        // - First column [1.0, 4.0] multiplied by 2 = [2.0, 8.0]
        // - Second and third columns [2.0, 3.0], [5.0, 6.0] multiplied by 3 = [6.0, 9.0], [15.0, 18.0]
        assert_eq!(result.shape(), &[2, 3]);
        
        // Check first transformer's result (first column)
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 8.0, epsilon = 1e-10);
        
        // Check second transformer's result (second and third columns)
        assert_abs_diff_eq!(result[[0, 1]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 2]], 18.0, epsilon = 1e-10);
    }

    #[test]
    fn test_column_transformer_fit_transform() {
        // Create transformers for different column subsets
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![0]),
            ("t2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>, vec![1, 2]),
        ];
        
        let mut transformer = ColumnTransformer::new(transformers);
        
        // Sample data with 3 columns
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        
        // Fit and transform the data in one step
        let result = transformer.fit_transform(&x.view()).unwrap();
        
        // Check the transformer is fitted
        assert!(transformer.fitted);
        
        // Check the result is the same as separate fit and transform
        assert_eq!(result.shape(), &[2, 3]);
        
        // Check first transformer's result (first column)
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 8.0, epsilon = 1e-10);
        
        // Check second transformer's result (second and third columns)
        assert_abs_diff_eq!(result[[0, 1]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 2]], 18.0, epsilon = 1e-10);
    }

    #[test]
    fn test_column_transformer_not_fitted() {
        // Create a transformer without fitting it
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![0]),
        ];
        
        let transformer = ColumnTransformer::new(transformers);
        
        // Sample data
        let x = array![[1.0, 2.0, 3.0]];
        
        // Try to transform without fitting
        let result = transformer.transform(&x.view());
        
        // Should fail with NotFitted error
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_column_transformer_with_standard_scaler() {
        // Create a ColumnTransformer with a StandardScaler for each column
        let transformers = vec![
            ("scale_first".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![0]),
            ("scale_last".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>, vec![1]),
        ];
        
        let mut transformer = ColumnTransformer::new(transformers);
        
        // Sample data with 2 columns
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit and transform the data
        let result = transformer.fit_transform(&x.view()).unwrap();
        
        // Expected result: each column standardized separately
        // First column [1, 3, 5] has mean=3, std=2 -> [-1, 0, 1]
        // Second column [2, 4, 6] has mean=4, std=2 -> [-1, 0, 1]
        assert_eq!(result.shape(), &[3, 2]);
        
        assert_abs_diff_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], 1.0, epsilon = 1e-10);
        
        assert_abs_diff_eq!(result[[0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_column_transformer_column_index_out_of_bounds() {
        // Create a transformer with invalid column index
        let transformers = vec![
            ("t1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>, vec![10]), // Out of bounds
        ];
        
        let mut transformer = ColumnTransformer::new(transformers);
        
        // Sample data with 3 columns
        let x = array![[1.0, 2.0, 3.0]];
        
        // Try to fit with invalid column index
        let result = transformer.fit(&x.view());
        
        // Should fail with ColumnIndexOutOfBounds error
        assert!(matches!(result, Err(PreprocessingError::ColumnIndexOutOfBounds(10))));
    }
}