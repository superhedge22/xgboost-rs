use crate::error::PreprocessingError;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use super::Transformer;

/// StandardScaler standardizes features by removing the mean and scaling to unit variance.
///
/// This transformer scales input features by first centering them (removing the mean) and then
/// dividing by their standard deviation, resulting in features with zero mean and unit variance.
/// This is a common preprocessing step for many machine learning algorithms.
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use xgboostrs::preprocessing::scaler::StandardScaler;
///
/// let mut scaler = StandardScaler::new();
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// // Fit and transform the data
/// let scaled_data = scaler.fit_transform(&data.view()).unwrap();
/// ```
pub struct StandardScaler {
    /// The mean of each feature, computed during fitting
    mean: Option<Array1<f64>>,
    /// The standard deviation of each feature, computed during fitting
    scale: Option<Array1<f64>>,
}

impl StandardScaler {
    /// Creates a new `StandardScaler` instance.
    ///
    /// # Returns
    ///
    /// A new `StandardScaler` with no computed statistics.
    ///
    /// # Examples
    ///
    /// ```
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    ///
    /// let scaler = StandardScaler::new();
    /// ```
    pub fn new() -> Self {
        StandardScaler {
            mean: None,
            scale: None,
        }
    }

    /// Computes the mean and standard deviation to be used for later scaling.
    ///
    /// This method calculates the mean and standard deviation of each feature in the input data,
    /// which will be used to transform data in subsequent calls to `transform()`.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data as a 2D array view, where each column is a feature and each row is a sample
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, PreprocessingError>` - The fitted scaler on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::EmptyArray` - If the input array has no rows
    /// * `PreprocessingError::ComputeMean` - If the mean computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    ///
    /// let mut scaler = StandardScaler::new();
    /// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// scaler.fit(&data.view()).unwrap();
    /// ```
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<&mut Self, PreprocessingError> {
        if x.nrows() == 0 {
            return Err(PreprocessingError::EmptyArray);
        }
        
        let mean = x.mean_axis(Axis(0))
            .ok_or(PreprocessingError::ComputeMean)?;
        
        let var = x.var_axis(Axis(0), 1.0);
        let mut scale = var.mapv(f64::sqrt);
        
        // Handle zeros in scale (avoid division by zero)
        for val in scale.iter_mut() {
            if *val == 0.0 {
                *val = 1.0;
            }
        }
        
        self.mean = Some(mean);
        self.scale = Some(scale);
        
        Ok(self)
    }
    
    /// Standardizes features by removing the mean and scaling to unit variance.
    ///
    /// Transforms the input data using the mean and standard deviation computed during
    /// the fitting phase. Each feature will be centered by subtracting the mean and then
    /// scaled by dividing by the standard deviation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The standardized data on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::NotFitted` - If the scaler has not been fitted
    /// * `PreprocessingError::FeatureMismatch` - If the number of features in `x` differs from what was used during fitting
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    ///
    /// let mut scaler = StandardScaler::new();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// scaler.fit(&train_data.view()).unwrap();
    ///
    /// let test_data = array![[0.0, 0.0], [10.0, 10.0]];
    /// let scaled_data = scaler.transform(&test_data.view()).unwrap();
    /// ```
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        let mean = self.mean.as_ref()
            .ok_or(PreprocessingError::NotFitted)?;
        
        let scale = self.scale.as_ref().unwrap();  // Safe to unwrap since mean is Some
        
        if x.ncols() != mean.len() {
            return Err(PreprocessingError::FeatureMismatch(x.ncols(), mean.len()));
        }
        
        // Create output array
        let mut result = Array2::zeros((x.nrows(), x.ncols()));
        
        // Apply transformation: (X - mean) / std
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                result[[i, j]] = (x[[i, j]] - mean[j]) / scale[j];
            }
        }
        
        Ok(result)
    }
    
    /// Fits the scaler on the input data and then transforms it.
    ///
    /// This is a convenience method that combines `fit` and `transform` into a single operation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The standardized data on success, or an error
    ///
    /// # Errors
    ///
    /// May return any errors that could be returned by `fit` or `transform`
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    ///
    /// let mut scaler = StandardScaler::new();
    /// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let scaled_data = scaler.fit_transform(&data.view()).unwrap();
    /// ```
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(x)?;
        self.transform(&x.view())
    }
}

/// Implementation of the `Transformer` trait for `StandardScaler`.
///
/// This allows the `StandardScaler` to be used with the general transformer interface.
impl Transformer for StandardScaler {
    /// Fits the scaler on the input data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit on
    ///
    /// # Returns
    ///
    /// `Result<(), PreprocessingError>` - Success or an error
    fn fit(&mut self, x: &ArrayView2<f64>) -> Result<(), PreprocessingError> {
        self.fit(x)?;
        Ok(())
    }

    /// Transforms the input data by standardizing features.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform
    ///
    /// # Returns
    ///
    /// `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.transform(x)
    }

    /// Fits the scaler and transforms the input data in one operation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit_transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_standard_scaler_new() {
        let scaler = StandardScaler::new();
        assert!(scaler.mean.is_none());
        assert!(scaler.scale.is_none());
    }
    
    #[test]
    fn test_standard_scaler_fit() {
        let mut scaler = StandardScaler::new();
        
        // Simple dataset with two features
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        
        scaler.fit(&x.view()).unwrap();
        
        // Check that mean and scale were computed correctly
        let mean = scaler.mean.as_ref().unwrap();
        let scale = scaler.scale.as_ref().unwrap();
        
        assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scale[0], 2.0, epsilon = 1e-10);  // std_dev of [1,3,5]
        assert_abs_diff_eq!(scale[1], 2.0, epsilon = 1e-10);  // std_dev of [2,4,6]
    }
    
    #[test]
    fn test_standard_scaler_fit_empty_array() {
        let mut scaler = StandardScaler::new();
        let x = Array2::<f64>::zeros((0, 2));
        
        let result = scaler.fit(&x.view());
        assert!(matches!(result, Err(PreprocessingError::EmptyArray)));
    }
    
    #[test]
    fn test_standard_scaler_transform() {
        let mut scaler = StandardScaler::new();
        
        // Simple dataset with two features
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        
        // Fit the scaler
        scaler.fit(&x.view()).unwrap();
        
        // Transform the same data
        let result = scaler.transform(&x.view()).unwrap();
        
        // Expected values after standardization:
        // For the first feature [1, 3, 5]:
        //   mean = 3, std = 2
        //   standardized = [-1.0, 0.0, 1.0]
        // For the second feature [2, 4, 6]:
        //   mean = 4, std = 2
        //   standardized = [-1.0, 0.0, 1.0]
        
        assert_abs_diff_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 1]], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_standard_scaler_transform_not_fitted() {
        let scaler = StandardScaler::new();
        
        // Try to transform without fitting
        let x = array![[1.0, 2.0]];
        let result = scaler.transform(&x.view());
        
        // Should fail with NotFitted error
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }
    
    #[test]
    fn test_standard_scaler_feature_mismatch() {
        let mut scaler = StandardScaler::new();
        
        // Fit with 2 features
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit(&x_train.view()).unwrap();
        
        // Transform with 3 features
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = scaler.transform(&x_test.view());
        
        // Should fail with FeatureMismatch error
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch(3, 2))
        ));
    }
    
    #[test]
    fn test_standard_scaler_fit_transform() {
        let mut scaler = StandardScaler::new();
        
        // Simple dataset with two features
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        
        // Apply fit_transform
        let result = scaler.fit_transform(&x.view()).unwrap();
        
        // Check the result is the same as fit and then transform
        assert_abs_diff_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 1]], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_standard_scaler_zero_variance() {
        let mut scaler = StandardScaler::new();
        
        // Dataset with zero variance in second feature
        let x = array![[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]];
        
        scaler.fit(&x.view()).unwrap();
        
        // Transform the data
        let result = scaler.transform(&x.view()).unwrap();
        
        // For zero variance features, scale should be 1.0
        // First feature should be standardized normally
        assert_abs_diff_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], 1.0, epsilon = 1e-10);
        
        // Second feature should be centered but not scaled (since variance is 0)
        // Original values are all 5.0, mean is 5.0, so (5-5)/1 = 0
        assert_abs_diff_eq!(result[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 1]], 0.0, epsilon = 1e-10);
    }
}