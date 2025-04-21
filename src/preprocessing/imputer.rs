use std::collections::HashMap;

use ndarray::{s, Array1, Array2, ArrayView2};

use crate::{error::PreprocessingError, parameters::preprocessing::ImputationStrategy};

use super::Transformer;

/// SimpleImputer imputes missing values using a specified strategy.
///
/// This transformer replaces missing values (represented as NaN) in a dataset with computed values
/// according to a specified strategy. The imputation strategies include mean, median, most frequent,
/// or a constant value.
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use xgboostrs::preprocessing::imputer::SimpleImputer;
/// use xgboostrs::parameters::preprocessing::ImputationStrategy;
/// use xgboostrs::preprocessing::Transformer;
///
/// // Create an imputer using the mean strategy
/// let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
///
/// // Data with missing values
/// let data = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
///
/// // Fit and transform the data
/// let result = imputer.fit_transform(&data.view()).unwrap();
/// ```
pub struct SimpleImputer {
    /// The imputation strategy to use for missing values
    strategy: ImputationStrategy,
    /// The imputation values calculated during fitting for each feature
    statistics: Option<Array1<f32>>,
}

impl SimpleImputer {
    /// Creates a new `SimpleImputer` with the specified strategy.
    ///
    /// # Parameters
    ///
    /// * `strategy` - The imputation strategy to use for replacing missing values
    ///
    /// # Returns
    ///
    /// A new `SimpleImputer` instance with the specified strategy and no learned statistics
    ///
    /// # Examples
    ///
    /// ```
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create imputer with mean strategy
    /// let mean_imputer = SimpleImputer::new(ImputationStrategy::Mean);
    ///
    /// // Create imputer with constant value strategy
    /// let constant_imputer = SimpleImputer::new(ImputationStrategy::Constant(0.0));
    /// ```
    pub fn new(strategy: ImputationStrategy) -> Self {
        SimpleImputer {
            strategy,
            statistics: None,
        }
    }
    
    /// Learns the statistics required for imputation from the input data.
    ///
    /// This method calculates imputation values for each feature according to the 
    /// strategy specified when the imputer was created.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data as a 2D array view, where each column is a feature and each row is a sample
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, PreprocessingError>` - The fitted imputer on success, or an error
    ///
    /// # Errors
    ///
    /// Returns `PreprocessingError::MostFrequent` if using `MostFrequent` strategy and no valid values are found
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
    /// let data = array![[1.0, 2.0], [3.0, f32::NAN], [f32::NAN, 6.0]];
    /// imputer.fit(&data.view()).unwrap();
    /// ```
    fn fit(&mut self, x: &ArrayView2<f32>) -> Result<&mut Self, PreprocessingError> {
        let n_features = x.ncols();
        let mut stats = Array1::zeros(n_features);
        
        for j in 0..n_features {
            let column = x.slice(s![.., j]);
            
            match &self.strategy {
                ImputationStrategy::Mean => {
                    // Filter out NaN values and compute mean
                    let valid_values: Vec<f32> = column.iter()
                        .filter(|&&x| !x.is_nan())
                        .copied()
                        .collect();
                    
                    if valid_values.is_empty() {
                        stats[j] = f32::NAN;
                    } else {
                        let sum: f32 = valid_values.iter().sum();
                        stats[j] = sum / valid_values.len() as f32;
                    }
                },
                ImputationStrategy::Median => {
                    // Filter out NaN values, sort, and find median
                    let mut valid_values: Vec<f32> = column.iter()
                        .filter(|&&x| !x.is_nan())
                        .copied()
                        .collect();
                    
                    if valid_values.is_empty() {
                        stats[j] = f32::NAN;
                    } else {
                        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = valid_values.len() / 2;
                        stats[j] = if valid_values.len() % 2 == 0 {
                            (valid_values[mid - 1] + valid_values[mid]) / 2.0
                        } else {
                            valid_values[mid]
                        };
                    }
                },
                ImputationStrategy::MostFrequent => {
                    let mut value_counts: HashMap<u32, usize> = HashMap::new();
                    
                    for &val in column.iter() {
                        if !val.is_nan() {
                            // Use bit representation as key to handle floating point comparisons
                            let bits = val.to_bits();
                            *value_counts.entry(bits).or_insert(0) += 1;
                        }
                    }
                    
                    if value_counts.is_empty() {
                        stats[j] = f32::NAN;
                    } else {
                        // Find most frequent value
                        let (&bits, _) = value_counts.iter()
                            .max_by_key(|&(_, count)| *count)
                            .ok_or(PreprocessingError::MostFrequent)?;
                        
                        stats[j] = f32::from_bits(bits);
                    }
                },
                ImputationStrategy::Constant(value) => {
                    stats[j] = *value;
                },
            }
        }
        
        self.statistics = Some(stats);
        Ok(self)
    }
    
    /// Imputes missing values using the statistics learned during fit.
    ///
    /// Replaces missing values (NaNs) in the input data with the corresponding values 
    /// computed during the fitting phase.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data with missing values to transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f32>, PreprocessingError>` - The transformed data with imputed values on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::NotFitted` - If the imputer has not been fitted
    /// * `PreprocessingError::FeatureMismatch` - If the number of features in `x` differs from what was used during fitting
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0]];
    /// imputer.fit(&train_data.view()).unwrap();
    ///
    /// let test_data = array![[f32::NAN, 5.0], [6.0, f32::NAN]];
    /// let transformed = imputer.transform(&test_data.view()).unwrap();
    /// ```
    fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        let stats = self.statistics.as_ref()
            .ok_or(PreprocessingError::NotFitted)?;
        
        if x.ncols() != stats.len() {
            return Err(PreprocessingError::FeatureMismatch(x.ncols(), stats.len()));
        }
        
        let mut result = x.to_owned();        
        // Replace NaN values with the corresponding statistic
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = stats[j];
                }
            }
        }
        
        Ok(result)
    }
    
    /// Fits the imputer and transforms the input data in one operation.
    ///
    /// This is a convenience method that combines `fit` and `transform` into a single operation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The transformed data with imputed values on success, or an error
    ///
    /// # Errors
    ///
    /// May return any errors that could be returned by `fit` or `transform`
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// let mut imputer = SimpleImputer::new(ImputationStrategy::Median);
    /// let data = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
    /// let result = imputer.fit_transform(&data.view()).unwrap();
    /// ```
    fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.fit(x)?;
        self.transform(&x.view())
    }

    
}

/// Implementation of the `Transformer` trait for `SimpleImputer`.
///
/// This allows the `SimpleImputer` to be used with the general transformer interface.
impl Transformer for SimpleImputer {
    /// Fits the imputer on the input data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit on
    ///
    /// # Returns
    ///
    /// `Result<(), PreprocessingError>` - Success or an error
    fn fit(&mut self, x: &ArrayView2<f32>) -> Result<(), PreprocessingError> {
        self.fit(x)?;
        Ok(())
    }

    /// Transforms the input data by imputing missing values.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform
    ///
    /// # Returns
    ///
    /// `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.transform(x)
    }

    /// Fits the imputer and transforms the input data in one operation.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        self.fit_transform(x)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;

    #[test]
    fn test_simple_imputer_new() {
        let imputer = SimpleImputer::new(ImputationStrategy::Mean);
        assert_eq!(imputer.strategy, ImputationStrategy::Mean);
        assert!(imputer.statistics.is_none());
        
        let imputer = SimpleImputer::new(ImputationStrategy::Constant(42.0));
        assert!(matches!(imputer.strategy, ImputationStrategy::Constant(42.0)));
    }

    #[test]
    fn test_simple_imputer_fit_mean() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Array with missing values
        let x = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
        
        imputer.fit(&x.view()).unwrap();
        
        // Check learned statistics
        let stats = imputer.statistics.as_ref().unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 2.5).abs() < 1e-10); // Mean of [1.0, 4.0]
        assert!((stats[1] - 2.5).abs() < 1e-10); // Mean of [2.0, 3.0]
    }

    #[test]
    fn test_simple_imputer_fit_median() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Median);
        
        // Array with missing values
        let x = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN], [5.0, 8.0]];
        
        imputer.fit(&x.view()).unwrap();
        
        // Check learned statistics
        let stats = imputer.statistics.as_ref().unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 4.0).abs() < 1e-10);  // Median of [1.0, 4.0, 5.0]
        assert!((stats[1] - 3.0).abs() < 1e-10);  // Median of [2.0, 3.0, 8.0]
    }

    #[test]
    fn test_simple_imputer_fit_most_frequent() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::MostFrequent);
        
        // Array with missing values and duplicates
        let x = array![
            [1.0, 2.0], 
            [f32::NAN, 3.0], 
            [2.0, f32::NAN], 
            [2.0, 3.0], 
            [5.0, 3.0]
        ];
        
        imputer.fit(&x.view()).unwrap();
        
        // Check learned statistics
        let stats = imputer.statistics.as_ref().unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 2.0).abs() < 1e-10);  // Most frequent in first column is 2.0
        assert!((stats[1] - 3.0).abs() < 1e-10);  // Most frequent in second column is 3.0
    }

    #[test]
    fn test_simple_imputer_fit_constant() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Constant(99.0));
        
        // Array with missing values
        let x = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
        
        imputer.fit(&x.view()).unwrap();
        
        // Check learned statistics
        let stats = imputer.statistics.as_ref().unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 99.0).abs() < 1e-10);
        assert!((stats[1] - 99.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_imputer_transform() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Fit with some data
        let x_train = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
        imputer.fit(&x_train.view()).unwrap();
        
        // Transform with missing values
        let x_test = array![[f32::NAN, 1.0], [3.0, f32::NAN]];
        let result = imputer.transform(&x_test.view()).unwrap();
        
        // Check transformed values
        assert!((result[[0, 0]] - 2.5).abs() < 1e-10);  // NaN replaced with mean 2.5
        assert!((result[[0, 1]] - 1.0).abs() < 1e-10);  // Not replaced
        assert!((result[[1, 0]] - 3.0).abs() < 1e-10);  // Not replaced
        assert!((result[[1, 1]] - 2.5).abs() < 1e-10);  // NaN replaced with mean 2.5
    }

    #[test]
    fn test_simple_imputer_fit_transform() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Array with missing values
        let x = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
        
        // Apply fit_transform
        let result = imputer.fit_transform(&x.view()).unwrap();
        
        // Check transformed values
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);  // Not replaced
        assert!((result[[0, 1]] - 2.0).abs() < 1e-10);  // Not replaced
        assert!((result[[1, 0]] - 2.5).abs() < 1e-10);  // NaN replaced with mean 2.5
        assert!((result[[1, 1]] - 3.0).abs() < 1e-10);  // Not replaced
        assert!((result[[2, 0]] - 4.0).abs() < 1e-10);  // Not replaced
        assert!((result[[2, 1]] - 2.5).abs() < 1e-10);  // NaN replaced with mean 2.5
    }

    #[test]
    fn test_simple_imputer_not_fitted() {
        let imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Try to transform without fitting
        let x = array![[1.0, 2.0], [f32::NAN, 3.0]];
        let result = imputer.transform(&x.view());
        
        // Should fail with NotFitted error
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_simple_imputer_feature_mismatch() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Fit with 2 features
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        imputer.fit(&x_train.view()).unwrap();
        
        // Transform with 3 features
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = imputer.transform(&x_test.view());
        
        // Should fail with FeatureMismatch error
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch(3, 2))
        ));
    }

    #[test]
    fn test_simple_imputer_all_nan_column() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Column with all NaN values
        let x = array![[1.0, f32::NAN], [2.0, f32::NAN], [3.0, f32::NAN]];
        
        imputer.fit(&x.view()).unwrap();
        
        // Check statistics - second value should be NaN
        let stats = imputer.statistics.as_ref().unwrap();
        assert!(!stats[0].is_nan());
        assert!(stats[1].is_nan());
        
        // Transform should keep NaN for the all-NaN column
        let result = imputer.transform(&x.view()).unwrap();
        
        assert!(!result[[0, 0]].is_nan());
        assert!(result[[0, 1]].is_nan());
    }
}
