use std::collections::HashMap;
use std::any::Any;

use ndarray::s;
use serde_json::{json, Value};

use crate::{error::PreprocessingError, parameters::preprocessing::ImputationStrategy};
use crate::types::{Array1, Array2, ArrayView2};

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
/// let result = imputer.fit_transform(&data.view(), None).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleImputer {
    /// The imputation strategy to use for missing values
    strategy: ImputationStrategy,
    /// The imputation values calculated during fitting for each feature
    statistics: Option<Array1<f64>>,
    /// Optional names of the input features
    columns: Option<Vec<String>>,
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
    ///
    /// # Panics
    ///
    /// Panics if an invalid strategy is provided
    pub fn new(strategy: ImputationStrategy) -> Self {
        SimpleImputer {
            strategy,
            statistics: None,
            columns: None,
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
    /// * `feature_names` - Optional names for input features
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
    /// imputer.fit(&data.view(), None).unwrap();
    /// ```
    pub fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>, feature_names: Option<Vec<String>>) -> Result<&mut Self, PreprocessingError> {
        let n_features = x.ncols();
        let mut stats = Array1::zeros(n_features);
        
        for j in 0..n_features {
            let column = x.slice(s![.., j]);
            
            match &self.strategy {
                ImputationStrategy::Mean => {
                    // Filter out NaN values and compute mean
                    let valid_values: Vec<f64> = column.iter()
                        .filter(|&&x| !x.into().is_nan())
                        .map(|&x| x.into())
                        .collect();
                    
                    if valid_values.is_empty() {
                        stats[j] = f64::NAN;
                    } else {
                        let sum: f64 = valid_values.iter().sum();
                        stats[j] = sum / valid_values.len() as f64;
                    }
                },
                ImputationStrategy::Median => {
                    // Filter out NaN values, sort, and find median
                    let mut valid_values: Vec<f64> = column.iter()
                        .filter(|&&x| !x.into().is_nan())
                        .map(|&x| x.into())
                        .collect();
                    
                    if valid_values.is_empty() {
                        stats[j] = f64::NAN;
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
                    let mut value_counts: HashMap<u64, usize> = HashMap::new();
                    
                    for &val in column.iter() {
                        if !val.into().is_nan() {
                            // Use bit representation as key to handle floating point comparisons
                            let bits = val.into().to_bits();
                            *value_counts.entry(bits).or_insert(0) += 1;
                        }
                    }
                    
                    if value_counts.is_empty() {
                        stats[j] = f64::NAN;
                    } else {
                        // Find most frequent value
                        let (&bits, _) = value_counts.iter()
                            .max_by_key(|&(_, count)| *count)
                            .ok_or(PreprocessingError::MostFrequent)?;
                        
                        stats[j] = f64::from_bits(bits);
                    }
                },
                ImputationStrategy::Constant(value) => {
                    stats[j] = *value;
                },
            }
        }
        
        self.statistics = Some(stats);
        self.columns = feature_names;
        
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
    /// imputer.fit(&train_data.view(), None).unwrap();
    ///
    /// let test_data = array![[f32::NAN, 5.0], [6.0, f32::NAN]];
    /// let transformed = imputer.transform(&test_data.view()).unwrap();
    /// ```
    pub fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        let stats = self.statistics.as_ref()
            .ok_or(PreprocessingError::NotFitted)?;
        
        if x.ncols() != stats.len() {
            return Err(PreprocessingError::FeatureMismatch(x.ncols(), stats.len()));
        }
        
        let mut result = x.mapv(|x| x.into());
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
    /// * `feature_names` - Optional names for input features
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f32>, PreprocessingError>` - The transformed data with imputed values on success, or an error
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
    /// let result = imputer.fit_transform(&data.view(), None).unwrap();
    /// ```
    pub fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>, feature_names: Option<Vec<String>>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(x, feature_names)?;
        self.transform(&x.view())
    }

    /// Converts the SimpleImputer to a JSON representation.
    ///
    /// This method serializes the complete imputer state to JSON, including:
    /// - The initialization parameters (strategy)
    /// - The learned statistics for each feature
    /// - The feature names, if available
    ///
    /// # Returns
    /// A JSON Value containing the serialized imputer
    pub fn to_json(&self) -> Value {
        // Convert statistics to Vec<f64> for serialization
        let statistics_json = if let Some(stats) = &self.statistics {
            stats.iter().map(|&s| s).collect::<Vec<f64>>()
        } else {
            Vec::new()
        };

        // Convert strategy to string and fill_value
        let (strategy_str, fill_value) = match self.strategy {
            ImputationStrategy::Mean => ("mean", None),
            ImputationStrategy::Median => ("median", None),
            ImputationStrategy::MostFrequent => ("most_frequent", None),
            ImputationStrategy::Constant(val) => ("constant", Some(val)),
        };

        // Create the init_params object
        let mut init_params = json!({
            "strategy": strategy_str
        });

        // Add fill_value if present
        if let Some(val) = fill_value {
            init_params["fill_value"] = json!(val);
        }

        // Create the JSON structure
        json!({
            "type": "SimpleImputer",
            "init_params": init_params,
            "attrs": {
                "statistics_": statistics_json,
                "columns": self.columns.clone()
            }
        })
    }

    /// Creates a SimpleImputer from its JSON representation.
    ///
    /// # Arguments
    /// * `json_data` - The JSON Value containing the serialized imputer state
    ///
    /// # Returns
    /// * `Result<Self, PreprocessingError>` - A new SimpleImputer instance with restored state,
    ///   or an error if the JSON data is invalid
    pub fn from_json(json_data: Value) -> Result<Self, PreprocessingError> {
        // Parse initialization parameters
        let init_params = json_data.get("init_params")
            .ok_or_else(|| PreprocessingError::NotFitted)?;
            
        // Parse strategy
        let strategy_str = init_params.get("strategy")
            .and_then(|v| v.as_str())
            .ok_or(PreprocessingError::NotFitted)?;

        // Parse fill_value if strategy is 'constant'
        let strategy = match strategy_str {
            "mean" => ImputationStrategy::Mean,
            "median" => ImputationStrategy::Median,
            "most_frequent" => ImputationStrategy::MostFrequent,
            "constant" => {
                let fill_value = init_params.get("fill_value")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                ImputationStrategy::Constant(fill_value)
            },
            _ => return Err(PreprocessingError::NotFitted), // Invalid strategy
        };
        
        // Create imputer with init params
        let mut imputer = SimpleImputer::new(strategy);
        
        // Parse attributes
        if let Some(attrs) = json_data.get("attrs") {
            // Parse statistics
            if let Some(stats_array) = attrs.get("statistics_") {
                if let Some(values) = stats_array.as_array() {
                    let statistics: Vec<f64> = values.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                        
                    if !statistics.is_empty() {
                        imputer.statistics = Some(Array1::from(statistics));
                    }
                }
            }
            
            // Parse column names
            if let Some(columns) = attrs.get("columns") {
                if let Some(columns_array) = columns.as_array() {
                    let feature_names: Vec<String> = columns_array.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                        
                    if !feature_names.is_empty() {
                        imputer.columns = Some(feature_names);
                    }
                }
            }
        }
        
        Ok(imputer)
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
    fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<(), PreprocessingError> {
        self.fit(x, None)?;
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
    /// `Result<Array2<f32>, PreprocessingError>` - The transformed data or an error
    fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
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
    /// `Result<Array2<f32>, PreprocessingError>` - The transformed data or an error
    fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit_transform(x, None)
    }
    
    /// Returns the JSON representation of the SimpleImputer.
    ///
    /// # Returns
    ///
    /// `Option<Value>` - JSON representation of the imputer
    fn to_json_opt(&self) -> Option<Value> {
        Some(self.to_json())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
    fn test_simple_imputer_invalid_strategy_json() {
        let json_data = json!({
            "type": "SimpleImputer",
            "init_params": {
                "strategy": "invalid_strategy"
            }
        });
        
        // This should return an error
        let result = SimpleImputer::from_json(json_data);
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_simple_imputer_fit_mean() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Array with missing values
        let x = array![[1.0, 2.0], [f32::NAN, 3.0], [4.0, f32::NAN]];
        
        imputer.fit(&x.view(), None).unwrap();
        
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
        
        imputer.fit(&x.view(), None).unwrap();
        
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
        
        imputer.fit(&x.view(), None).unwrap();
        
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
        
        imputer.fit(&x.view(), None).unwrap();
        
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
        imputer.fit(&x_train.view(), None).unwrap();
        
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
        let result = imputer.fit_transform(&x.view(), None).unwrap();
        
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
        imputer.fit(&x_train.view(), None).unwrap();
        
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
        
        imputer.fit(&x.view(), None).unwrap();
        
        // Check statistics - second value should be NaN
        let stats = imputer.statistics.as_ref().unwrap();
        assert!(!stats[0].is_nan());
        assert!(stats[1].is_nan());
        
        // Transform should keep NaN for the all-NaN column
        let result = imputer.transform(&x.view()).unwrap();
        
        assert!(!result[[0, 0]].is_nan());
        assert!(result[[0, 1]].is_nan());
    }

    #[test]
    fn test_simple_imputer_to_json() {
        let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
        
        // Fit with data and feature names
        let x = array![[1.0, 2.0], [3.0, 4.0], [f32::NAN, f32::NAN]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        imputer.fit(&x.view(), Some(feature_names)).unwrap();
        
        let json_data = imputer.to_json();
        
        // Check that JSON contains the correct data
        assert_eq!(json_data["type"], "SimpleImputer");
        assert_eq!(json_data["init_params"]["strategy"], "mean");
        
        // Check statistics
        let statistics = &json_data["attrs"]["statistics_"];
        assert_eq!(statistics.as_array().unwrap().len(), 2);
        
        // Check feature names
        let columns = &json_data["attrs"]["columns"];
        assert_eq!(columns[0], "feature1");
        assert_eq!(columns[1], "feature2");
    }

    #[test]
    fn test_simple_imputer_from_json() {
        // Create a JSON representation with constant strategy
        let json_data = json!({
            "type": "SimpleImputer",
            "init_params": {
                "strategy": "constant",
                "fill_value": 99.5
            },
            "attrs": {
                "statistics_": [99.5, 99.5],
                "columns": ["feature1", "feature2"]
            }
        });
        
        // Create imputer from JSON
        let imputer = SimpleImputer::from_json(json_data).unwrap();
        
        // Check initialization parameters
        match imputer.strategy {
            ImputationStrategy::Constant(val) => assert_eq!(val, 99.5),
            _ => panic!("Expected Constant strategy"),
        }
        
        // Check statistics
        let statistics = imputer.statistics.as_ref().unwrap();
        assert_eq!(statistics.len(), 2);
        assert_eq!(statistics[0], 99.5);
        assert_eq!(statistics[1], 99.5);
        
        // Check feature names
        let columns = imputer.columns.as_ref().unwrap();
        assert_eq!(columns, &vec!["feature1".to_string(), "feature2".to_string()]);
    }

    #[test]
    fn test_simple_imputer_serialization_roundtrip() {
        let mut original_imputer = SimpleImputer::new(ImputationStrategy::Median);
        
        // Fit with data
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let feature_names = vec!["age".to_string(), "weight".to_string()];
        original_imputer.fit(&x.view(), Some(feature_names)).unwrap();
        
        // Convert to JSON and back
        let json_data = original_imputer.to_json();
        let restored_imputer = SimpleImputer::from_json(json_data).unwrap();
        
        // Check that restored imputer has the same parameters
        assert!(matches!(restored_imputer.strategy, ImputationStrategy::Median));
        
        // Check statistics (they should be the same)
        let original_stats = original_imputer.statistics.as_ref().unwrap();
        let restored_stats = restored_imputer.statistics.as_ref().unwrap();
        assert_eq!(original_stats.len(), restored_stats.len());
        
        for (orig, restored) in original_stats.iter().zip(restored_stats.iter()) {
            assert!((orig - restored).abs() < 1e-10);
        }
        
        // Check feature names
        assert_eq!(
            original_imputer.columns.as_ref().unwrap(),
            restored_imputer.columns.as_ref().unwrap()
        );
    }
}
