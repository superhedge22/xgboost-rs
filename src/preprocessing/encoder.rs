use ndarray::s;
use serde_json::{json, Value};
use std::any::Any;

use crate::{error::PreprocessingError, parameters::preprocessing::HandleUnknown};
use crate::types::{Array1, Array2, ArrayView2};

use super::Transformer;

/// One-hot encodes categorical features.
///
/// This transformer encodes categorical features as a one-hot numeric array,
/// where each category is represented as a binary feature.
///
/// # Fields
/// * `handle_unknown` - Strategy for handling unknown categories during transform
/// * `categories` - Learned categories for each feature (available after fitting)
/// * `feature_names_in` - Optional names of the input features
/// 

#[derive(Debug, Clone, PartialEq)]
pub struct OneHotEncoder {
    handle_unknown: HandleUnknown,
    categories: Option<Vec<Array1<f64>>>,
    feature_names_in: Option<Vec<String>>,
}

impl OneHotEncoder {
    /// Creates a new OneHotEncoder with the specified strategy for handling unknown categories.
    ///
    /// # Arguments
    /// * `handle_unknown` - Strategy to use for handling unknown categories:
    ///    - `HandleUnknown::Error` - Raise an error when unknown categories are encountered
    ///    - `HandleUnknown::Ignore` - Ignore unknown categories (all zeros will be encoded)
    ///
    /// # Returns
    /// A new instance of OneHotEncoder
    ///
    /// # Errors
    /// Panics if an invalid handle_unknown strategy is provided
    pub fn new(handle_unknown: HandleUnknown) -> Self {
        OneHotEncoder {
            handle_unknown,
            categories: None,
            feature_names_in: None,
        }
    }
    
    /// Learns the categories for one-hot encoding.
    ///
    /// # Arguments
    /// * `x` - Input data array where each column is a feature to encode
    /// * `feature_names` - Optional names for input features
    ///
    /// # Returns
    /// * `Ok(&mut Self)` - Reference to self for method chaining
    /// * `Err(PreprocessingError)` - If the fitting process fails
    pub fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>, feature_names: Option<Vec<String>>) -> Result<&mut Self, PreprocessingError> {
        let n_features = x.ncols();
        let mut categories = Vec::with_capacity(n_features);
        
        for j in 0..n_features {
            let column = x.slice(s![.., j]);
            
            // Find unique categories
            let mut unique_cats: Vec<f64> = Vec::new();
            for &val in column.iter() {
                let val_f64: f64 = val.into();
                if !unique_cats.contains(&val_f64) {
                    unique_cats.push(val_f64);
                }
            }
            // Sort categories
            unique_cats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            categories.push(Array1::from(unique_cats));
        }
        
        self.categories = Some(categories);
        self.feature_names_in = feature_names;
        
        Ok(self)
    }
    
    /// Transforms input data using one-hot encoding based on learned categories.
    ///
    /// # Arguments
    /// * `x` - Input data array to transform
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - One-hot encoded data
    /// * `Err(PreprocessingError)` - If transformation fails, such as when:
    ///   - The encoder hasn't been fitted
    ///   - The number of features doesn't match what was learned during fit
    ///   - Unknown categories are encountered with `HandleUnknown::Error`
    pub fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        let categories = self.categories.as_ref()
            .ok_or(PreprocessingError::NotFitted)?;
        
        if x.ncols() != categories.len() {
            return Err(PreprocessingError::FeatureMismatch(x.ncols(), categories.len()));
        }
        
        // Determine output shape
        let n_samples = x.nrows();
        let n_features_out = categories.iter().map(|cats| cats.len()).sum();
        
        let mut result = Array2::zeros((n_samples, n_features_out));
        let mut col_idx = 0;
        for j in 0..x.ncols() {
            let cats = &categories[j];
            
            for i in 0..n_samples {
                let val = x[[i, j]];
                let cat_idx = cats.iter().position(|&c| c == val.into());
                
                match cat_idx {
                    Some(idx) => {
                        result[[i, col_idx + idx]] = 1.0;
                    },
                    None => {
                        // Handle unknown category
                        match self.handle_unknown {
                            HandleUnknown::Error => {
                                return Err(PreprocessingError::UnknownCategory(val.into(), j));
                            },
                            HandleUnknown::Ignore => {
                                // leave as 0 (already done by initializing with zeros)
                            },
                        }
                    },
                }
            }
            
            col_idx += cats.len();
        }
        
        Ok(result)
    }
    
    /// Fits to data, then transforms it in one step.
    ///
    /// # Arguments
    /// * `x` - Input data array to fit and transform
    /// * `feature_names` - Optional names for input features
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - One-hot encoded data
    /// * `Err(PreprocessingError)` - If fitting or transformation fails
    pub fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>, feature_names: Option<Vec<String>>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(x, feature_names)?;
        self.transform(&x.view())
    }
    
    /// Returns the names of the transformed features.
    ///
    /// Feature names are constructed from the input feature names (if provided)
    /// or automatically generated names (`x0`, `x1`, etc.) combined with the
    /// category values.
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - Feature names in the format `{feature_name}_{category}`
    /// * `Err(PreprocessingError)` - If the encoder hasn't been fitted
    pub fn get_feature_names_out(&self) -> Result<Vec<String>, PreprocessingError> {
        let categories = self.categories.as_ref()
            .ok_or(PreprocessingError::NotFitted)?;
        
        let mut result = Vec::new();
        
        for (j, cats) in categories.iter().enumerate() {
            let feature_name = if let Some(names) = &self.feature_names_in {
                names.get(j).map(|s| s.clone()).unwrap_or_else(|| format!("x{}", j))
            } else {
                format!("x{}", j)
            };
            
            for &cat in cats.iter() {
                result.push(format!("{}_{}", feature_name, cat));
            }
        }
        
        Ok(result)
    }
    
    /// Converts the OneHotEncoder to a JSON representation.
    ///
    /// This method serializes the complete encoder state to JSON, including:
    /// - The initialization parameters (handle_unknown)
    /// - The learned categories for each feature
    /// - The feature names, if available
    ///
    /// # Returns
    /// A JSON Value containing the serialized encoder
    pub fn to_json(&self) -> Value {
        // Convert categories to Vec<Vec<f64>> for serialization
        let categories_json = if let Some(categories) = &self.categories {
            // Convert each Array1 to a Vec<f64>
            categories.iter()
                .map(|cat_array| cat_array.iter().map(|&c| c.into()).collect::<Vec<f64>>())
                .collect::<Vec<Vec<f64>>>()
        } else {
            Vec::new()
        };

        // Convert handle_unknown to string
        let handle_unknown_str = match self.handle_unknown {
            HandleUnknown::Error => "error",
            HandleUnknown::Ignore => "ignore",
        };

        // Create the JSON structure
        json!({
            "type": "OneHotEncoder",
            "init_params": {
                "handle_unknown": handle_unknown_str
            },
            "attrs": {
                "categories": categories_json,
                "feature_names_in": self.feature_names_in.clone()
            }
        })
    }

    /// Creates a OneHotEncoder from its JSON representation.
    ///
    /// # Arguments
    /// * `json_data` - The JSON Value containing the serialized encoder state
    ///
    /// # Returns
    /// * `Result<Self, PreprocessingError>` - A new OneHotEncoder instance with restored state,
    ///   or an error if the JSON data is invalid
    pub fn from_json(json_data: Value) -> Result<Self, PreprocessingError> {
        // Parse initialization parameters
        let init_params = json_data.get("init_params")
            .ok_or_else(|| PreprocessingError::NotFitted)?;
            
        // Parse handle_unknown
        let handle_unknown_str = init_params.get("handle_unknown")
            .and_then(|v| v.as_str())
            .unwrap_or("error");
            
        let handle_unknown = match handle_unknown_str {
            "error" => HandleUnknown::Error,
            "ignore" => HandleUnknown::Ignore,
            _ => return Err(PreprocessingError::NotFitted), // Invalid handle_unknown
        };
        
        // Create encoder with init params
        let mut encoder = OneHotEncoder::new(handle_unknown);
        
        // Parse attributes
        if let Some(attrs) = json_data.get("attrs") {
            // Parse categories
            if let Some(cat_array) = attrs.get("categories") {
                if let Some(cat_array) = cat_array.as_array() {
                    let mut categories = Vec::with_capacity(cat_array.len());
                    
                    for cat in cat_array {
                        if let Some(values) = cat.as_array() {
                            let array_values: Vec<f64> = values.iter()
                                .filter_map(|v| v.as_f64())
                                .collect();
                                
                            categories.push(Array1::from(array_values));
                        }
                    }
                    
                    if !categories.is_empty() {
                        encoder.categories = Some(categories);
                    }
                }
            }
            
            // Parse feature names
            if let Some(names) = attrs.get("feature_names_in") {
                if let Some(names_array) = names.as_array() {
                    let feature_names: Vec<String> = names_array.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                        
                    if !feature_names.is_empty() {
                        encoder.feature_names_in = Some(feature_names);
                    }
                }
            }
        }
        
        Ok(encoder)
    }
}

/// Implementation of the Transformer trait for OneHotEncoder.
///
/// This allows OneHotEncoder to be used with the same interface as other transformers.
impl Transformer for OneHotEncoder {
    /// Fits the encoder to the input data.
    ///
    /// # Arguments
    /// * `x` - Input data array
    ///
    /// # Returns
    /// * `Ok(())` - On success
    /// * `Err(PreprocessingError)` - If fitting fails
    fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<(), PreprocessingError> {
        self.fit(x, None)?;
        Ok(())
    }
    
    
    /// Transforms input data using one-hot encoding.
    ///
    /// # Arguments
    /// * `x` - Input data array
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - Transformed data
    /// * `Err(PreprocessingError)` - If transformation fails
    fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        self.transform(x)
    }

    /// Fits the encoder to data, then transforms it.
    ///
    /// # Arguments
    /// * `x` - Input data array
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - Transformed data
    /// * `Err(PreprocessingError)` - If fitting or transformation fails
    fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit_transform(x, None)
    }
    
    /// Returns the JSON representation of the OneHotEncoder.
    ///
    /// # Returns
    ///
    /// `Option<Value>` - JSON representation of the encoder
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
    use super::*;
    use ndarray::array;

    #[test]
    fn test_one_hot_encoder_new() {
        let encoder = OneHotEncoder::new(HandleUnknown::Error);
        assert!(encoder.categories.is_none());
        assert!(encoder.feature_names_in.is_none());
        assert_eq!(encoder.handle_unknown, HandleUnknown::Error);
        
        let encoder = OneHotEncoder::new(HandleUnknown::Ignore);
        assert_eq!(encoder.handle_unknown, HandleUnknown::Ignore);
    }

    #[test]
    fn test_one_hot_encoder_fit() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Simple dataset with two features
        let x = array![[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
        
        encoder.fit(&x.view(), None).unwrap();
        
        // Check that categories were learned correctly
        let categories = encoder.categories.as_ref().unwrap();
        assert_eq!(categories.len(), 2);
        assert_eq!(categories[0], array![0.0, 1.0, 2.0]);
        assert_eq!(categories[1], array![1.0, 2.0]);
    }

    #[test]
    fn test_one_hot_encoder_fit_with_feature_names() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Simple dataset with two features
        let x = array![[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        
        encoder.fit(&x.view(), Some(feature_names.clone())).unwrap();
        
        // Check that feature names were stored
        assert_eq!(encoder.feature_names_in.as_ref().unwrap(), &feature_names);
    }

    #[test]
    fn test_one_hot_encoder_transform() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Simple dataset with two features
        let x = array![[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
        
        // Fit the encoder
        encoder.fit(&x.view(), None).unwrap();
        
        // Transform the same data
        let result = encoder.transform(&x.view()).unwrap();
        
        // Expected:
        // Feature 1 has categories [0, 1, 2]
        // Feature 2 has categories [1, 2]
        // So result should have 5 columns
        assert_eq!(result.shape(), &[3, 5]);
        
        // First row: [0.0, 1.0] should become [1, 0, 0, 1, 0]
        assert_eq!(result.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0]);
        
        // Second row: [1.0, 2.0] should become [0, 1, 0, 0, 1]
        assert_eq!(result.row(1).to_vec(), vec![0.0, 1.0, 0.0, 0.0, 1.0]);
        
        // Third row: [2.0, 1.0] should become [0, 0, 1, 1, 0]
        assert_eq!(result.row(2).to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_one_hot_encoder_unknown_category() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Fit with some categories
        let x_train = array![[0.0, 1.0], [1.0, 2.0]];
        encoder.fit(&x_train.view(), None).unwrap();
        
        // Transform with unknown category
        let x_test = array![[3.0, 1.0]];
        let result = encoder.transform(&x_test.view());
        
        // Should fail with UnknownCategory error
        assert!(matches!(
            result,
            Err(PreprocessingError::UnknownCategory(3.0, 0))
        ));
    }

    #[test]
    fn test_one_hot_encoder_ignore_unknown() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Ignore);
        
        // Fit with some categories
        let x_train = array![[0.0, 1.0], [1.0, 2.0]];
        encoder.fit(&x_train.view(), None).unwrap();
        
        // Transform with unknown category
        let x_test = array![[3.0, 1.0]];
        let result = encoder.transform(&x_test.view()).unwrap();
        
        // Should have zeros for unknown category
        assert_eq!(result.row(0).to_vec(), vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_one_hot_encoder_fit_transform() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Simple dataset
        let x = array![[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
        
        // Apply fit_transform
        let result = encoder.fit_transform(&x.view(), None).unwrap();
        
        // Check output shape and values (same as in transform test)
        assert_eq!(result.shape(), &[3, 5]);
        assert_eq!(result.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(result.row(1).to_vec(), vec![0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(result.row(2).to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_one_hot_encoder_get_feature_names_out() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Fit with feature names
        let x = array![[0.0, 1.0], [1.0, 2.0]];
        let feature_names = vec!["age".to_string(), "education".to_string()];
        
        encoder.fit(&x.view(), Some(feature_names)).unwrap();
        
        // Get feature names out
        let names = encoder.get_feature_names_out().unwrap();
        
        // Check names
        assert_eq!(names, vec!["age_0", "age_1", "education_1", "education_2"]);
    }

    #[test]
    fn test_one_hot_encoder_get_feature_names_out_without_names() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Fit without feature names
        let x = array![[0.0, 1.0], [1.0, 2.0]];
        encoder.fit(&x.view(), None).unwrap();
        
        // Get feature names out
        let names = encoder.get_feature_names_out().unwrap();
        
        // Should use default naming x{i}_{value}
        assert_eq!(names, vec!["x0_0", "x0_1", "x1_1", "x1_2"]);
    }

    #[test]
    fn test_one_hot_encoder_not_fitted() {
        let encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Try to transform without fitting
        let x = array![[0.0, 1.0]];
        let result = encoder.transform(&x.view());
        
        // Should fail with NotFitted error
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
        
        // Get feature names without fitting
        let result = encoder.get_feature_names_out();
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_one_hot_encoder_feature_mismatch() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Fit with 2 features
        let x_train = array![[0.0, 1.0], [1.0, 2.0]];
        encoder.fit(&x_train.view(), None).unwrap();
        
        // Transform with 3 features
        let x_test = array![[0.0, 1.0, 2.0]];
        let result = encoder.transform(&x_test.view());
        
        // Should fail with FeatureMismatch error
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch(3, 2))
        ));
    }

    #[test]
    fn test_one_hot_encoder_to_json() {
        let mut encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        let x = array![[0.0, 1.0], [1.0, 2.0]];
        let feature_names = vec!["age".to_string(), "education".to_string()];
        
        encoder.fit(&x.view(), Some(feature_names)).unwrap();
        
        let json_data = encoder.to_json();
        
        // Check that JSON contains the correct data
        assert_eq!(json_data["type"], "OneHotEncoder");
        assert_eq!(json_data["init_params"]["handle_unknown"], "error");
        
        // Check categories
        let categories = &json_data["attrs"]["categories"];
        assert_eq!(categories.as_array().unwrap().len(), 2);
        
        // Check feature names
        let names = &json_data["attrs"]["feature_names_in"];
        assert_eq!(names[0], "age");
        assert_eq!(names[1], "education");
    }

    #[test]
    fn test_one_hot_encoder_from_json() {
        // Create a JSON representation
        let json_data = json!({
            "type": "OneHotEncoder",
            "init_params": {
                "handle_unknown": "ignore"
            },
            "attrs": {
                "categories": [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0]
                ],
                "feature_names_in": ["feature1", "feature2"]
            }
        });
        
        // Create encoder from JSON
        let encoder = OneHotEncoder::from_json(json_data).unwrap();
        
        // Check initialization parameters
        assert_eq!(encoder.handle_unknown, HandleUnknown::Ignore);
        
        // Check categories
        let categories = encoder.categories.as_ref().unwrap();
        assert_eq!(categories.len(), 2);
        assert_eq!(categories[0], array![0.0, 1.0, 2.0]);
        assert_eq!(categories[1], array![1.0, 2.0]);
        
        // Check feature names
        let feature_names = encoder.feature_names_in.as_ref().unwrap();
        assert_eq!(feature_names, &vec!["feature1".to_string(), "feature2".to_string()]);
    }

    #[test]
    fn test_one_hot_encoder_serialization_roundtrip() {
        let mut original_encoder = OneHotEncoder::new(HandleUnknown::Error);
        
        // Fit with data
        let x = array![[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
        let feature_names = vec!["age".to_string(), "education".to_string()];
        original_encoder.fit(&x.view(), Some(feature_names)).unwrap();
        
        // Convert to JSON and back
        let json_data = original_encoder.to_json();
        let restored_encoder = OneHotEncoder::from_json(json_data).unwrap();
        
        // Check that restored encoder has the same parameters
        assert_eq!(restored_encoder.handle_unknown, original_encoder.handle_unknown);
        
        // Check categories (they should be the same, but we need to compare values, not references)
        let original_cats = original_encoder.categories.as_ref().unwrap();
        let restored_cats = restored_encoder.categories.as_ref().unwrap();
        assert_eq!(original_cats.len(), restored_cats.len());
        
        for (orig, restored) in original_cats.iter().zip(restored_cats.iter()) {
            assert_eq!(orig.len(), restored.len());
            for (a, b) in orig.iter().zip(restored.iter()) {
                assert_eq!(*a, *b);
            }
        }
        
        // Check feature names
        assert_eq!(
            original_encoder.feature_names_in.as_ref().unwrap(),
            restored_encoder.feature_names_in.as_ref().unwrap()
        );
    }
}