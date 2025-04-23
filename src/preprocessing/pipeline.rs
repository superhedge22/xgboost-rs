use std::collections::HashMap;
use serde_json::{json, Value};
use std::any::Any;

use crate::{error::PreprocessingError, Predict, XGBError};
use crate::types::{Array2, Array2F, ArrayView2, ArrayView2F};

use super::{Transformer, TransformerType};
use super::scaler::StandardScaler;
use super::imputer::SimpleImputer;
use super::encoder::OneHotEncoder;
use super::transform::ColumnTransformer;


/// Pipeline chains multiple transformers into a single transformation pipeline.
///
/// This structure allows combining several preprocessing steps sequentially into a unified workflow.
/// Each step in the pipeline transforms the data and passes it to the next step. The pipeline
/// can also include an optional final predictor for performing predictions on the transformed data.
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use xgboostrs::preprocessing::pipeline::Pipeline;
/// use xgboostrs::preprocessing::imputer::SimpleImputer;
/// use xgboostrs::preprocessing::scaler::StandardScaler;
/// use xgboostrs::parameters::preprocessing::ImputationStrategy;
/// use xgboostrs::booster::Booster;
/// use xgboostrs::preprocessing::{Transformer, TransformerType};
///
/// // Create preprocessing steps
/// let imputer = SimpleImputer::new(ImputationStrategy::Mean);
/// let scaler = StandardScaler::new();
///
/// // Create a pipeline with multiple transformers
/// let steps = vec![
///     ("imputer".to_string(), TransformerType::Imputer(imputer)),
///     ("scaler".to_string(), TransformerType::StandardScaler(scaler)),
/// ];
///
/// let mut pipeline = Pipeline::new(steps);
///
/// // Sample data for fitting
/// let data = array![[1.0, f32::NAN], [3.0, 2.0], [f32::NAN, 5.0]];
///
/// // Fit and transform the data
/// let transformed = pipeline.fit_transform(&data.view()).unwrap();
/// ```
pub struct Pipeline {
    /// Ordered sequence of named transformers that make up the pipeline
    steps: Vec<(String, TransformerType)>,
    /// Map of step names to transformers for easier access
    named_steps: HashMap<String, usize>,
    /// Optional predictor to use after all transformations
    predict: Option<Box<dyn Predict>>,
    /// Whether the pipeline has been fitted
    fitted: bool,
}

impl Pipeline {
    /// Creates a new Pipeline with the specified transformation steps.
    ///
    /// # Parameters
    ///
    /// * `steps` - A vector of (name, transformer) pairs that define the pipeline sequence
    ///
    /// # Returns
    ///
    /// A new unfitted Pipeline with the provided steps.
    ///
    /// # Examples
    ///
    /// ```
    /// use xgboostrs::preprocessing::pipeline::Pipeline;
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::{Transformer, TransformerType};
    ///
    /// // Create transformers
    /// let imputer = SimpleImputer::new(ImputationStrategy::Mean);
    /// let scaler = StandardScaler::new();
    ///
    /// // Build the pipeline
    /// let steps = vec![
    ///     ("imputer".to_string(), TransformerType::Imputer(imputer)),
    ///     ("scaler".to_string(), TransformerType::StandardScaler(scaler)),
    /// ];
    ///
    /// let pipeline = Pipeline::new(steps);
    /// ```
    pub fn new(steps: Vec<(String, TransformerType)>) -> Self {
        // Create a mapping of step names to their indices
        let named_steps = steps.iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();
        
        Pipeline {
            steps,
            named_steps,
            predict: None,
            fitted: false,
        }
    }

    /// Creates a new Pipeline with the specified transformation steps and predictor.
    ///
    /// # Parameters
    ///
    /// * `steps` - A vector of (name, transformer) pairs that define the pipeline sequence
    /// * `predictor` - The predictor to use after all transformations
    ///
    /// # Returns
    ///
    /// A new Pipeline with the provided steps and predictor.
    pub fn new_with_predictor(steps: Vec<(String, TransformerType)>, predictor: Box<dyn Predict>) -> Self {
        // Create a mapping of step names to their indices
        let named_steps = steps.iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();
            
        Pipeline {
            steps,
            named_steps,
            predict: Some(predictor),
            fitted: false,
        }
    }
    
    /// Fits all transformers in the pipeline on the input data.
    ///
    /// This method fits each transformer in sequence, passing the transformed data
    /// from one step to the next. Each transformer, except the last one, first fits
    /// and then transforms the data. The last transformer is only fitted.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit the pipeline on
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, PreprocessingError>` - The fitted pipeline on success, or an error
    ///
    /// # Errors
    ///
    /// * `PreprocessingError::NoSteps` - If the pipeline has no steps
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::pipeline::Pipeline;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::{Transformer, TransformerType};
    ///
    /// // Create a pipeline with a single scaler
    /// let steps = vec![
    ///     ("scaler".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
    /// ];
    ///
    /// let mut pipeline = Pipeline::new(steps);
    ///
    /// // Sample data
    /// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    ///
    /// // Fit the pipeline
    /// pipeline.fit(&data.view()).unwrap();
    /// ```
    pub fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<&mut Self, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::NoSteps);
        }
        
        let mut x_transformed = x.mapv(Into::into);
        
        let steps_len = self.steps.len();
        // Fit and transform each step except the last one
        for (_, transform) in &mut self.steps[..steps_len - 1] {
            let x_view = x_transformed.view();
            x_transformed = transform.fit_transform(&x_view)?;
        }
        
        // Fit the last step
        let (_, last_transform) = self.steps.last_mut().unwrap();
        let x_view = x_transformed.view();
        last_transform.fit(&x_view)?;
        
        self.fitted = true;
        Ok(self)
    }
    
    /// Applies all transformations in the pipeline to the input data.
    ///
    /// This method sequentially applies each transformer in the pipeline to the input data,
    /// passing the transformed output from one step as input to the next step.
    /// It skips any steps with names "classifier" or "regressor".
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
    /// * `PreprocessingError::NotFitted` - If the pipeline has not been fitted
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::pipeline::Pipeline;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::preprocessing::{Transformer, TransformerType};
    ///
    /// // Create and fit a pipeline
    /// let steps = vec![
    ///     ("scaler".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
    /// ];
    ///
    /// let mut pipeline = Pipeline::new(steps);
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// pipeline.fit(&train_data.view()).unwrap();
    ///
    /// // Transform new data
    /// let test_data = array![[2.0, 3.0], [4.0, 5.0]];
    /// let transformed = pipeline.transform(&test_data.view()).unwrap();
    /// ```
    pub fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        
        let mut x_transformed = x.mapv(Into::into);
        
        // Apply each transform in the pipeline, skipping classifier/regressor steps
        for (name, transform) in &self.steps {
            if name == "classifier" || name == "regressor" {
                continue;
            }
            let x_view = x_transformed.view();
            x_transformed = transform.transform(&x_view)?;
        }
        
        Ok(x_transformed)
    }
    
    /// Fits the pipeline to the data and then transforms it.
    ///
    /// This is a convenience method that combines `fit` and `transform` into a single operation.
    /// It fits each transformer in sequence and transforms the data with all steps,
    /// except those named "classifier" or "regressor".
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
    /// * `PreprocessingError::NoSteps` - If the pipeline has no steps
    /// * Other errors that might be returned by individual transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::pipeline::Pipeline;
    /// use xgboostrs::preprocessing::imputer::SimpleImputer;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::parameters::preprocessing::ImputationStrategy;
    /// use xgboostrs::preprocessing::{Transformer, TransformerType};
    ///
    /// // Create a pipeline with multiple steps
    /// let steps = vec![
    ///     ("imputer".to_string(), TransformerType::Imputer(SimpleImputer::new(ImputationStrategy::Mean))),
    ///     ("scaler".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
    /// ];
    ///
    /// let mut pipeline = Pipeline::new(steps);
    ///
    /// // Sample data with missing values
    /// let data = array![[1.0, 2.0], [f32::NAN, 4.0], [5.0, f32::NAN]];
    ///
    /// // Fit and transform in one step
    /// let transformed = pipeline.fit_transform(&data.view()).unwrap();
    /// ```
    pub fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::NoSteps);
        }
        
        let mut x_transformed = x.mapv(Into::into);
        
        // Apply fit_transform for each step, skipping classifier/regressor steps
        for (name, transform) in &mut self.steps {
            if name == "classifier" || name == "regressor" {
                // Just fit classifier/regressor steps but don't transform with them
                let x_view = x_transformed.view();
                transform.fit(&x_view)?;
                continue;
            }
            
            let x_view = x_transformed.view();
            x_transformed = transform.fit_transform(&x_view)?;
        }
        
        self.fitted = true;
        Ok(x_transformed)
    }

    /// Makes predictions using the final predictor on transformed data.
    ///
    /// This method applies all transformations in the pipeline to the input data,
    /// then passes the transformed data to the predictor component to generate predictions.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform and use for prediction
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, XGBError>` - The predictions on success, or an error
    ///
    /// # Errors
    ///
    /// * `XGBError` containing `PreprocessingError::NotFitted` - If the pipeline has not been fitted
    /// * `XGBError` containing `PreprocessingError::NoPredict` - If no predictor has been set
    /// * Other errors that might be returned by the predictor or transformers
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, ArrayView2};
    /// use xgboostrs::preprocessing::pipeline::Pipeline;
    /// use xgboostrs::preprocessing::scaler::StandardScaler;
    /// use xgboostrs::booster::Booster;
    /// use xgboostrs::preprocessing::{Transformer, TransformerType};
    ///
    /// // Create a pipeline with a scaler
    /// let steps = vec![
    ///     ("scaler".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
    /// ];
    ///
    /// let mut pipeline = Pipeline::new(steps);
    ///
    /// // Set up a predictor (would be an XGBoost model in practice)
    /// // pipeline.predict = Some(Box::new(model));
    ///
    /// // Fit the pipeline on training data
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0]];
    /// pipeline.fit(&train_data.view()).unwrap();
    ///
    /// // Make predictions on new data
    /// let test_data = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
    /// // let predictions = pipeline.predict(&test_data.view()).unwrap();
    /// ```
    pub fn predict<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, XGBError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted).map_err(|e| XGBError::new(e.to_string()));
        }

        if self.predict.is_none() {
            return Err(PreprocessingError::NoPredict).map_err(|e| XGBError::new(e.to_string()));
        }
        
        // First transform the data using all transformers except classifier/regressor
        let transformed = self.transform(x)
            .map_err(|e| XGBError::new(e.to_string()))?;

        let transformed_f32 = transformed.mapv(|v| v as f32);
        
        // Then make predictions using the predictor
        let predictions = self.predict.as_ref().unwrap().predict(&transformed_f32.view())
            .map_err(|e| XGBError::new(e.to_string()))?;

        let predictions_f64 = predictions.mapv(|v| v as f64);
        Ok(predictions_f64)
    }

    pub fn get_named_step(&self, name: &str) -> Option<&TransformerType> {
        self.named_steps.get(name).map(|&index| &self.steps[index].1)
    }

    pub fn get_named_step_mut(&mut self, name: &str) -> Option<&mut TransformerType> {
        self.named_steps.get(name).map(|&index| &mut self.steps[index].1)
    }

    /// Converts the Pipeline to a JSON representation.
    ///
    /// This method serializes the complete pipeline state to JSON, including:
    /// - The initialization parameters (steps)
    /// - The state of each transformer
    /// 
    /// # Returns
    /// A JSON Value containing the serialized pipeline
    pub fn to_json(&self) -> Value {
        let mut steps_json = Vec::new();
        
        // Serialize each step, excluding classifier/regressor steps
        for (name, transformer) in &self.steps {
            // Skip classifier/regressor steps
            if name == "classifier" || name == "regressor" {
                continue;
            }
            
            // Try to get serialized transformer if it implements to_json_opt
            let transformer_serialized = transformer.to_json_opt()
                .unwrap_or_else(|| json!({"type": "Unknown"}));
            
            // Add to steps list
            steps_json.push(json!({
                "name": name,
                "transformer": transformer_serialized
            }));
        }
        
        // Create the JSON structure
        json!({
            "type": "Pipeline",
            "init_params": {
                "steps": steps_json
            },
            "attrs": {
                "fitted": self.fitted
            }
        })
    }

    /// Creates a Pipeline from its JSON representation.
    ///
    /// # Arguments
    /// * `json_data` - The JSON Value containing the serialized pipeline state
    ///
    /// # Returns
    /// * `Result<Self, PreprocessingError>` - A new Pipeline instance with restored state,
    ///   or an error if the JSON data is invalid
    pub fn from_json(json_data: Value) -> Result<Self, PreprocessingError> {
        // Parse steps
        let steps_data = json_data.get("init_params")
            .and_then(|p| p.get("steps"))
            .and_then(|s| s.as_array())
            .ok_or(PreprocessingError::NotFitted)?;
        
        let mut steps = Vec::new();
        
        for step_info in steps_data {
            // Get name
            let name = step_info.get("name")
                .and_then(|n| n.as_str())
                .ok_or(PreprocessingError::NotFitted)?
                .to_string();
            
            // Get transformer data
            let transformer_data = step_info.get("transformer")
                .ok_or(PreprocessingError::NotFitted)?;
            
            // Deserialize transformer based on type
            let transformer_type = transformer_data.get("type")
                .and_then(|t| t.as_str())
                .ok_or(PreprocessingError::NotFitted)?;
            
            let transformer: TransformerType = match transformer_type {
                "StandardScaler" => TransformerType::StandardScaler(StandardScaler::from_json(transformer_data.clone())?),
                "SimpleImputer" => TransformerType::Imputer(SimpleImputer::from_json(transformer_data.clone())?),
                "OneHotEncoder" => TransformerType::OneHotEncoder(OneHotEncoder::from_json(transformer_data.clone())?),
                "ColumnTransformer" => TransformerType::ColumnTransformer(ColumnTransformer::from_json(transformer_data.clone())?),
                "Pipeline" => TransformerType::Pipeline(Pipeline::from_json(transformer_data.clone())?),
                _ => return Err(PreprocessingError::UnknownModelType(transformer_type.to_string())),
            };
            
            steps.push((name, transformer));
        }
        
        // Create Pipeline
        let mut pipeline = Pipeline::new(steps);
        pipeline.fitted = true;
        
        Ok(pipeline)
    }

    /// Saves the pipeline to a JSON file.
    ///
    /// # Parameters
    ///
    /// * `filename` - The path to save the JSON file to
    ///
    /// # Returns
    ///
    /// * `Result<(), PreprocessingError>` - Success or an error
    ///
    /// # Errors
    ///
    /// * IO errors during file writing
    pub fn save_to_file(&self, filename: &str) -> Result<(), PreprocessingError> {
        let json_data = self.to_json();
        
        let file = std::fs::File::create(filename)
            .map_err(|e| PreprocessingError::IoError(e.to_string()))?;
            
        serde_json::to_writer_pretty(file, &json_data)
            .map_err(|e| PreprocessingError::IoError(e.to_string()))
    }

    /// Loads a pipeline from a JSON file.
    ///
    /// # Parameters
    ///
    /// * `filename` - The path to load the JSON file from
    ///
    /// # Returns
    ///
    /// * `Result<Self, PreprocessingError>` - The loaded pipeline or an error
    ///
    /// # Errors
    ///
    /// * IO errors during file reading
    /// * JSON parsing errors
    /// * Invalid pipeline JSON structure
    pub fn load_from_file(filename: &str) -> Result<Self, PreprocessingError> {
        let file = std::fs::File::open(filename)
            .map_err(|e| PreprocessingError::IoError(e.to_string()))?;
            
        let json_data: Value = serde_json::from_reader(file)
            .map_err(|e| PreprocessingError::IoError(e.to_string()))?;
            
        Self::from_json(json_data)
    }

    /// Gets a reference to a transformer by name.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the transformer to retrieve
    ///
    /// # Returns
    ///
    /// * `Option<&Box<dyn Transformer>>` - A reference to the transformer if found, or None
    pub fn get_transformer(&self, name: &str) -> Option<&TransformerType> {
        self.named_steps.get(name).map(|&index| &self.steps[index].1)
    }

    /// Gets a mutable reference to a transformer by name.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the transformer to retrieve
    ///
    /// # Returns
    ///
    /// * `Option<&mut Box<dyn Transformer>>` - A mutable reference to the transformer if found, or None
    pub fn get_transformer_mut(&mut self, name: &str) -> Option<&mut TransformerType> {
        self.named_steps.get(name).map(|&index| &mut self.steps[index].1)
    }
}

impl Transformer for Pipeline {
    /// Fits the pipeline to the data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit the pipeline to
    ///
    /// # Returns
    ///
    /// * `Result<(), PreprocessingError>` - Success or an error
    fn fit<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<(), PreprocessingError> {
        self.fit(x)?;
        Ok(())
    }

    /// Transforms the data using the pipeline.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn transform<T: Copy + 'static + Into<f64>>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        self.transform(x)
    }

    /// Fits and transforms the data using the pipeline.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data to fit and transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, PreprocessingError>` - The transformed data or an error
    fn fit_transform<T: Copy + 'static + Into<f64>>(&mut self, x: &ArrayView2<T>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit_transform(x)
    }
    
    /// Returns the JSON representation of the Pipeline.
    ///
    /// # Returns
    ///
    /// `Option<Value>` - JSON representation of the pipeline
    fn to_json_opt(&self) -> Option<Value> {
        Some(self.to_json())
    }
    
    /// Returns a reference to the Pipeline as a `dyn Any`.
    ///
    /// # Returns
    ///
    /// * `&dyn Any` - A reference to the Pipeline
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    /// Returns a mutable reference to the Pipeline as a `dyn Any`.
    ///
    /// # Returns
    ///
    /// * `&mut dyn Any` - A mutable reference to the Pipeline
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PreprocessingError;
    use ndarray::array;
    use std::fmt::Debug;
    use std::path::Path;
    use crate::preprocessing::scaler::StandardScaler;

    // A simple mock predictor that multiplies values by a constant factor
    struct MockPredictor {
        factor: f32,
    }

    impl MockPredictor {
        fn new(factor: f32) -> Self {
            MockPredictor {
                factor,
            }
        }
    }

    impl Predict for MockPredictor {
        fn predict(&self, x: &ArrayView2F) -> Result<Array2F, XGBError> {
            let mut result = x.to_owned();
            result.mapv_inplace(|v| v * self.factor);
            Ok(result)
        }
    }

    impl Debug for MockPredictor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockPredictor {{ factor: {} }}", self.factor)
        }
    }

    #[test]
    fn test_pipeline_new() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let pipeline = Pipeline::new(steps);
        assert!(!pipeline.fitted);
        assert!(pipeline.predict.is_none());
        assert_eq!(pipeline.steps.len(), 2);
        assert_eq!(pipeline.named_steps.len(), 2);
        assert!(pipeline.named_steps.contains_key("step1"));
        assert!(pipeline.named_steps.contains_key("step2"));
    }

    #[test]
    fn test_pipeline_fit() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x.view()).unwrap();
        
        // Check that it's now fitted
        assert!(pipeline.fitted);
    }

    #[test]
    fn test_pipeline_fit_no_steps() {
        let steps: Vec<(String, TransformerType)> = vec![];
        let mut pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        // Try to fit with no steps
        let result = pipeline.fit(&x.view());
        
        // Should fail with NoSteps error
        assert!(matches!(result, Err(PreprocessingError::NoSteps)));
    }

    #[test]
    fn test_pipeline_transform() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Transform new data
        let x_test = array![
            [2.0, 3.0],
            [4.0, 5.0],
        ];
        
        let result = pipeline.transform(&x_test.view()).unwrap();
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_pipeline_transform_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("classifier".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Transform new data
        let x_test = array![
            [2.0, 3.0],
            [4.0, 5.0],
        ];
        
        let result = pipeline.transform(&x_test.view()).unwrap();
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[2, 2]);
        
        // The classifier step should be skipped, so the result should only be transformed by step1
    }

    #[test]
    fn test_pipeline_transform_not_fitted() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        // Try to transform without fitting
        let result = pipeline.transform(&x.view());
        
        // Should fail with NotFitted error
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_pipeline_fit_transform() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit and transform in one step
        let result = pipeline.fit_transform(&x.view()).unwrap();
        
        // Check that the pipeline is fitted
        assert!(pipeline.fitted);
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_pipeline_fit_transform_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("classifier".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit and transform in one step
        let result = pipeline.fit_transform(&x.view()).unwrap();
        
        // Check that the pipeline is fitted
        assert!(pipeline.fitted);
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[3, 2]);
        
        // The classifier step should be skipped in the transform phase,
        // but should still be fitted
    }

    #[test]
    fn test_pipeline_fit_transform_no_steps() {
        let steps: Vec<(String, TransformerType)> = vec![];
        let mut pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        // Try to fit_transform with no steps
        let result = pipeline.fit_transform(&x.view());
        
        // Should fail with NoSteps error
        assert!(matches!(result, Err(PreprocessingError::NoSteps)));
    }

    #[test]
    fn test_pipeline_predict() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Set a mock predictor
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        // Sample data
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Predict on new data
        let x_test = array![
            [2.0, 3.0],
            [4.0, 5.0],
        ];
        
        let result = pipeline.predict(&x_test.view()).unwrap();
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[2, 2]);
        
        // Expected: data is first standardized (handled by StandardScaler),
        // then each value is multiplied by 2 (MockPredictor with factor=2.0)
    }

    #[test]
    fn test_pipeline_predict_with_classifier() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("classifier".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Set a mock predictor
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        // Sample data
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Predict on new data
        let x_test = array![
            [2.0, 3.0],
            [4.0, 5.0],
        ];
        
        let result = pipeline.predict(&x_test.view()).unwrap();
        
        // Verify shape of the result
        assert_eq!(result.shape(), &[2, 2]);
        
        // Expected: data is first standardized, then the classifier step should be skipped,
        // and finally each value is multiplied by 2 (MockPredictor with factor=2.0)
    }

    #[test]
    fn test_pipeline_predict_not_fitted() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        let x = array![[1.0_f32, 2.0_f32]];
        
        // Try to predict without fitting
        let result = pipeline.predict(&x.view());
        
        // Should fail with NotFitted error
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_predict_no_predictor() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline without setting a predictor
        let x_train = array![[1.0, 2.0]];
        pipeline.fit(&x_train.view()).unwrap();
        
        // Try to predict without a predictor
        let x_test = array![[2.0, 3.0]];
        let result = pipeline.predict(&x_test.view());
        
        // Should fail with NoPredictorSet error
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_to_json() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        pipeline.fit(&x.view()).unwrap();
        
        // Convert to JSON
        let json_data = pipeline.to_json();
        
        // Check basic structure
        assert_eq!(json_data["type"], "Pipeline");
        assert!(json_data["init_params"]["steps"].is_array());
        assert_eq!(json_data["attrs"]["fitted"], true);
        
        // Check steps
        let steps_json = json_data["init_params"]["steps"].as_array().unwrap();
        assert_eq!(steps_json.len(), 2);
        
        // Check first step
        let first_step = &steps_json[0];
        assert_eq!(first_step["name"], "step1");
        assert_eq!(first_step["transformer"]["type"], "StandardScaler");
        
        // Check second step
        let second_step = &steps_json[1];
        assert_eq!(second_step["name"], "step2");
        assert_eq!(second_step["transformer"]["type"], "StandardScaler");
    }

    #[test]
    fn test_pipeline_to_json_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("classifier".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        pipeline.fit(&x.view()).unwrap();
        
        // Convert to JSON
        let json_data = pipeline.to_json();
        
        // Check steps
        let steps_json = json_data["init_params"]["steps"].as_array().unwrap();
        assert_eq!(steps_json.len(), 1);
        
        // Verify names are preserved
        assert_eq!(steps_json[0]["name"], "step1");
    }

    #[test]
    fn test_pipeline_get_transformer() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let pipeline = Pipeline::new(steps);
        
        // Get transformer by name
        let step1 = pipeline.get_transformer("step1").unwrap();
        let step2 = pipeline.get_transformer("step2").unwrap();
        
        // Check they're the right types
        match step1 {
            TransformerType::StandardScaler(_) => {},
            _ => panic!("Expected step1 to be StandardScaler"),
        }
        
        match step2 {
            TransformerType::StandardScaler(_) => {},
            _ => panic!("Expected step2 to be StandardScaler"),
        }
        
        // Check non-existent step returns None
        assert!(pipeline.get_transformer("nonexistent").is_none());
    }

    #[test]
    fn test_pipeline_get_transformer_mut() {
        let steps = vec![
            ("step1".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
            ("step2".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Modify transformer through mutable reference
        {
            let step1 = pipeline.get_transformer_mut("step1").unwrap();
            if let TransformerType::StandardScaler(_) = step1 {
                // Scaler is now modified
            } else {
                panic!("Expected step1 to be StandardScaler");
            }
        }
        
        // Check non-existent step returns None
        assert!(pipeline.get_transformer_mut("nonexistent").is_none());
    }

    #[test]
    fn test_pipeline_from_json() {
        // Create a JSON representation
        let json_data = json!({
            "type": "Pipeline",
            "init_params": {
                "steps": [
                    {
                        "name": "step1",
                        "transformer": {
                            "type": "StandardScaler",
                            "init_params": {},
                            "attrs": {
                                "mean_": [1.0, 2.0],
                                "scale_": [1.0, 1.0]
                            }
                        }
                    },
                    {
                        "name": "step2",
                        "transformer": {
                            "type": "StandardScaler",
                            "init_params": {},
                            "attrs": {
                                "mean_": [0.0, 0.0],
                                "scale_": [1.0, 1.0]
                            }
                        }
                    }
                ]
            },
            "attrs": {
                "fitted": true
            }
        });
        
        // Create pipeline from JSON
        let pipeline = Pipeline::from_json(json_data).unwrap();
        
        // Check it's fitted
        assert!(pipeline.fitted);
        
        // Check steps
        assert_eq!(pipeline.steps.len(), 2);
        
        // Check step names
        assert_eq!(pipeline.steps[0].0, "step1");
        assert_eq!(pipeline.steps[1].0, "step2");
        
        // Check named_steps mapping
        assert!(pipeline.named_steps.contains_key("step1"));
        assert!(pipeline.named_steps.contains_key("step2"));
    }

    #[test]
    fn test_pipeline_save_and_load_from_file() {
        use std::fs;
        
        // Temporary file for testing
        let temp_file = "test_pipeline.json";
        
        // Create a simple pipeline with a StandardScaler
        let steps = vec![
            ("scaler".to_string(), TransformerType::StandardScaler(StandardScaler::new())),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline on some data
        let data = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ];
        
        pipeline.fit(&data.view()).unwrap();
        
        // Save to file
        pipeline.save_to_file(temp_file).unwrap();
        
        // Check the file exists
        assert!(Path::new(temp_file).exists());
        
        // Load from file
        let loaded_pipeline = Pipeline::load_from_file(temp_file).unwrap();
        
        // Compare properties
        assert_eq!(loaded_pipeline.steps.len(), pipeline.steps.len());
        assert_eq!(loaded_pipeline.steps[0].0, pipeline.steps[0].0);
        assert!(loaded_pipeline.fitted);
        
        // Check the loaded pipeline can transform data
        let test_data = array![
            [2.0, 3.0],
            [4.0, 5.0]
        ];
        
        let result = loaded_pipeline.transform(&test_data.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Clean up
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_pipeline_load_from_file() {
        use std::fs;
        
        // Temporary file for testing
        let temp_file = "test_pipeline_load.json";
        
        // Write a valid JSON representation to file
        let json_content = r#"{
            "type": "Pipeline",
            "init_params": {
                "steps": [
                    {
                        "name": "scaler",
                        "transformer": {
                            "type": "StandardScaler",
                            "init_params": {},
                            "attrs": {
                                "mean_": [3.0, 4.0],
                                "scale_": [2.0, 2.0]
                            }
                        }
                    }
                ]
            },
            "attrs": {
                "fitted": true
            }
        }"#;
        
        fs::write(temp_file, json_content).unwrap();
        
        // Load from file
        let pipeline = Pipeline::load_from_file(temp_file).unwrap();
        
        // Check properties
        assert_eq!(pipeline.steps.len(), 1);
        assert_eq!(pipeline.steps[0].0, "scaler");
        assert!(pipeline.fitted);
        
        // Check the loaded pipeline can transform data
        let test_data = array![
            [5.0, 6.0],
            [7.0, 8.0]
        ];
        
        let result = pipeline.transform(&test_data.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Clean up
        fs::remove_file(temp_file).unwrap();
    }
}