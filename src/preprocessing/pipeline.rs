use std::collections::HashMap;
use serde_json::{json, Value};
use std::any::Any;

use crate::{error::PreprocessingError, Predict, XGBError};
use crate::types::{Array2F, ArrayView2F};

use super::Transformer;
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
/// use xgboostrs::preprocessing::Transformer;
///
/// // Create preprocessing steps
/// let imputer = SimpleImputer::new(ImputationStrategy::Mean);
/// let scaler = StandardScaler::new();
///
/// // Create a pipeline with multiple transformers
/// let steps = vec![
///     ("imputer".to_string(), Box::new(imputer) as Box<dyn Transformer>),
///     ("scaler".to_string(), Box::new(scaler) as Box<dyn Transformer>),
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
    steps: Vec<(String, Box<dyn Transformer>)>,
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
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create transformers
    /// let imputer = SimpleImputer::new(ImputationStrategy::Mean);
    /// let scaler = StandardScaler::new();
    ///
    /// // Build the pipeline
    /// let steps = vec![
    ///     ("imputer".to_string(), Box::new(imputer) as Box<dyn Transformer>),
    ///     ("scaler".to_string(), Box::new(scaler) as Box<dyn Transformer>),
    /// ];
    ///
    /// let pipeline = Pipeline::new(steps);
    /// ```
    pub fn new(steps: Vec<(String, Box<dyn Transformer>)>) -> Self {
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
    pub fn new_with_predictor(steps: Vec<(String, Box<dyn Transformer>)>, predictor: Box<dyn Predict>) -> Self {
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
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create a pipeline with a single scaler
    /// let steps = vec![
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>),
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
    pub fn fit(&mut self, x: &ArrayView2F) -> Result<&mut Self, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::NoSteps);
        }
        
        let mut x_transformed = x.to_owned();
        
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
    /// * `Result<Array2<f32>, PreprocessingError>` - The transformed data on success, or an error
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
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create and fit a pipeline
    /// let steps = vec![
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>),
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
    pub fn transform(&self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        
        let mut x_transformed = x.to_owned();
        
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
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create a pipeline with multiple steps
    /// let steps = vec![
    ///     ("imputer".to_string(), Box::new(SimpleImputer::new(ImputationStrategy::Mean)) as Box<dyn Transformer>),
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>),
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
    pub fn fit_transform(&mut self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::NoSteps);
        }
        
        let mut x_transformed = x.to_owned();
        
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
    /// * `Result<Array2<f32>, XGBError>` - The predictions on success, or an error
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
    /// use xgboostrs::preprocessing::Transformer;
    ///
    /// // Create a pipeline with a scaler
    /// let steps = vec![
    ///     ("scaler".to_string(), Box::new(StandardScaler::new()) as Box<dyn Transformer>),
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
    pub fn predict(&self, x: &ArrayView2F) -> Result<Array2F, XGBError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted).map_err(|e| XGBError::new(e.to_string()));
        }

        if self.predict.is_none() {
            return Err(PreprocessingError::NoPredict).map_err(|e| XGBError::new(e.to_string()));
        }
        
        // First transform the data using all transformers except classifier/regressor
        let transformed = self.transform(x)
            .map_err(|e| XGBError::new(e.to_string()))?;
        
        // Then make predictions using the predictor
        self.predict.as_ref().unwrap().predict(&transformed.view())
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
            
            let transformer: Box<dyn Transformer> = match transformer_type {
                "StandardScaler" => Box::new(StandardScaler::from_json(transformer_data.clone())?),
                "SimpleImputer" => Box::new(SimpleImputer::from_json(transformer_data.clone())?),
                "OneHotEncoder" => Box::new(OneHotEncoder::from_json(transformer_data.clone())?),
                "ColumnTransformer" => Box::new(ColumnTransformer::from_json(transformer_data.clone())?),
                "Pipeline" => Box::new(Pipeline::from_json(transformer_data.clone())?),
                _ => return Err(PreprocessingError::NotFitted),
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
    pub fn get_transformer(&self, name: &str) -> Option<&Box<dyn Transformer>> {
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
    pub fn get_transformer_mut(&mut self, name: &str) -> Option<&mut Box<dyn Transformer>> {
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
    fn fit(&mut self, x: &ArrayView2F) -> Result<(), PreprocessingError> {
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
    /// * `Result<Array2<f32>, PreprocessingError>` - The transformed data or an error
    fn transform(&self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
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
    /// * `Result<Array2<f32>, PreprocessingError>` - The transformed data or an error
    fn fit_transform(&mut self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
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
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::fmt::Debug;
    use std::path::Path;

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
        fn fit(&mut self, _: &ArrayView2F) -> Result<(), PreprocessingError> {
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
            if !self.fitted {
                return Err(PreprocessingError::NotFitted);
            }
            let mut result = x.to_owned();
            result.mapv_inplace(|v| v * self.factor);
            Ok(result)
        }

        fn fit_transform(&mut self, x: &ArrayView2F) -> Result<Array2F, PreprocessingError> {
            self.fit(x)?;
            self.transform(x)
        }
        
        fn to_json_opt(&self) -> Option<Value> {
            Some(json!({
                "type": "MockTransformer",
                "init_params": {
                    "factor": self.factor
                },
                "attrs": {
                    "fitted": self.fitted
                }
            }))
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
        
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Debug for MockTransformer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockTransformer {{ factor: {} }}", self.factor)
        }
    }

    // A simple mock predictor
    struct MockPredictor {
        factor: f32,
    }

    impl MockPredictor {
        fn new(factor: f32) -> Self {
            MockPredictor { factor }
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
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let pipeline = Pipeline::new(steps);
        assert!(!pipeline.fitted);
        assert!(pipeline.predict.is_none());
        assert_eq!(pipeline.steps.len(), 2);
        
        // Check named_steps
        assert_eq!(pipeline.named_steps.len(), 2);
        assert_eq!(pipeline.named_steps.get("step1"), Some(&0));
        assert_eq!(pipeline.named_steps.get("step2"), Some(&1));
    }

    #[test]
    fn test_pipeline_fit() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x.view()).unwrap();
        
        // Check that it's now fitted
        assert!(pipeline.fitted);
    }

    #[test]
    fn test_pipeline_fit_no_steps() {
        let steps: Vec<(String, Box<dyn Transformer>)> = vec![];
        let mut pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        let result = pipeline.fit(&x.view());
        assert!(matches!(result, Err(PreprocessingError::NoSteps)));
    }

    #[test]
    fn test_pipeline_transform() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x.view()).unwrap();
        
        // Transform the data
        let result = pipeline.transform(&x.view()).unwrap();
        
        // Expected result: first multiply by 2, then by 3
        // [1.0, 2.0] * 2 = [2.0, 4.0] * 3 = [6.0, 12.0]
        // [3.0, 4.0] * 2 = [6.0, 8.0] * 3 = [18.0, 24.0]
        assert_abs_diff_eq!(result[[0, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 18.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_transform_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("classifier".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x.view()).unwrap();
        
        // Transform the data
        let result = pipeline.transform(&x.view()).unwrap();
        
        // Expected result: multiply by 2 only (classifier step should be skipped)
        // [1.0, 2.0] * 2 = [2.0, 4.0]
        // [3.0, 4.0] * 2 = [6.0, 8.0]
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_transform_not_fitted() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
        ];

        let pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        let result = pipeline.transform(&x.view());
        assert!(matches!(result, Err(PreprocessingError::NotFitted)));
    }

    #[test]
    fn test_pipeline_fit_transform() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit and transform in one step
        let result = pipeline.fit_transform(&x.view()).unwrap();
        
        // Expected result same as fit + transform
        assert_abs_diff_eq!(result[[0, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 18.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 24.0, epsilon = 1e-10);
        
        // Check that pipeline is fitted
        assert!(pipeline.fitted);
    }

    #[test]
    fn test_pipeline_fit_transform_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("classifier".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Sample data
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit and transform in one step
        let result = pipeline.fit_transform(&x.view()).unwrap();
        
        // Expected result: only first step transforms, classifier fits but doesn't transform
        // [1.0, 2.0] * 2 = [2.0, 4.0]
        // [3.0, 4.0] * 2 = [6.0, 8.0]
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 8.0, epsilon = 1e-10);
        
        // Check that both steps are fitted
        let step1 = pipeline.get_transformer("step1").unwrap();
        let classifier = pipeline.get_transformer("classifier").unwrap();
        
        let step1_mock = step1.as_any().downcast_ref::<MockTransformer>().unwrap();
        let classifier_mock = classifier.as_any().downcast_ref::<MockTransformer>().unwrap();
        
        assert!(step1_mock.fitted);
        assert!(classifier_mock.fitted);
    }

    #[test]
    fn test_pipeline_fit_transform_no_steps() {
        let steps: Vec<(String, Box<dyn Transformer>)> = vec![];
        let mut pipeline = Pipeline::new(steps);
        
        let x = array![[1.0, 2.0]];
        
        let result = pipeline.fit_transform(&x.view());
        assert!(matches!(result, Err(PreprocessingError::NoSteps)));
    }

    #[test]
    fn test_pipeline_predict() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Set a mock predictor
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        // Sample data for fit
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Sample data for predict (note: f32 for predict)
        let x_pred = array![
            [1.0_f32, 2.0_f32],
            [3.0_f32, 4.0_f32],
        ];
        
        // Make prediction
        let result = pipeline.predict(&x_pred.view()).unwrap();
        
        // Expected: first transform with step1 (multiply by 2)
        // Then predict (multiply by 2 again)
        assert_abs_diff_eq!(result[[0, 0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 0]], 12.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 1]], 16.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pipeline_predict_with_classifier() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("classifier".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Set a mock predictor
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        // Sample data for fit
        let x_train = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        // Fit the pipeline
        pipeline.fit(&x_train.view()).unwrap();
        
        // Sample data for predict
        let x_pred = array![
            [1.0_f32, 2.0_f32],
            [3.0_f32, 4.0_f32],
        ];
        
        // Make prediction
        let result = pipeline.predict(&x_pred.view()).unwrap();
        
        // Expected: transform with step1 (multiply by 2), skip classifier,
        // then predict (multiply by 2 again)
        assert_abs_diff_eq!(result[[0, 0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 0]], 12.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 1]], 16.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pipeline_predict_not_fitted() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        pipeline.predict = Some(Box::new(MockPredictor::new(2.0)));
        
        let x = array![[1.0_f32, 2.0_f32]];
        
        let result = pipeline.predict(&x.view());
        assert!(result.is_err()); // Should fail with NotFitted error
    }

    #[test]
    fn test_pipeline_predict_no_predictor() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline without setting a predictor
        let x_train = array![[1.0, 2.0]];
        pipeline.fit(&x_train.view()).unwrap();
        
        let x_pred = array![[1.0_f32, 2.0_f32]];
        
        let result = pipeline.predict(&x_pred.view());
        assert!(result.is_err()); // Should fail with NoPredict error
    }

    #[test]
    fn test_pipeline_to_json() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
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
        assert_eq!(first_step["transformer"]["type"], "MockTransformer");
        assert_eq!(first_step["transformer"]["init_params"]["factor"], 2.0);
        
        // Check second step
        let second_step = &steps_json[1];
        assert_eq!(second_step["name"], "step2");
        assert_eq!(second_step["transformer"]["type"], "MockTransformer");
        assert_eq!(second_step["transformer"]["init_params"]["factor"], 3.0);
    }

    #[test]
    fn test_pipeline_to_json_skip_classifier() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("classifier".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        pipeline.fit(&x.view()).unwrap();
        
        // Convert to JSON
        let json_data = pipeline.to_json();
        
        // Check steps - classifier should be excluded
        let steps_json = json_data["init_params"]["steps"].as_array().unwrap();
        assert_eq!(steps_json.len(), 1);
        
        // Check the only step is step1
        let first_step = &steps_json[0];
        assert_eq!(first_step["name"], "step1");
    }

    #[test]
    fn test_pipeline_get_transformer() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let pipeline = Pipeline::new(steps);
        
        // Get transformer by name
        let step1 = pipeline.get_transformer("step1").unwrap();
        let step1_transformer = step1.as_any().downcast_ref::<MockTransformer>().unwrap();
        assert!(!step1_transformer.fitted);
        assert_eq!(step1_transformer.factor, 2.0);
        
        let step2 = pipeline.get_transformer("step2").unwrap();
        let step2_transformer = step2.as_any().downcast_ref::<MockTransformer>().unwrap();
        assert!(!step2_transformer.fitted);
        assert_eq!(step2_transformer.factor, 3.0);
        
        // Non-existent transformer
        assert!(pipeline.get_transformer("step3").is_none());
    }

    #[test]
    fn test_pipeline_get_transformer_mut() {
        let steps = vec![
            ("step1".to_string(), Box::new(MockTransformer::new(2.0)) as Box<dyn Transformer>),
            ("step2".to_string(), Box::new(MockTransformer::new(3.0)) as Box<dyn Transformer>),
        ];

        let mut pipeline = Pipeline::new(steps);
        
        // Modify transformer through mutable reference
        {
            let step1 = pipeline.get_transformer_mut("step1").unwrap();
            let step1_transformer = step1.as_any_mut().downcast_mut::<MockTransformer>().unwrap();
            step1_transformer.factor = 4.0;
        }
        
        // Check that it was updated
        let step1 = pipeline.get_transformer("step1").unwrap();
        let step1_transformer = step1.as_any().downcast_ref::<MockTransformer>().unwrap();
        assert_eq!(step1_transformer.factor, 4.0);
    }

    #[test]
    fn test_pipeline_from_json() {
        // Create a simple JSON representation of a pipeline with a StandardScaler
        let json_data = json!({
            "type": "Pipeline",
            "init_params": {
                "steps": [
                    {
                        "name": "scaler",
                        "transformer": {
                            "type": "StandardScaler",
                            "init_params": {},
                            "attrs": {
                                "mean_": [1.0, 2.0],
                                "scale_": [0.5, 0.5]
                            }
                        }
                    }
                ]
            },
            "attrs": {
                "fitted": true
            }
        });
        
        // Deserialize the pipeline
        let pipeline = Pipeline::from_json(json_data).unwrap();
        
        // Check that the pipeline was loaded correctly
        assert!(pipeline.fitted);
        assert_eq!(pipeline.steps.len(), 1);
        
        // Check the step name
        let (name, _) = &pipeline.steps[0];
        assert_eq!(name, "scaler");
        
        // Verify we can get the transformer by name
        let scaler = pipeline.get_transformer("scaler").unwrap();
        // Verify it's a StandardScaler
        assert!(scaler.as_any().downcast_ref::<StandardScaler>().is_some());
        
        // Test that the transformer works
        let test_data = array![[3.0, 4.0], [5.0, 6.0]];
        let transformed = pipeline.transform(&test_data.view()).unwrap();
        
        // With mean [1.0, 2.0] and scale [0.5, 0.5], the transformation should be:
        // [3.0, 4.0] -> [(3.0-1.0)/0.5, (4.0-2.0)/0.5] = [4.0, 4.0]
        // [5.0, 6.0] -> [(5.0-1.0)/0.5, (6.0-2.0)/0.5] = [8.0, 8.0]
        assert_abs_diff_eq!(transformed[[0, 0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[0, 1]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[1, 0]], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[1, 1]], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pipeline_save_and_load_from_file() {
        // Create a simple pipeline with a StandardScaler
        let scaler = StandardScaler::new();
        let steps = vec![
            ("scaler".to_string(), Box::new(scaler) as Box<dyn Transformer>),
        ];
        
        let mut pipeline = Pipeline::new(steps);
        
        // Fit the pipeline on some data
        let data = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        pipeline.fit(&data.view()).unwrap();
        
        // Create a temporary file path for saving/loading
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("pipeline_test.json");
        let file_path_str = file_path.to_str().unwrap();
        
        // Save the pipeline to a file
        pipeline.save_to_file(file_path_str).unwrap();
        
        // Verify the file exists
        assert!(Path::new(file_path_str).exists());
        
        // Load the pipeline back from the file
        let loaded_pipeline = Pipeline::load_from_file(file_path_str).unwrap();
        
        // Check that the loaded pipeline has the same structure
        assert_eq!(loaded_pipeline.steps.len(), 1);
        let (name, _) = &loaded_pipeline.steps[0];
        assert_eq!(name, "scaler");
        
        // Test that the loaded pipeline produces the same transforms
        let test_data = array![[5.0, 6.0]];
        let original_transform = pipeline.transform(&test_data.view()).unwrap();
        let loaded_transform = loaded_pipeline.transform(&test_data.view()).unwrap();
        
        // The transformations should be identical - compare element by element
        assert_abs_diff_eq!(original_transform[[0, 0]], loaded_transform[[0, 0]], epsilon = 1e-6);
        assert_abs_diff_eq!(original_transform[[0, 1]], loaded_transform[[0, 1]], epsilon = 1e-6);
        
        // Clean up the temporary file
        std::fs::remove_file(file_path).unwrap_or(());
    }

    #[test]
    fn test_pipeline_load_from_file() {
        // Path to the real pipeline.json file in tests/data
        let file_path = "tests/data/pipeline.json";
        
        // Create test data using from_shape_vec instead of array! macro
        let row0 = vec![-1.0, 84577.0, 2.2419, 0.0001, 0.0001, 0.24986625, 80.75427, 0.01441536, 3.0, -1.389481, 0.092525534, -0.001182328, 0.025774926, 95.96049, 0.682449, 102.24195, 0.3143804, 0.3581181, 0.08653573, 0.11837241, 54.97756, -94.12309, 61.227406, 0.11676354, 59.15876, 0.50125945, 19.0, 2.0, 1.0, 0.0, 322.19, 5600.96, -0.024238005, -0.23266621, 0.792418, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        
        let row1 = vec![1.0, 97872.6, 3.43212, 100.0, 10.0, -16.970562, -1822.5255, -1.4142135, 0.0, -0.12072615, 0.042353157, -0.020430574, 0.6551357, 100.3187, 0.69645035, 99.73405, 0.5190517, 0.9050452, 0.022071825, 0.2977666, 23.773653, -125.53118, 75.935844, 0.24536449, 60.134014, 0.69258237, 17.0, 3.0, 1.0, 0.0, 106.39334, 1287.7201, -0.20301133, -1.7567018, 0.17261323, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        
        let row2 = vec![-1.0, 97028.0, 2762.84, 10.0, 0.0001, 0.26832816, 156290.16, 0.004472136, 20.0, -0.033421878, 0.040152337, -0.030909354, 0.3776858, 99.45543, 0.7778584, 99.62445, 0.37965867, -0.04775532, 0.023975633, 0.056866672, 44.443737, 126.62386, 55.768024, 0.08746645, 56.41818, 0.37846315, 19.0, 6.0, 1.0, 1.0, 582458.0, 34947540.0, -0.04731411, -0.77857465, 0.45905986, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        
        // Combine all rows into a single vector
        let mut all_data = Vec::new();
        all_data.extend_from_slice(&row0);
        all_data.extend_from_slice(&row1);
        all_data.extend_from_slice(&row2);
        
        // Create a 2D array with shape (3, 65)
        let test_data = Array2F::from_shape_vec((3, 65), all_data).unwrap();
        
        // Load the pipeline from file
        let loaded_pipeline = Pipeline::load_from_file(file_path).unwrap();
        
        // Test 1: Check the pipeline structure
        assert!(loaded_pipeline.fitted);
        
        // Test 2: Check that there's a single step named "preprocessor"
        assert_eq!(loaded_pipeline.steps.len(), 1);
        let (step_name, _) = &loaded_pipeline.steps[0];
        assert_eq!(step_name, "preprocessor");
        
        // Test 3: Verify the transformer type is ColumnTransformer
        let preprocessor = loaded_pipeline.get_transformer("preprocessor").unwrap();
        assert!(preprocessor.as_any().downcast_ref::<ColumnTransformer>().is_some());
        
        // Test 4: Look at the serialized form to check the pipeline structure
        let file = std::fs::File::open(file_path).unwrap();
        let pipeline_json: Value = serde_json::from_reader(file).unwrap();
        
        // Check the structure matches what we expect - a pipeline with preprocessor
        let steps = &pipeline_json["init_params"]["steps"]
            .as_array().expect("Missing steps array");
        let preprocessor_step = &steps[0];
        assert_eq!(preprocessor_step["name"].as_str().unwrap(), "preprocessor");
        
        // Test 5: Verify the preprocessor is a ColumnTransformer with a num Pipeline
        let transformers = &preprocessor_step["transformer"]["init_params"]["transformers"]
            .as_array().expect("Missing transformers");
        let num_transformer = &transformers[0];
        assert_eq!(num_transformer["name"].as_str().unwrap(), "num");
        
        // Test 6: Verify the num Pipeline has imputer and scaler in correct order
        let num_steps = &num_transformer["transformer"]["init_params"]["steps"]
            .as_array().expect("Missing num steps");
        
        // First step should be imputer
        let first_step = &num_steps[0];
        assert_eq!(first_step["name"].as_str().unwrap(), "imputer");
        assert_eq!(first_step["transformer"]["type"].as_str().unwrap(), "SimpleImputer");
        
        // Second step should be scaler
        let second_step = &num_steps[1];
        assert_eq!(second_step["name"].as_str().unwrap(), "scaler");
        assert_eq!(second_step["transformer"]["type"].as_str().unwrap(), "StandardScaler");
        
        // Test 7: Try transforming the test data
        let transformed = loaded_pipeline.transform(&test_data.view()).unwrap();
        
        // Test 8: Check that the transformation output has the expected shape
        assert_eq!(transformed.shape()[0], test_data.shape()[0]); // Same number of rows
        assert!(transformed.shape()[1] > 0); // At least one column of output
    }
}