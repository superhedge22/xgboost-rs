use ndarray::{Array2, ArrayView2};

use crate::{error::PreprocessingError, Predict, XGBError};

use super::Transformer;

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
        Pipeline {
            steps,
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
        Pipeline {
            steps,
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
    pub fn fit(&mut self, x: &ArrayView2<f32>) -> Result<&mut Self, PreprocessingError> {
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
    pub fn transform(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        
        let mut x_transformed = x.to_owned();
        
        // Apply each transform in the pipeline
        for (_, transform) in &self.steps {
            let x_view = x_transformed.view();
            x_transformed = transform.transform(&x_view)?;
        }
        
        Ok(x_transformed)
    }
    
    /// Fits the pipeline to the data and then transforms it.
    ///
    /// This is a convenience method that combines `fit` and `transform` into a single operation.
    /// It fits each transformer in sequence and transforms the data with all steps.
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
    pub fn fit_transform(&mut self, x: &ArrayView2<f32>) -> Result<Array2<f32>, PreprocessingError> {
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
        
        // Fit and transform the last step
        let (_, last_transform) = self.steps.last_mut().unwrap();
        let x_view = x_transformed.view();
        x_transformed = last_transform.fit_transform(&x_view)?;
        
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
    pub fn predict(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, XGBError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted).map_err(|e| XGBError::new(e.to_string()));
        }

        if self.predict.is_none() {
            return Err(PreprocessingError::NoPredict).map_err(|e| XGBError::new(e.to_string()));
        }
        
        self.predict.as_ref().unwrap().predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PreprocessingError;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
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
        fn predict(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, XGBError> {
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
        
        // Expected: multiply by 2
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 0]], 6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 1]], 8.0, epsilon = 1e-6);
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
}