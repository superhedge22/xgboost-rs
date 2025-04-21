/// Imputation strategies supported by SimpleImputer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImputationStrategy {
    Mean,
    Median,
    MostFrequent,
    Constant(f32),
}

/// Strategy for handling unknown categories in OneHotEncoder
#[derive(Debug, Clone, PartialEq)]
pub enum HandleUnknown {
    Error,
    Ignore,
}