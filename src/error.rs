//! Functionality related to errors and error handling.

use std::error::Error;
use std::ffi::CStr;
use std::fmt::{self, Display};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum PreprocessingError {
    #[error("Cannot fit with empty array")]
    EmptyArray,
    #[error("Failed to compute mean")]
    ComputeMean,
    #[error("Failed to find most frequent value")]
    MostFrequent,
    #[error("Model has not been fitted yet. Call 'fit' first.")]
    NotFitted,
    #[error("X has {0} features, but model was fitted with {1} features.")]
    FeatureMismatch(usize, usize),
    #[error("Found unknown category {0} in column {1} during transform")]
    UnknownCategory(f64, usize),
    #[error("Column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
    #[error("Model has no steps")]
    NoSteps,
    #[error("Model has no predict")]
    NoPredict,
    #[error("IO error: {0}")]
    IoError(String),
}

/// Convenience return type for most operations which can return an `XGBError`.
pub type XGBResult<T> = std::result::Result<T, XGBError>;

/// Wrap errors returned by the `XGBoost` library.
#[derive(Debug, Eq, PartialEq)]
pub struct XGBError {
    desc: String,
}

impl XGBError {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        XGBError { desc: desc.into() }
    }

    /// Check the return value from an `XGBoost` FFI call, and return the last error message on error.
    ///
    /// Return values of 0 are treated as success, returns values of -1 are treated as errors.
    ///
    /// Meaning of any other return values are undefined, and will cause a panic.
    pub(crate) fn check_return_value(ret_val: i32) -> XGBResult<()> {
        match ret_val {
            0 => Ok(()),
            -1 => Err(XGBError::from_xgboost()),
            _ => panic!("unexpected return value '{ret_val}', expected 0 or -1"),
        }
    }

    /// Get the last error message from `XGBoost`.
    fn from_xgboost() -> Self {
        let c_str = unsafe { CStr::from_ptr(xgboost_rs_sys::XGBGetLastError()) };
        let str_slice = c_str.to_str().unwrap();
        XGBError {
            desc: str_slice.to_owned(),
        }
    }
}

impl Error for XGBError {}

impl Display for XGBError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "XGBoost error: {}", &self.desc)
    }
}

impl From<std::ffi::NulError> for XGBError {
    fn from(source: std::ffi::NulError) -> Self {
        XGBError {
            desc: source.to_string(),
        }
    }
}

impl From<std::str::Utf8Error> for XGBError {
    fn from(source: std::str::Utf8Error) -> Self {
        XGBError {
            desc: source.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_value_handling() {
        let result = XGBError::check_return_value(0);
        assert_eq!(result, Ok(()));

        let result = XGBError::check_return_value(-1);
        assert_eq!(
            result,
            Err(XGBError {
                desc: String::new()
            })
        );
    }
}
