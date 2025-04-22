#![allow(dead_code)]

use error::XGBError;
use ndarray::{Array2, ArrayView2};

extern crate derive_builder;
extern crate indexmap;
extern crate libc;
extern crate log;
extern crate tempfile;
extern crate xgboost_rs_sys;


pub trait Predict {
    fn predict(&self, x: &ArrayView2<f32>) -> Result<Array2<f32>, XGBError>;
}

macro_rules! xgb_call {
    ($x:expr) => {
        XGBError::check_return_value(unsafe { $x })
    };
}



/// Trait for integer square root
trait IntegerSqrt {
    fn integer_sqrt(self) -> usize;
}

impl IntegerSqrt for usize {
    fn integer_sqrt(self) -> usize {
        let sqrt = (self as f64).sqrt();
        let sqrt_int = sqrt.round() as usize;
        
        // Verify the result is (almost) an integer
        assert!((sqrt - sqrt_int as f64).abs() < 1e-10, 
                "Square root of {} is not an integer: {}", self, sqrt);
        
        sqrt_int
    }
}

pub mod error;
pub mod dmatrix;
pub mod booster;
pub mod parameters;

pub mod preprocessing;
pub mod types;