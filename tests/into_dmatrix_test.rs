use xgboostrs::error::XGBResult;
use xgboostrs::types::{Array2F, IntoDMatrix};
use ndarray::{Array, ShapeBuilder};

#[test]
fn test_array2f_into_dmatrix() -> XGBResult<()> {
    // Create a simple 2x3 array
    let array = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_arrayview2f_into_dmatrix() -> XGBResult<()> {
    // Create a simple 2x3 array
    let array = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    // Get a view of the array
    let view = array.view();
    
    // Convert to DMatrix
    let dmatrix = view.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_non_standard_layout() -> XGBResult<()> {
    // Create a array in column-major (Fortran) order
    let array = Array::from_shape_vec((2, 3).f(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    
    // Verify it's not in standard layout
    assert!(!array.is_standard_layout());
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_integer_array_into_dmatrix() -> XGBResult<()> {
    // Create an integer array
    let array = Array::from_shape_vec((2, 3), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_f64_array_into_dmatrix() -> XGBResult<()> {
    // Create a f64 array which gets converted to f32
    let array = Array::from_shape_vec((2, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_integer_arrayview_into_dmatrix() -> XGBResult<()> {
    // Create an integer array
    let array = Array::from_shape_vec((2, 3), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
    
    // Get a view of the array
    let view = array.view();
    
    // Convert to DMatrix
    let dmatrix = view.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 2);
    assert_eq!(dmatrix.num_cols(), 3);
    
    Ok(())
}

#[test]
fn test_empty_array_into_dmatrix() -> XGBResult<()> {
    // Create an empty array (0 rows, 3 columns)
    let array: Array2F = Array::from_shape_vec((0, 3), vec![]).unwrap();
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), 0);
    assert_eq!(dmatrix.num_cols(), 0); // empty is empty
    
    Ok(())
}

#[test]
fn test_large_array_into_dmatrix() -> XGBResult<()> {
    // Create a larger array to test performance with bigger data
    let n_rows = 1000;
    let n_cols = 50;
    let data: Vec<f32> = (0..(n_rows * n_cols)).map(|i| i as f32).collect();
    let array = Array::from_shape_vec((n_rows, n_cols), data).unwrap();
    
    // Convert to DMatrix
    let dmatrix = array.into_dmatrix()?;
    
    // Verify dimensions
    assert_eq!(dmatrix.num_rows(), n_rows);
    assert_eq!(dmatrix.num_cols(), n_cols);
    
    Ok(())
} 