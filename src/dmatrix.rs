use libc::{c_float, c_uint};
use log::info;
use std::os::unix::ffi::OsStrExt;
use std::{ffi, path::Path, ptr, slice};
use serde_json::json;

use crate::error::XGBResult;
use crate::error::XGBError;

static KEY_GROUP_PTR: &str = "group_ptr";
static KEY_GROUP: &str = "group";
static KEY_LABEL: &str = "label";
static KEY_WEIGHT: &str = "weight";
static KEY_BASE_MARGIN: &str = "base_margin";

/// Data matrix used throughout `XGBoost` for training/predicting [`Booster`](struct.Booster.html) models.
///
/// It's used as a container for both features (i.e. a row for every instance), and an optional true label for that
/// instance (as an `f32` value).
#[derive(Debug, Clone)]
pub struct DMatrix {
    pub(super) handle: xgboost_rs_sys::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

unsafe impl Send for DMatrix {}
unsafe impl Sync for DMatrix {}

impl DMatrix {
    /// Construct a new instance from a `DMatrixHandle` created by the `XGBoost` C API.
    fn new(handle: xgboost_rs_sys::DMatrixHandle) -> XGBResult<Self> {
        // number of rows/cols are frequently read throughout applications, so more convenient to pull them out once
        // when the matrix is created, instead of having to check errors each time XGDMatrixNum* is called
        let mut out = 0;
        xgb_call!(xgboost_rs_sys::XGDMatrixNumRow(handle, &mut out))?;
        let num_rows = out as usize;

        let mut out = 0;
        xgb_call!(xgboost_rs_sys::XGDMatrixNumCol(handle, &mut out))?;
        let num_cols = out as usize;
        info!("Loaded DMatrix with shape: {}x{}", num_rows, num_cols);
        Ok(DMatrix {
            handle,
            num_rows,
            num_cols,
        })
    }

    /// Create a new `DMatrix` from slice in column-major order.
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix creation fails with an error that doesn't come from `XGBoost`.
    pub fn from_col_major_f32(
        data: &[f32],
        byte_size_ax_0: usize,
        byte_size_ax_1: usize,
        n_rows: usize,
        n_cols: usize,
        n_thread: i32,
        nan: f32,
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();

        // xg needs to know, where the first element of the data resides in memory
        let data_ptr_address = data.as_ptr() as usize;

        // most important part is the definition of the strides!
        // also, the strides are given as the number of bytes times whatever ndarray's stride function returns.
        // e.g. arr.strides() -> [1,10320] must be passed as [8, 82560].
        let array_config = format!(
            "{{
            \"data\": [{data_ptr_address}, false], 
            \"strides\": [{byte_size_ax_0}, {byte_size_ax_1}], 
            \"descr\": [[\"\", \"<f4\"]], 
            \"typestr\": \"<f4\", 
            \"shape\": [{n_rows}, {n_cols}], 
            \"version\": 3
        }}"
        );

        let json_config = format!(
            "
                {{ \"missing\": {nan}, \"nthread\": {n_thread}}}
                "
        );

        let array_config_cstr = ffi::CString::new(array_config).unwrap();
        let json_config_cstr = ffi::CString::new(json_config).unwrap();

        xgb_call!(xgboost_rs_sys::XGDMatrixCreateFromDense(
            array_config_cstr.as_ptr(),
            json_config_cstr.as_ptr(),
            &mut handle
        ))?;
        Ok(DMatrix::new(handle).unwrap())
    }

    /// Create a new `DMatrix` from dense array in row-major order.
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix creation fails with an error not coming from `XGBoost`.

    pub fn from_dense(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_rs_sys::XGDMatrixCreateFromMat(
            data.as_ptr(),
            num_rows as xgboost_rs_sys::bst_ulong,
            (data.len() / num_rows) as xgboost_rs_sys::bst_ulong,
            0.0, // TODO: can values be missing here?
            &mut handle
        ))?;
        Ok(DMatrix::new(handle).unwrap())
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_cols` is set to None, number of columns will be inferred from given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix creation fails with an error not coming from `XGBoost`.
    pub fn from_csr(
        indptr: &[usize],
        indices: &[usize],
        data: &[f32],
        num_cols: Option<usize>,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        
        // Convert vectors to the correct types
        let indptr_vec: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices_vec: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let num_cols = num_cols.unwrap_or(0); // infer from data if 0
        
        // Get pointers for passing to C API
        let indptr_ptr = indptr_vec.as_ptr() as usize;
        let indices_ptr = indices_vec.as_ptr() as usize;
        let data_ptr = data.as_ptr() as usize;
        
        // Create array interfaces for indptr, indices and data
        let indptr_config = json!({
            "data": [indptr_ptr, false],
            "typestr": "<u8",
            "shape": [indptr_vec.len()],
            "version": 3
        });
        let indptr_str = indptr_config.to_string();
        
        let indices_config = json!({
            "data": [indices_ptr, false],
            "typestr": "<u4",
            "shape": [indices_vec.len()],
            "version": 3
        });
        let indices_str = indices_config.to_string();
        
        let data_config = json!({
            "data": [data_ptr, false],
            "typestr": "<f4", 
            "shape": [data.len()],
            "version": 3
        });
        let data_str = data_config.to_string();
        
        // Create the CSR configuration with required params
        let csr_config = format!(
            "{{\"missing\": 0.0, \"nthread\": 0}}"
        );
        
        let indptr_cstr = ffi::CString::new(indptr_str).unwrap();
        let indices_cstr = ffi::CString::new(indices_str).unwrap();
        let data_cstr = ffi::CString::new(data_str).unwrap();
        let csr_config_cstr = ffi::CString::new(csr_config).unwrap();
        
        xgb_call!(xgboost_rs_sys::XGDMatrixCreateFromCSR(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_cols as xgboost_rs_sys::bst_ulong,
            csr_config_cstr.as_ptr(),
            &mut handle
        ))?;
        
        Ok(DMatrix::new(handle).unwrap())
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrix.
    ///
    /// Uses standard CSC representation where the row indices for column _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_rows` is set to None, number of rows will be inferred from given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix creation fails with an error not coming from `XGBoost`.
    pub fn from_csc(
        indptr: &[usize],
        indices: &[usize],
        data: &[f32],
        num_rows: Option<usize>,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        
        // Convert vectors to the correct types
        let indptr_vec: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices_vec: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let num_rows = num_rows.unwrap_or(0); // infer from data if 0
        
        // Get pointers for passing to C API
        let indptr_ptr = indptr_vec.as_ptr() as usize;
        let indices_ptr = indices_vec.as_ptr() as usize;
        let data_ptr = data.as_ptr() as usize;
        
        // Create array interfaces for indptr, indices and data
        let indptr_config = json!({
            "data": [indptr_ptr, false],
            "typestr": "<u8",
            "shape": [indptr_vec.len()],
            "version": 3
        });
        let indptr_str = indptr_config.to_string();
        
        let indices_config = json!({
            "data": [indices_ptr, false],
            "typestr": "<u4",
            "shape": [indices_vec.len()],
            "version": 3
        });
        let indices_str = indices_config.to_string();
        
        let data_config = json!({
            "data": [data_ptr, false],
            "typestr": "<f4", 
            "shape": [data.len()],
            "version": 3
        });
        let data_str = data_config.to_string();
        
        // Create the CSC configuration with required params
        let csc_config = format!(
            "{{\"missing\": 0.0, \"nthread\": 0}}"
        );
        
        let indptr_cstr = ffi::CString::new(indptr_str).unwrap();
        let indices_cstr = ffi::CString::new(indices_str).unwrap();
        let data_cstr = ffi::CString::new(data_str).unwrap();
        let csc_config_cstr = ffi::CString::new(csc_config).unwrap();
        
        xgb_call!(xgboost_rs_sys::XGDMatrixCreateFromCSC(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_rows as xgboost_rs_sys::bst_ulong,
            csc_config_cstr.as_ptr(),
            &mut handle
        ))?;
        
        Ok(DMatrix::new(handle).unwrap())
    }

    /// Create a new `DMatrix` from given file.
    ///
    /// Supports text files in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, CSV,
    /// binary files written either by `save`, or from another `XGBoost` library.
    ///
    /// For more details on accepted formats, seem the
    /// [`XGBoost` input format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    /// documentation.
    ///
    /// # LIBSVM format
    ///
    /// Specified data in a sparse format as:
    /// ```text
    /// <label> <index>:<value> [<index>:<value> ...]
    /// ```
    ///
    /// E.g.
    /// ```text
    /// 0 1:1 9:0 11:0
    /// 1 9:1 11:0.375 15:1
    /// 0 1:0 8:0.22 11:1
    /// ```
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix creation fails with an error not coming from `XGBoost`.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        let path_as_string = path.as_ref().display().to_string();
        
        let mut handle = ptr::null_mut();
        let silent = 1; // 1 for silent, 0 for not silent
        
        let config = json!({
            "uri": path_as_string.replace("\\", "\\\\"),
            "silent": silent
        });
        
        
        let config_cstr = ffi::CString::new(config.to_string()).unwrap();
        xgb_call!(xgboost_rs_sys::XGDMatrixCreateFromURI(
            config_cstr.as_ptr(),
            &mut handle
        ))?;
        Ok(DMatrix::new(handle).unwrap())
    }

    /// Serialise this `DMatrix` as a binary file to given path.
    ///
    /// # Panics
    ///
    /// Will panic, if the matrix saving fails with an error not coming from `XGBoost`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        let silent = true;
        xgb_call!(xgboost_rs_sys::XGDMatrixSaveBinary(
            self.handle,
            fname.as_ptr(),
            i32::from(silent)
        ))
    }

    /// Get the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Get the shape (rows x columns) of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    /// Get a new `DMatrix` as a containing only given indices.
    ///
    /// # Panics
    ///
    /// Will panic, if the slice creation fails with an error not coming from `XGBoost`.
    pub fn slice(&self, indices: &[usize]) -> XGBResult<DMatrix> {
        let mut out_handle = ptr::null_mut();
        let indices: Vec<i32> = indices.iter().map(|x| *x as i32).collect();
        xgb_call!(xgboost_rs_sys::XGDMatrixSliceDMatrix(
            self.handle,
            indices.as_ptr(),
            indices.len() as xgboost_rs_sys::bst_ulong,
            &mut out_handle
        ))?;
        Ok(DMatrix::new(out_handle).unwrap())
    }

    /// Get ground truth labels for each row of this matrix.
    pub fn get_labels(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_LABEL)
    }

    /// Set ground truth labels for each row of this matrix.
    pub fn set_labels(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_LABEL, array)
    }

    /// Get weights of each instance.
    pub fn get_weights(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_WEIGHT)
    }

    /// Set weights of each instance.
    pub fn set_weights(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_WEIGHT, array)
    }

    /// Get base margin.
    pub fn get_base_margin(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_BASE_MARGIN)
    }

    /// Set base margin.
    ///
    /// If specified, xgboost will start from this margin, can be used to specify initial prediction to boost from.
    pub fn set_base_margin(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_BASE_MARGIN, array)
    }

    /// Set the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the `XGBoost` documentation for more information.
    pub fn set_group(&mut self, group: &[u32]) -> XGBResult<()> {
        // same as xgb_call!(xgboost_rs_sys::XGDMatrixSetGroup(self.handle, group.as_ptr(), group.len() as u64))
        self.set_uint_info(KEY_GROUP, group)
    }

    /// Get the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the `XGBoost` documentation for more information.
    pub fn get_group(&self) -> XGBResult<&[u32]> {
        self.get_uint_info(KEY_GROUP_PTR)
    }

    fn get_float_info(&self, field: &str) -> XGBResult<&[f32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_rs_sys::XGDMatrixGetFloatInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        // In XGBoost 3.0.0, when there is no info, out_dptr might be null even with out_len = 0
        if out_dptr.is_null() || out_len == 0 {
            // Return an empty slice when no data is available
            return Ok(&[]);
        }

        Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_float, out_len as usize) })
    }

    fn set_float_info(&mut self, field: &str, array: &[f32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        
        // Get pointer for passing to C API
        let data_ptr = array.as_ptr() as usize;
        
        // Create array interface JSON for float data
        let array_config = json!({
            "data": [data_ptr, false],
            "typestr": "<f4",
            "shape": [array.len()],
            "version": 3
        });
        let array_str = array_config.to_string();
        let array_cstr = ffi::CString::new(array_str).unwrap();
        
        xgb_call!(xgboost_rs_sys::XGDMatrixSetInfoFromInterface(
            self.handle,
            field.as_ptr(),
            array_cstr.as_ptr()
        ))
    }

    fn get_uint_info(&self, field: &str) -> XGBResult<&[u32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_rs_sys::XGDMatrixGetUIntInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;
        
        // In XGBoost 3.0.0, when there is no info, out_dptr might be null even with out_len = 0
        if out_dptr.is_null() || out_len == 0 {
            // Return an empty slice when no data is available
            return Ok(&[]);
        }
        
        Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_uint, out_len as usize) })
    }

    fn set_uint_info(&mut self, field: &str, array: &[u32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        
        // Get pointer for passing to C API
        let data_ptr = array.as_ptr() as usize;
        
        // Create array interface JSON for uint data
        let array_config = json!({
            "data": [data_ptr, false],
            "typestr": "<u4",
            "shape": [array.len()],
            "version": 3
        });
        let array_str = array_config.to_string();
        let array_cstr = ffi::CString::new(array_str).unwrap();
        
        xgb_call!(xgboost_rs_sys::XGDMatrixSetInfoFromInterface(
            self.handle,
            field.as_ptr(),
            array_cstr.as_ptr()
        ))
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        xgb_call!(xgboost_rs_sys::XGDMatrixFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_train_matrix() -> XGBResult<DMatrix> {
        // Use env! to locate the file relative to the project root
        // This works in any build environment, including when running tests
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
        // Specify that column index 5 (6th column, "Class_ID") is the label column in the CSV file
        DMatrix::load(format!("{data_path}/data.csv?format=csv&label_column=5"))
    }

    #[test]
    fn test_read_matrix() {
        assert!(read_train_matrix().is_ok());
    }

    #[test]
    fn test_read_num_rows() {
        assert_eq!(read_train_matrix().unwrap().num_rows(), 23946);
    }

    #[test]
    fn test_read_num_cols() {
        assert_eq!(read_train_matrix().unwrap().num_cols(), 5);
    }

    #[test]
    fn test_writing_and_reading() {
        let dmat = read_train_matrix().unwrap();

        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = tmp_dir.path().join("dmat.bin");
        dmat.save(&out_path).unwrap();

        let dmat2 = DMatrix::load(&out_path).unwrap();

        assert_eq!(dmat.num_rows(), dmat2.num_rows());
        assert_eq!(dmat.num_cols(), dmat2.num_cols());
        // TODO: check contents as well, if possible
    }

    #[test]
    fn test_get_set_labels() {
        let mut dmat = read_train_matrix().unwrap();
        assert_eq!(dmat.get_labels().unwrap().len(), 23946);
        let labels = vec![0.0; dmat.get_labels().unwrap().len()];
        assert!(dmat.set_labels(&labels).is_ok());
        assert_eq!(dmat.get_labels().unwrap(), labels);
    }

    #[test]
    fn test_get_set_weights() {
        let error_margin = f32::EPSILON;
        let mut dmat = read_train_matrix().unwrap();
        let empty_weights: Vec<f32> = vec![];
        assert_eq!(dmat.get_weights().unwrap(), empty_weights.as_slice());

        let weight = [1.0, 10.0, 44.9555];
        assert!(dmat.set_weights(&weight).is_ok());
        dmat.get_weights()
            .unwrap()
            .iter()
            .zip(weight.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < error_margin);
            });
    }

    #[test]
    fn test_get_set_base_margin() {
        let mut dmat = read_train_matrix().unwrap();
        let empty_slice: Vec<f32> = vec![];
        assert_eq!(dmat.get_base_margin().unwrap(), empty_slice.as_slice());
        let base_margin = vec![1337.0; dmat.num_rows()];
        assert!(dmat.set_base_margin(&base_margin).is_ok());
        assert_eq!(dmat.get_base_margin().unwrap(), base_margin);
    }

    #[test]
    fn test_get_set_group() {
        let mut dmat = read_train_matrix().unwrap();
        let empty_slice: Vec<u32> = vec![];
        assert_eq!(dmat.get_group().unwrap(), empty_slice.as_slice());

        let group = [1];
        assert!(dmat.set_group(&group).is_ok());
        assert_eq!(dmat.get_group().unwrap(), &[0, 1]);
    }

    #[test]
    fn test_from_csr() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 0); // https://github.com/dmlc/xgboost/pull/7265

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 10);
    }

    #[test]
    fn test_from_csc() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
    }

    #[test]
    fn test_from_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_rows = 2;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(dmat.num_cols(), 3);

        let data = vec![1.0, 2.0, 3.0];
        let num_rows = 3;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 1);
    }

    #[test]
    fn test_slice_from_indices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();

        assert_eq!(dmat.shape(), (4, 2));

        assert_eq!(dmat.slice(&[]).unwrap().shape(), (0, 2));
        assert_eq!(dmat.slice(&[1]).unwrap().shape(), (1, 2));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 2));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 2));
    }

    #[test]
    fn test_slice() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 3));

        assert_eq!(dmat.slice(&[0, 1, 2, 3]).unwrap().shape(), (4, 3));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[1, 0]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[0, 1, 2]).unwrap().shape(), (3, 3));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 3));
    }
}
