import numpy as np
import pandas as pd
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)



class CustomStandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X_array = np.asarray(X)
        self.mean_ = np.mean(X_array, axis=0)
        self.scale_ = np.std(X_array, axis=0, ddof=1)
        # Handle zeros in scale (avoid division by zero)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        X_array = np.asarray(X)
        return (X_array - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def to_json(self):
        """Serialize the complete scaler state to JSON."""
        return {
            'type': 'StandardScaler',
            'init_params': {},  # No init params for this implementation
            'attrs': {
                'mean_': self.mean_,
                'scale_': self.scale_
            }
        }

    @classmethod
    def from_json(cls, json_data):
        """Create a scaler from complete JSON state."""
        # Create a new instance with init params
        scaler = cls(**json_data.get('init_params', {}))

        # Set attributes directly
        attrs = json_data.get('attrs', {})
        if 'mean_' in attrs:
            scaler.mean_ = np.array(attrs['mean_'])
        if 'scale_' in attrs:
            scaler.scale_ = np.array(attrs['scale_'])

        return scaler


class CustomSimpleImputer:
    """
    Imputation for completing missing values using specified strategy.
    """

    def __init__(self, strategy='median', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy should be one of {valid_strategies}")

    def fit(self, X, y=None):
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            X_array = X.values
            self.columns = X.columns
        else:
            X_array = np.asarray(X)

        # Calculate statistics for each column
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X_array, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X_array, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = np.zeros(X_array.shape[1])

            for j in range(X_array.shape[1]):
                column = X_array[:, j]
                # Remove NaN values
                valid_values = column[~np.isnan(column)]

                if len(valid_values) > 0:
                    # Find the most frequent value
                    unique_values, counts = np.unique(valid_values, return_counts=True)
                    self.statistics_[j] = unique_values[np.argmax(counts)]
                else:
                    self.statistics_[j] = np.nan
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X_array.shape[1], self.fill_value)

        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise ValueError("Imputer has not been fitted yet. Call 'fit' first.")

        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            X_array = X.values.copy()
            columns = X.columns
            index = X.index
        else:
            X_array = np.asarray(X).copy()

        # Replace NaN values with the corresponding statistic
        for j in range(X_array.shape[1]):
            mask = np.isnan(X_array[:, j])
            X_array[mask, j] = self.statistics_[j]

        # Return the same type as input
        if is_dataframe:
            return pd.DataFrame(X_array, index=index, columns=columns)
        else:
            return X_array

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def to_json(self):
        """Serialize the complete imputer state to JSON."""
        # Handle the columns attribute which might be a numpy array or pandas Index
        columns_attr = getattr(self, 'columns', None)
        serialized_columns = None

        if columns_attr is not None:
            if hasattr(columns_attr, 'tolist'):
                serialized_columns = columns_attr.tolist()
            elif isinstance(columns_attr, list):
                serialized_columns = columns_attr
            else:
                serialized_columns = str(columns_attr)

        return {
            'type': 'SimpleImputer',
            'init_params': {
                'strategy': self.strategy,
                'fill_value': self.fill_value
            },
            'attrs': {
                'statistics_': self.statistics_,
                'columns': serialized_columns
            }
        }

    @classmethod
    def from_json(cls, json_data):
        """Create an imputer from complete JSON state."""
        # Create a new instance with init params
        imputer = cls(**json_data.get('init_params', {}))

        # Set attributes directly
        attrs = json_data.get('attrs', {})
        if 'statistics_' in attrs:
            imputer.statistics_ = np.array(attrs['statistics_'])
        if 'columns' in attrs and attrs['columns'] is not None:
            imputer.columns = attrs['columns']

        return imputer


class CustomOneHotEncoder:
    """
    One-hot encoding for categorical features.
    """

    def __init__(self, handle_unknown='error'):
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.feature_names_in_ = None

        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown should be 'error' or 'ignore'")

    def fit(self, X, y=None):
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            self.feature_names_in_ = X.columns
            X_array = X.values
        else:
            X_array = np.asarray(X)

        # Learn categories for each feature
        self.categories_ = []
        for j in range(X_array.shape[1]):
            column = X_array[:, j]
            unique_cats = np.unique(column)
            self.categories_.append(unique_cats)

        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder has not been fitted yet. Call 'fit' first.")

        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            X_array = X.values
            index = X.index
        else:
            X_array = np.asarray(X)

        # Determine output shape and create empty array
        n_samples = X_array.shape[0]
        n_features = sum(len(cats) for cats in self.categories_)
        X_encoded = np.zeros((n_samples, n_features))

        # Generate column names for the encoded features
        if is_dataframe and self.feature_names_in_ is not None:
            column_names = []
            for i, (feature, cats) in enumerate(zip(self.feature_names_in_, self.categories_)):
                for cat in cats:
                    column_names.append(f"{feature}_{cat}")
        else:
            column_names = None

        # Fill the encoded matrix
        col_idx = 0
        for j, categories in enumerate(self.categories_):
            for cat in categories:
                # Find where the current feature equals the current category
                is_category = (X_array[:, j] == cat)

                # Handle unknown categories
                if self.handle_unknown == 'ignore':
                    X_encoded[is_category, col_idx] = 1
                else:  # 'error'
                    if not np.all(np.isin(X_array[:, j], categories)):
                        unknown = np.setdiff1d(X_array[:, j], categories)
                        raise ValueError(f"Found unknown categories {unknown} in column {j} during transform")
                    X_encoded[is_category, col_idx] = 1

                col_idx += 1

        # Return DataFrame if input was DataFrame
        if is_dataframe and column_names:
            return pd.DataFrame(X_encoded, index=index, columns=column_names)
        else:
            return X_encoded

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def to_json(self):
        """Serialize the complete encoder state to JSON."""
        # Convert categories to lists of strings for JSON compatibility
        categories_list = []
        for category_array in self.categories_ if self.categories_ is not None else []:
            categories_list.append([str(c) for c in category_array])

        return {
            'type': 'OneHotEncoder',
            'init_params': {
                'handle_unknown': self.handle_unknown
            },
            'attrs': {
                'categories_': categories_list,
                'feature_names_in_': (
                    list(self.feature_names_in_)
                    if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None
                    else None
                )
            }
        }

    @classmethod
    def from_json(cls, json_data):
        """Create an encoder from complete JSON state."""
        # Create a new instance with init params
        encoder = cls(**json_data.get('init_params', {}))

        # Set attributes directly
        attrs = json_data.get('attrs', {})
        if 'categories_' in attrs:
            encoder.categories_ = [np.array(cat) for cat in attrs['categories_']]
        if 'feature_names_in_' in attrs and attrs['feature_names_in_'] is not None:
            encoder.feature_names_in_ = np.array(attrs['feature_names_in_'])

        return encoder


class ColumnTransformer:
    """
    Applies transformers to columns of an array or pandas DataFrame.
    """

    def __init__(self, transformers):
        self.transformers = transformers
        self.transformers_dict = {name: transformer for name, transformer, _ in transformers}
        self.fitted_transformers = {}

    def fit(self, X, y=None, **fit_params):
        # Handle pandas DataFrame
        is_dataframe = isinstance(X, pd.DataFrame)

        # Fit each transformer on its subset of columns
        for name, transformer, columns in self.transformers:
            if is_dataframe:
                # Get subset of data for this transformer
                X_subset = X[columns] if isinstance(columns, list) else X.iloc[:, columns]
            else:
                # For numpy arrays, columns must be indices
                X_subset = X[:, columns]

            # Fit the transformer, passing y to match sklearn's API
            transformer.fit(X_subset, y)

            # Store the fitted transformer
            self.fitted_transformers[name] = transformer

        return self

    def transform(self, X):
        if not self.fitted_transformers:
            raise ValueError("ColumnTransformer has not been fitted yet. Call 'fit' first.")

        # Handle pandas DataFrame
        is_dataframe = isinstance(X, pd.DataFrame)

        # Apply each transformer to its subset of columns
        transformed_arrays = []
        column_names = []

        for name, _, columns in self.transformers:
            transformer = self.fitted_transformers[name]

            if is_dataframe:
                # Get subset of data for this transformer
                X_subset = X[columns] if isinstance(columns, list) else X.iloc[:, columns]
            else:
                # For numpy arrays, columns must be indices
                X_subset = X[:, columns]

            # Transform the subset
            X_transformed = transformer.transform(X_subset)

            # Store the transformed data
            transformed_arrays.append(X_transformed)

            # Collect column names if available
            if is_dataframe and hasattr(X_transformed, 'columns'):
                column_names.extend(X_transformed.columns)
            elif is_dataframe and isinstance(columns, list):
                if X_transformed.shape[1] == len(columns):
                    column_names.extend([f"{name}_{col}" for col in columns])
                else:
                    column_names.extend([f"{name}_feature_{i}" for i in range(X_transformed.shape[1])])

        # Concatenate the transformed arrays
        if all(isinstance(arr, pd.DataFrame) for arr in transformed_arrays):
            # If all are DataFrames, concatenate as DataFrame
            result = pd.concat(transformed_arrays, axis=1)
        else:
            # Convert any DataFrames to arrays
            arrays = [arr.values if isinstance(arr, pd.DataFrame) else arr for arr in transformed_arrays]
            # Concatenate along columns
            result = np.hstack(arrays)

            # Convert back to DataFrame if input was DataFrame and we have column names
            if is_dataframe and column_names:
                result = pd.DataFrame(result, index=X.index, columns=column_names)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        """
        Added the y parameter to match sklearn's API
        """
        return self.fit(X, y, **fit_params).transform(X)

    def to_json(self):
        """Serialize the complete column transformer state to JSON."""
        transformer_data = {
            'type': 'ColumnTransformer',
            'init_params': {
                'transformers': []
            },
            'attrs': {
                'fitted_transformers': {}
            }
        }

        # Serialize each transformer
        for name, transformer, columns in self.transformers:
            # Process columns to include indices
            column_info = {'names': [], 'indices': []}

            if isinstance(columns, list):
                column_info['names'] = columns
                column_info['indices'] = list(range(len(columns)))
            elif hasattr(columns, 'tolist'):
                column_names = columns.tolist()
                column_info['names'] = column_names
                column_info['indices'] = list(range(len(column_names)))
            else:
                column_info['names'] = str(columns)
                column_info['indices'] = None

            # Serialize the transformer
            if hasattr(transformer, 'to_json'):
                tr_serialized = transformer.to_json()
            else:
                tr_serialized = {'type': type(transformer).__name__}

            # Add to transformers list
            transformer_data['init_params']['transformers'].append({
                'name': name,
                'transformer': tr_serialized,
                'columns': column_info
            })

            # Add fitted transformer if available
            if name in self.fitted_transformers:
                fitted_transformer = self.fitted_transformers[name]
                if hasattr(fitted_transformer, 'to_json'):
                    transformer_data['attrs']['fitted_transformers'][name] = fitted_transformer.to_json()

        return transformer_data

    @classmethod
    def from_json(cls, json_data):
        """Create a column transformer from complete JSON state."""

        # Function to deserialize a transformer
        def deserialize_transformer(tr_data):
            tr_type = tr_data.get('type')

            if tr_type == 'StandardScaler':
                return CustomStandardScaler.from_json(tr_data)
            elif tr_type == 'SimpleImputer':
                return CustomSimpleImputer.from_json(tr_data)
            elif tr_type == 'OneHotEncoder':
                return CustomOneHotEncoder.from_json(tr_data)
            elif tr_type == 'Pipeline':
                return Pipeline.from_json(tr_data)
            elif tr_type == 'ColumnTransformer':
                return ColumnTransformer.from_json(tr_data)
            else:
                raise ValueError(f"Unknown transformer type: {tr_type}")

        # Build transformers list for initialization
        transformers_data = json_data.get('init_params', {}).get('transformers', [])
        transformers = []

        for tr_info in transformers_data:
            name = tr_info['name']
            transformer = deserialize_transformer(tr_info['transformer'])

            # Get columns (use names by default)
            columns = tr_info['columns']['names']

            transformers.append((name, transformer, columns))

        # Create column transformer
        col_transformer = cls(transformers=transformers)

        # Set fitted transformers if available
        fitted_transformers_data = json_data.get('attrs', {}).get('fitted_transformers', {})
        for name, tr_data in fitted_transformers_data.items():
            col_transformer.fitted_transformers[name] = deserialize_transformer(tr_data)

        return col_transformer


class Pipeline:
    """
    Chain multiple transformers into a single transformation pipeline.
    """

    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: transform for name, transform in steps}

    def fit(self, X, y=None, **fit_params):
        X_transformed = X

        # Fit each step in the pipeline except the last one
        for name, transform in self.steps[:-1]:
            X_transformed = transform.fit_transform(X_transformed, y)

        # Fit the last step separately
        last_name, last_transform = self.steps[-1]
        last_transform.fit(X_transformed, y)

        return self

    def transform(self, X):
        """
        Transform X by applying each transformer in the pipeline.
        This does NOT apply transform on the final estimator.
        """
        X_transformed = X

        # Apply transform to all but the last step
        for name, transform in self.steps:
            if name == 'classifier':
                continue
            X_transformed = transform.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        X_transformed = X

        # Fit and transform each step in the pipeline except the last one
        for name, transform in self.steps:
            if name == 'classifier' or name == 'regressor':
                continue
            X_transformed = transform.fit_transform(X_transformed, y)

        # Fit the last step if it's a classifier or regressor
        last_name, last_transform = self.steps[-1]
        if last_name == 'classifier' or last_name == 'regressor':
            last_transform.fit(X_transformed, y)

        return X_transformed

    def predict(self, X):
        """
        Apply transforms and predict with the final estimator.
        """
        # Transform the data using all transformers (not the final estimator)
        X_transformed = self.transform(X)

        # Predict with the final estimator
        return self.steps[-1][1].predict(X_transformed)

    def predict_proba(self, X):
        """
        Apply transforms and predict_proba with the final estimator.
        """
        # Transform the data using all transformers (not the final estimator)
        X_transformed = self.transform(X)

        # Predict probabilities with the final estimator
        return self.steps[-1][1].predict_proba(X_transformed)

    def to_json(self):
        """
        Serialize the preprocessing pipeline state to JSON.
        This automatically excludes classifier steps.
        """
        serialized = {
            'type': 'Pipeline',
            'init_params': {
                'steps': []  # We'll populate this differently
            },
            'attrs': {}
        }

        # Determine which steps to serialize (exclude classifier steps)
        steps_to_serialize = []
        for name, transform in self.steps:
            # Skip classifier steps - detect by type name or predict method
            if (type(transform).__name__.lower().endswith(('classifier', 'regressor')) or
                hasattr(transform, 'predict')):
                print(f"Skipping classifier/regressor step: {name}")
                continue
            steps_to_serialize.append((name, transform))

        # Serialize each preprocessing step
        for name, transform in steps_to_serialize:
            # Serialize the transformer
            if hasattr(transform, 'to_json'):
                transform_data = transform.to_json()
            else:
                # Fallback for unknown transformers
                transform_data = {'type': type(transform).__name__}

            serialized['init_params']['steps'].append({
                'name': name,
                'transformer': transform_data
            })

        return serialized

    @classmethod
    def from_json(cls, json_data):
        """Create a pipeline from complete JSON state."""

        # Function to deserialize a transformer
        def deserialize_transformer(tr_data):
            tr_type = tr_data.get('type')

            if tr_type == 'StandardScaler':
                return CustomStandardScaler.from_json(tr_data)
            elif tr_type == 'SimpleImputer':
                return CustomSimpleImputer.from_json(tr_data)
            elif tr_type == 'OneHotEncoder':
                return CustomOneHotEncoder.from_json(tr_data)
            elif tr_type == 'Pipeline':
                return Pipeline.from_json(tr_data)
            elif tr_type == 'ColumnTransformer':
                return ColumnTransformer.from_json(tr_data)
            else:
                raise ValueError(f"Unknown transformer type: {tr_type}")

        # Build steps for initialization
        steps_data = json_data.get('init_params', {}).get('steps', [])
        steps = []

        for step_info in steps_data:
            name = step_info['name']
            transformer = deserialize_transformer(step_info['transformer'])
            steps.append((name, transformer))

        # Create pipeline
        return cls(steps=steps)

    def save_to_file(self, filename):
        """
        Serialize the preprocessing pipeline to a JSON file.
        This automatically excludes classifier steps.

        Parameters:
        -----------
        filename : str
            The filename to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load_from_file(cls, filename):
        """
        Load a pipeline from a JSON file.

        Parameters:
        -----------
        filename : str
            The filename to load from

        Returns:
        --------
        Pipeline
            The loaded pipeline
        """
        with open(filename, 'r') as f:
            json_data = json.load(f)
        return cls.from_json(json_data)

    def extract_preprocessing_metadata(self, filename=None, debug=False):
        """
        Extract key preprocessing metadata with column indices for Rust implementation.

        Parameters:
        -----------
        filename : str, optional
            If provided, save the metadata to this JSON file
        debug : bool, default=False
            If True, print debug information about the extraction process

        Returns:
        --------
        dict
            Dictionary with extracted metadata
        """
        metadata = {
            'numeric_features': {
                'column_names': [],
                'column_indices': [],
                'imputer': {
                    'strategy': None,
                    'statistics': None
                },
                'scaler': {
                    'mean': None,
                    'scale': None
                }
            },
            'categorical_features': {
                'column_names': [],
                'column_indices': [],
                'encoder': {
                    'categories': [],
                    'handle_unknown': None
                }
            }
        }

        # Find the ColumnTransformer
        col_transformer = None
        for name, transform in self.steps:
            if isinstance(transform, ColumnTransformer):
                if debug:
                    print(f"Found ColumnTransformer: {name}")
                col_transformer = transform
                break

        if col_transformer is None:
            # Check nested pipelines
            for name, transform in self.steps:
                if isinstance(transform, Pipeline):
                    if debug:
                        print(f"Checking nested pipeline: {name}")
                    for sub_name, sub_transform in transform.steps:
                        if isinstance(sub_transform, ColumnTransformer):
                            if debug:
                                print(f"Found ColumnTransformer in nested pipeline: {sub_name}")
                            col_transformer = sub_transform
                            break
                    if col_transformer:
                        break

        if col_transformer is None:
            if debug:
                print("No ColumnTransformer found in pipeline!")
            if filename:
                with open(filename, 'w') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            return metadata

        # Process each transformer in the ColumnTransformer
        for name, transformer, columns in col_transformer.transformers:
            if debug:
                print(f"Processing transformer: {name} ({type(transformer).__name__})")

            # Get column names and indices
            col_names = []
            col_indices = []

            if isinstance(columns, list):
                col_names = columns
                col_indices = list(range(len(col_names)))
            elif hasattr(columns, 'tolist'):
                col_names = columns.tolist()
                col_indices = list(range(len(col_names)))
            else:
                col_names = [str(columns)]
                col_indices = [0]

            # Check if this is a numeric transformer
            if name == 'num':
                metadata['numeric_features']['column_names'] = col_names
                metadata['numeric_features']['column_indices'] = col_indices

                # If the transformer is itself a Pipeline, we need to extract from its steps
                if isinstance(transformer, Pipeline):
                    if debug:
                        print(f"  Numeric transformer is a Pipeline with {len(transformer.steps)} steps")
                        for step_name, step in transformer.steps:
                            print(f"    Step: {step_name} ({type(step).__name__})")

                    # Look for the imputer
                    for step_name, step in transformer.steps:
                        if isinstance(step, CustomSimpleImputer):
                            if debug:
                                print(f"  Found imputer in step: {step_name}")
                            metadata['numeric_features']['imputer']['strategy'] = step.strategy
                            if hasattr(step, 'statistics_') and step.statistics_ is not None:
                                metadata['numeric_features']['imputer']['statistics'] = step.statistics_.tolist()

                    # Look for the scaler
                    for step_name, step in transformer.steps:
                        if isinstance(step, CustomStandardScaler):
                            if debug:
                                print(f"  Found scaler in step: {step_name}")
                                if hasattr(step, 'mean_'):
                                    print(f"    Scaler has mean_ attribute with {len(step.mean_)} values")
                                if hasattr(step, 'scale_'):
                                    print(f"    Scaler has scale_ attribute with {len(step.scale_)} values")

                            if hasattr(step, 'mean_') and step.mean_ is not None:
                                metadata['numeric_features']['scaler']['mean'] = step.mean_.tolist()
                            if hasattr(step, 'scale_') and step.scale_ is not None:
                                metadata['numeric_features']['scaler']['scale'] = step.scale_.tolist()

                # If the transformer is a direct scaler or imputer
                elif isinstance(transformer, CustomStandardScaler):
                    if debug:
                        print("  Numeric transformer is a direct StandardScaler")
                    if hasattr(transformer, 'mean_') and transformer.mean_ is not None:
                        metadata['numeric_features']['scaler']['mean'] = transformer.mean_.tolist()
                    if hasattr(transformer, 'scale_') and transformer.scale_ is not None:
                        metadata['numeric_features']['scaler']['scale'] = transformer.scale_.tolist()

                elif isinstance(transformer, CustomSimpleImputer):
                    if debug:
                        print("  Numeric transformer is a direct SimpleImputer")
                    metadata['numeric_features']['imputer']['strategy'] = transformer.strategy
                    if hasattr(transformer, 'statistics_') and transformer.statistics_ is not None:
                        metadata['numeric_features']['imputer']['statistics'] = transformer.statistics_.tolist()

                # Check if we need to get the fitted transformer from the ColumnTransformer
                if metadata['numeric_features']['scaler'][
                    'mean'] is None and name in col_transformer.fitted_transformers:
                    fitted_transformer = col_transformer.fitted_transformers[name]
                    if debug:
                        print(f"  Checking fitted transformer: {type(fitted_transformer).__name__}")

                    if isinstance(fitted_transformer, Pipeline):
                        for step_name, step in fitted_transformer.steps:
                            if isinstance(step, CustomStandardScaler):
                                if debug:
                                    print(f"  Found scaler in fitted step: {step_name}")
                                if hasattr(step, 'mean_') and step.mean_ is not None:
                                    metadata['numeric_features']['scaler']['mean'] = step.mean_.tolist()
                                if hasattr(step, 'scale_') and step.scale_ is not None:
                                    metadata['numeric_features']['scaler']['scale'] = step.scale_.tolist()

            # Handle categorical features similarly
            elif name == 'cat':
                metadata['categorical_features']['column_names'] = col_names
                metadata['categorical_features']['column_indices'] = col_indices

                # Similar logic for categorical transformer...
                if isinstance(transformer, Pipeline):
                    for step_name, step in transformer.steps:
                        if isinstance(step, CustomOneHotEncoder):
                            metadata['categorical_features']['encoder']['handle_unknown'] = step.handle_unknown
                            if hasattr(step, 'categories_') and step.categories_ is not None:
                                categories = [[str(cat) for cat in category] for category in step.categories_]
                                metadata['categorical_features']['encoder']['categories'] = categories

                elif isinstance(transformer, CustomOneHotEncoder):
                    metadata['categorical_features']['encoder']['handle_unknown'] = transformer.handle_unknown
                    if hasattr(transformer, 'categories_') and transformer.categories_ is not None:
                        categories = [[str(cat) for cat in category] for category in transformer.categories_]
                        metadata['categorical_features']['encoder']['categories'] = categories

        # Add metadata validation
        has_scaler = metadata['numeric_features']['scaler']['mean'] is not None
        metadata['has_scaler'] = has_scaler
        metadata['feature_count'] = len(metadata['numeric_features']['column_names'])
        metadata['has_categorical'] = len(metadata['categorical_features']['column_names']) > 0

        if debug:
            print(f"Extracted metadata summary:")
            print(f"  Features: {metadata['feature_count']}")
            print(f"  Has scaler: {metadata['has_scaler']}")
            print(f"  Has categorical: {metadata['has_categorical']}")

        # Save to file if requested
        if filename:
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            if debug:
                print(f"Saved metadata to {filename}")

        return metadata