# NeuralForecast API Reference

## Core API Classes

### NeuralForecast

Primary orchestration class providing unified interface for multiple forecasting models.

#### Constructor

```python
NeuralForecast(
    models: List[BaseModel],
    freq: str,
    local_scaler_type: Optional[str] = None
)
```

**Parameters**:
- `models`: List of forecasting model instances
- `freq`: Pandas frequency string (e.g., 'D', 'H', 'M')
- `local_scaler_type`: Scaling method ('standard', 'minmax', 'robust', None)

#### Core Methods

##### fit()
```python
fit(
    df: Union[pd.DataFrame, pl.DataFrame],
    static_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    val_size: Union[int, float] = 0,
    test_size: Union[int, float] = 0,
    n_windows: Optional[int] = None,
    step_size: Optional[int] = None,
    random_seed: int = 1,
    distributed_config: Optional[Dict] = None,
    use_init_validation: bool = False,
    **fit_kwargs
) -> 'NeuralForecast'
```

**Parameters**:
- `df`: Time series DataFrame with columns ['unique_id', 'ds', 'y']
- `static_df`: Static features DataFrame with columns ['unique_id', ...]
- `val_size`: Validation set size (int for samples, float for proportion)
- `test_size`: Test set size (int for samples, float for proportion)
- `n_windows`: Number of cross-validation windows
- `step_size`: Step size for expanding cross-validation
- `random_seed`: Random seed for reproducibility
- `distributed_config`: Ray cluster configuration
- `use_init_validation`: Enable initial validation

**Returns**: Self for method chaining

##### predict()
```python
predict(
    df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    static_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    futr_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    level: Optional[List[Union[int, float]]] = None,
    prediction_intervals: Optional[PredictionIntervals] = None,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    X_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    **predict_kwargs
) -> Union[pd.DataFrame, pl.DataFrame]
```

**Parameters**:
- `df`: Time series DataFrame for context
- `static_df`: Static features DataFrame
- `futr_df`: Future exogenous variables DataFrame
- `level`: Confidence levels for prediction intervals (e.g., [80, 90])
- `prediction_intervals`: Prediction interval configuration
- `id_col`: Name of unique identifier column
- `time_col`: Name of timestamp column
- `target_col`: Name of target variable column
- `X_df`: Alternative name for futr_df

**Returns**: DataFrame with forecasts and optional prediction intervals

##### cross_validation()
```python
cross_validation(
    df: Union[pd.DataFrame, pl.DataFrame],
    static_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    n_windows: int = 1,
    h: Optional[int] = None,
    step_size: Optional[int] = None,
    fitted: bool = False,
    refit: Union[bool, int] = True,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    **cv_kwargs
) -> Union[pd.DataFrame, pl.DataFrame]
```

**Parameters**:
- `df`: Time series DataFrame
- `static_df`: Static features DataFrame
- `n_windows`: Number of cross-validation windows
- `h`: Forecast horizon (uses model horizon if None)
- `step_size`: Step size between windows
- `fitted`: Use already fitted models
- `refit`: Refit frequency (True=always, False=never, int=every N windows)
- `id_col`, `time_col`, `target_col`: Column name specifications

**Returns**: DataFrame with cross-validation results

##### save() / load()
```python
save(
    path: str,
    model_index: Optional[List[int]] = None,
    overwrite: bool = False,
    save_dataset: bool = False
) -> None

@classmethod
load(
    path: str,
    verbose: bool = True
) -> 'NeuralForecast'
```

**Save Parameters**:
- `path`: Directory path for saving
- `model_index`: Indices of models to save (all if None)
- `overwrite`: Allow overwriting existing files
- `save_dataset`: Include dataset in checkpoint

**Load Parameters**:
- `path`: Directory path for loading
- `verbose`: Enable logging output

## Base Model Interface

### BaseModel

Abstract base class defining the interface for all forecasting models.

#### Class Properties

```python
# Capability flags (set by each model)
EXOGENOUS_FUTR: bool = False    # Supports future exogenous variables
EXOGENOUS_HIST: bool = False    # Supports historical exogenous variables
EXOGENOUS_STAT: bool = False    # Supports static exogenous variables
MULTIVARIATE: bool = False      # Produces multivariate forecasts
RECURRENT: bool = False         # Uses recursive prediction
```

#### Constructor Pattern

```python
def __init__(
    self,
    h: int,                              # Forecast horizon
    input_size: int,                     # Input window size
    loss: Union[str, Callable] = 'mae',  # Loss function
    valid_loss: Optional[Union[str, Callable]] = None,  # Validation loss
    max_steps: int = 1000,               # Maximum training steps
    learning_rate: float = 1e-3,         # Learning rate
    num_lr_decays: int = -1,             # Learning rate decay steps
    early_stop_patience_steps: int = -1,  # Early stopping patience
    val_check_steps: int = 100,          # Validation frequency
    batch_size: int = 32,                # Training batch size
    valid_batch_size: Optional[int] = None,  # Validation batch size
    scaler_type: str = 'identity',       # Data scaling method
    random_seed: int = 1,                # Random seed
    num_workers_loader: int = 0,         # DataLoader workers
    drop_last_loader: bool = False,      # Drop incomplete batches
    trainer_kwargs: Optional[Dict] = None,  # PyTorch Lightning trainer args
    **model_kwargs                       # Model-specific parameters
)
```

#### Abstract Methods

```python
def fit(
    self,
    dataset: TimeSeriesDataset,
    val_size: Union[int, float] = 0,
    test_size: Union[int, float] = 0,
    **fit_kwargs
) -> 'BaseModel'

def predict(
    self,
    dataset: TimeSeriesDataset,
    **predict_kwargs
) -> Dict[str, np.ndarray]
```

## Time Series Dataset

### TimeSeriesDataset

Primary data container for time series preprocessing and batching.

#### Constructor

```python
TimeSeriesDataset(
    Y_df: pd.DataFrame,                     # Target time series
    X_df: Optional[pd.DataFrame] = None,    # Exogenous variables
    S_df: Optional[pd.DataFrame] = None,    # Static features
    f_cols: Optional[List[str]] = None,     # Future exogenous columns
    sort_df: bool = True,                   # Sort by time
    id_col: str = 'unique_id',              # ID column name
    time_col: str = 'ds',                   # Time column name
    target_col: str = 'y',                  # Target column name
    **kwargs
)
```

#### Key Methods

```python
# Data manipulation
def temporal_train_test_split(
    self,
    test_size: Union[int, float],
    step_size: int 
) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']

def append(
    self, 
    ts_dataset: 'TimeSeriesDataset'
) -> 'TimeSeriesDataset'

def filter(
    self,
    condition: np.ndarray
) -> 'TimeSeriesDataset'

# Data access
def get_index(self) -> pd.DataFrame
def get_temporal_cols(self) -> List[str] 
def get_static_cols(self) -> List[str]
def get_futr_exog_cols(self) -> List[str]
def get_hist_exog_cols(self) -> List[str]
```

### TimeSeriesLoader

Custom PyTorch DataLoader for efficient batching.

```python
TimeSeriesLoader(
    dataset: TimeSeriesDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    **dataloader_kwargs
)
```

## Loss Functions

### Point Forecasting Losses

#### MAE (Mean Absolute Error)
```python
class MAE:
    def __init__(self, horizon_weight: Optional[np.ndarray] = None)
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor
```

#### MSE (Mean Squared Error)  
```python
class MSE:
    def __init__(self, horizon_weight: Optional[np.ndarray] = None)
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor
```

#### RMSE (Root Mean Squared Error)
```python
class RMSE:
    def __init__(self, horizon_weight: Optional[np.ndarray] = None) 
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor
```

#### MAPE (Mean Absolute Percentage Error)
```python
class MAPE:
    def __init__(self, horizon_weight: Optional[np.ndarray] = None)
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor
```

#### SMAPE (Symmetric Mean Absolute Percentage Error)
```python
class SMAPE:
    def __init__(self, horizon_weight: Optional[np.ndarray] = None)
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor
```

### Probabilistic Losses

#### DistributionLoss
```python
class DistributionLoss:
    def __init__(
        self,
        distribution: str,                    # 'Normal', 'Poisson', 'Negative Binomial', etc.
        horizon_weight: Optional[np.ndarray] = None,
        return_params: bool = False
    )
    def __call__(self, y: torch.Tensor, distr_args: torch.Tensor) -> torch.Tensor
```

**Supported Distributions**:
- `'Normal'`: Gaussian distribution
- `'Poisson'`: Poisson distribution  
- `'NegativeBinomial'`: Negative binomial distribution
- `'StudentT'`: Student's t-distribution
- `'Beta'`: Beta distribution

## Utility Functions

### Data Generation

#### generate_series()
```python
def generate_series(
    n_series: int = 100,
    n_temporal_features: int = 0,
    n_static_features: int = 0,
    equal_ends: bool = False,
    n_max: int = 500,
    n_min: int = 50,
    freq: str = 'D',
    static_as_cat: bool = True,
    with_trend: bool = False,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]
```

**Parameters**:
- `n_series`: Number of time series to generate
- `n_temporal_features`: Number of time-varying exogenous variables
- `n_static_features`: Number of static features per series
- `equal_ends`: All series end at same time
- `n_max`/`n_min`: Maximum/minimum series length
- `freq`: Pandas frequency string
- `static_as_cat`: Generate categorical static features
- `with_trend`: Include trend component
- `seed`: Random seed

**Returns**: Tuple of (time_series_df, static_features_df)

### Time Feature Extraction

#### time_features_from_frequency_str()
```python
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]
```

**Parameters**:
- `freq_str`: Pandas frequency string ('H', 'D', 'M', etc.)

**Returns**: List of appropriate time feature extractors

#### augment_calendar_df()
```python
def augment_calendar_df(
    df: pd.DataFrame,
    freq: str,
    id_col: str = 'unique_id',
    time_col: str = 'ds'
) -> pd.DataFrame
```

**Parameters**:
- `df`: Input DataFrame with time series
- `freq`: Data frequency
- `id_col`: Unique identifier column name
- `time_col`: Timestamp column name

**Returns**: DataFrame with added calendar features

### Prediction Intervals

#### PredictionIntervals
```python
class PredictionIntervals:
    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = 'conformal_distribution'
    )
```

**Methods**:
- `'conformal_distribution'`: Distribution-based conformal prediction
- `'conformal_error'`: Error-based conformal prediction
- `'quantile'`: Direct quantile prediction

## Model Implementations

### Basic Neural Networks

#### MLP
```python
class MLP(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        hidden_size: int = 256,          # Hidden layer size
        n_hidden_layers: int = 2,        # Number of hidden layers
        activation: str = 'ReLU',        # Activation function
        dropout: float = 0.0,            # Dropout rate
        **base_model_kwargs
    )
```

#### LSTM
```python
class LSTM(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size  
        encoder_n_layers: int = 2,       # LSTM layers
        encoder_hidden_size: int = 128,  # LSTM hidden size
        encoder_dropout: float = 0.0,    # LSTM dropout
        decoder_hidden_size: int = 128,  # Decoder hidden size
        decoder_layers: int = 2,         # Decoder layers
        **base_model_kwargs
    )
```

#### GRU
```python  
class GRU(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        encoder_n_layers: int = 2,       # GRU layers
        encoder_hidden_size: int = 128,  # GRU hidden size
        encoder_dropout: float = 0.0,    # GRU dropout
        **base_model_kwargs
    )
```

### Advanced Models

#### NBEATS
```python
class NBEATS(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        stack_types: List[str] = ['identity', 'trend', 'seasonality'],
        n_blocks: List[int] = [3, 3, 3], # Blocks per stack
        n_layers: int = 4,               # Layers per block
        layer_widths: int = 512,         # Layer width
        share_weights_in_stack: bool = False,  # Share weights
        **base_model_kwargs
    )
```

#### NHiTS
```python
class NHiTS(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        n_pool_kernel_size: List[int] = [2, 2, 1],  # Pooling kernels
        n_freq_downsample: List[int] = [4, 2, 1],   # Downsampling
        interpolation_mode: str = 'linear',          # Interpolation
        n_blocks: List[int] = [1, 1, 1],            # Blocks per stack
        **base_model_kwargs
    )
```

#### Autoformer
```python
class Autoformer(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        hidden_size: int = 512,          # Model dimension
        n_head: int = 8,                 # Attention heads
        n_layers: int = 2,               # Encoder/decoder layers
        dropout: float = 0.05,           # Dropout rate
        factor: int = 1,                 # Attention factor
        **base_model_kwargs
    )
```

#### TFT (Temporal Fusion Transformer)
```python
class TFT(BaseModel):
    def __init__(
        self,
        h: int,                          # Forecast horizon
        input_size: int,                 # Input window size
        hidden_size: int = 128,          # Hidden state size
        lstm_layers: int = 1,            # LSTM layers
        num_attention_heads: int = 4,    # Attention heads
        dropout: float = 0.1,            # Dropout rate
        **base_model_kwargs
    )
```

## Data Formats and Schemas

### Standard Time Series Format

```python
# Required columns
df = pd.DataFrame({
    'unique_id': str,        # Series identifier (required)
    'ds': pd.Timestamp,      # Timestamp (required) 
    'y': float,              # Target variable (required)
})

# With exogenous variables
df_with_exog = pd.DataFrame({
    'unique_id': str,        # Series identifier
    'ds': pd.Timestamp,      # Timestamp
    'y': float,              # Target variable
    'exog_1': float,         # Exogenous variable 1
    'exog_2': float,         # Exogenous variable 2
    # ... additional exogenous variables
})

# Static features
static_df = pd.DataFrame({
    'unique_id': str,        # Series identifier (required)
    'static_0': float,       # Numeric static feature
    'static_1': str,         # Categorical static feature
    # ... additional static features
})

# Future exogenous variables
futr_df = pd.DataFrame({
    'unique_id': str,        # Series identifier
    'ds': pd.Timestamp,      # Future timestamp
    'exog_1': float,         # Future exogenous variable
    # ... additional future exogenous variables
})
```

### Forecast Output Format

```python
# Basic forecast output
forecasts = pd.DataFrame({
    'unique_id': str,           # Series identifier
    'ds': pd.Timestamp,         # Forecast timestamp
    'ModelName': float,         # Point forecast
})

# With prediction intervals
forecasts_with_intervals = pd.DataFrame({
    'unique_id': str,           # Series identifier
    'ds': pd.Timestamp,         # Forecast timestamp
    'ModelName': float,         # Point forecast
    'ModelName-lo-80': float,   # 80% lower bound
    'ModelName-hi-80': float,   # 80% upper bound
    'ModelName-lo-90': float,   # 90% lower bound  
    'ModelName-hi-90': float,   # 90% upper bound
})
```

## Configuration Reference

### Frequency Strings

Standard pandas frequency aliases:
- `'H'`: Hourly
- `'D'`: Daily  
- `'W'`: Weekly
- `'M'`: Monthly
- `'Q'`: Quarterly
- `'Y'`: Yearly
- `'B'`: Business day
- `'MS'`: Month start
- `'QS'`: Quarter start

### Scaler Types

Available scaling methods:
- `'identity'`: No scaling
- `'standard'`: Standard scaling (mean=0, std=1)
- `'minmax'`: Min-max scaling (range=[0,1])
- `'robust'`: Robust scaling (median=0, IQR=1)

### Activation Functions

Supported activation functions:
- `'ReLU'`: Rectified Linear Unit
- `'GELU'`: Gaussian Error Linear Unit
- `'Tanh'`: Hyperbolic tangent
- `'Sigmoid'`: Sigmoid function
- `'LeakyReLU'`: Leaky ReLU
- `'ELU'`: Exponential Linear Unit

This comprehensive API reference covers all major interfaces and parameters needed for implementing a complete Rust port of NeuralForecast with 100% feature parity.