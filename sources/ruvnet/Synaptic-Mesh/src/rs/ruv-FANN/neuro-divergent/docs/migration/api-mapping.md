# API Mapping: Python NeuralForecast to Rust neuro-divergent

This document provides complete API equivalence mappings between Python NeuralForecast and Rust neuro-divergent, covering all classes, methods, parameters, and return types.

## Core Classes

### NeuralForecast

| Python | Rust | Notes |
|--------|------|-------|
| `NeuralForecast(models, freq)` | `NeuralForecast::builder().with_models(models).with_frequency(freq).build()?` | Builder pattern |
| `nf.fit(df)` | `nf.fit(df)?` | Error handling with `?` |
| `nf.predict()` | `nf.predict()?` | Returns `Result<DataFrame>` |
| `nf.predict(futr_df)` | `nf.predict_with_future(futr_df)?` | Explicit method name |
| `nf.cross_validation(df, n_windows)` | `nf.cross_validation(df, n_windows)?` | Same interface |
| `nf.save(path)` | `nf.save(path)?` | Error-safe serialization |
| `NeuralForecast.load(path)` | `NeuralForecast::load(path)?` | Static method |

#### Python Example
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

nf = NeuralForecast(models=[LSTM(h=12)], freq='D')
nf.fit(df)
forecasts = nf.predict()
```

#### Rust Equivalent
```rust
use neuro_divergent::{NeuralForecast, models::LSTM, Frequency};

let lstm = LSTM::builder().horizon(12).build()?;
let mut nf = NeuralForecast::builder()
    .with_models(vec![Box::new(lstm)])
    .with_frequency(Frequency::Daily)
    .build()?;

nf.fit(df)?;
let forecasts = nf.predict()?;
```

## Model Base Classes

### BaseModel Interface Mapping

| Python | Rust | Notes |
|--------|------|-------|
| `model.fit(df, val_df=None)` | `model.fit(df, val_df)?` | Optional validation data |
| `model.predict(df)` | `model.predict(df)?` | Error handling |
| `model.predict_insample(df)` | `model.predict_insample(df)?` | In-sample predictions |
| `model.save(path)` | `model.save(path)?` | Model serialization |
| `Model.load(path)` | `Model::load(path)?` | Static loading method |
| `model.get_params()` | `model.params()` | Getter method |
| `model.set_params(**params)` | `model.set_params(params)?` | Setter with validation |

### Model Configuration Mapping

| Python Parameter | Rust Parameter | Type Conversion |
|------------------|----------------|-----------------|
| `h` | `horizon` | `usize` |
| `input_size` | `input_size` | `usize` |
| `max_steps` | `max_steps` | `usize` |
| `learning_rate` | `learning_rate` | `f64` |
| `batch_size` | `batch_size` | `usize` |
| `random_state` | `random_seed` | `Option<u64>` |
| `verbose` | `verbose` | `bool` |
| `early_stop_patience_steps` | `early_stop_patience` | `usize` |

## Basic Models

### MLP (Multi-Layer Perceptron)

**Python**:
```python
from neuralforecast.models import MLP

model = MLP(
    h=12,
    input_size=24,
    hidden_size=128,
    num_layers=3,
    dropout=0.1,
    activation='ReLU',
    learning_rate=0.001,
    max_steps=1000,
    batch_size=32
)
```

**Rust**:
```rust
use neuro_divergent::models::MLP;
use neuro_divergent::models::Activation;

let model = MLP::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .num_layers(3)
    .dropout(0.1)
    .activation(Activation::ReLU)
    .learning_rate(0.001)
    .max_steps(1000)
    .batch_size(32)
    .build()?;
```

### DLinear

**Python**:
```python
from neuralforecast.models import DLinear

model = DLinear(
    h=12,
    input_size=24,
    kernel_size=25,
    learning_rate=0.001
)
```

**Rust**:
```rust
use neuro_divergent::models::DLinear;

let model = DLinear::builder()
    .horizon(12)
    .input_size(24)
    .kernel_size(25)
    .learning_rate(0.001)
    .build()?;
```

### NLinear

**Python**:
```python
from neuralforecast.models import NLinear

model = NLinear(h=12, input_size=24)
```

**Rust**:
```rust
use neuro_divergent::models::NLinear;

let model = NLinear::builder()
    .horizon(12)
    .input_size(24)
    .build()?;
```

## Recurrent Models

### LSTM

**Python**:
```python
from neuralforecast.models import LSTM

model = LSTM(
    h=12,
    input_size=24,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    bidirectional=False,
    learning_rate=0.001
)
```

**Rust**:
```rust
use neuro_divergent::models::LSTM;

let model = LSTM::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .num_layers(2)
    .dropout(0.1)
    .bidirectional(false)
    .learning_rate(0.001)
    .build()?;
```

### GRU

**Python**:
```python
from neuralforecast.models import GRU

model = GRU(
    h=12,
    input_size=24,
    hidden_size=128,
    num_layers=2
)
```

**Rust**:
```rust
use neuro_divergent::models::GRU;

let model = GRU::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .num_layers(2)
    .build()?;
```

### DeepAR

**Python**:
```python
from neuralforecast.models import DeepAR

model = DeepAR(
    h=12,
    input_size=24,
    hidden_size=64,
    num_layers=2,
    num_samples=100,
    learning_rate=0.001
)
```

**Rust**:
```rust
use neuro_divergent::models::DeepAR;

let model = DeepAR::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(64)
    .num_layers(2)
    .num_samples(100)
    .learning_rate(0.001)
    .build()?;
```

## Advanced Models

### NBEATS

**Python**:
```python
from neuralforecast.models import NBEATS

model = NBEATS(
    h=12,
    input_size=24,
    stack_types=['trend', 'seasonality'],
    n_blocks=[3, 3],
    mlp_units=[[256, 256], [256, 256]],
    n_harmonics=1,
    n_polynomials=2
)
```

**Rust**:
```rust
use neuro_divergent::models::NBEATS;

let model = NBEATS::builder()
    .horizon(12)
    .input_size(24)
    .stack_types(vec!["trend", "seasonality"])
    .n_blocks(vec![3, 3])
    .mlp_units(vec![vec![256, 256], vec![256, 256]])
    .n_harmonics(1)
    .n_polynomials(2)
    .build()?;
```

### NBEATSx

**Python**:
```python
from neuralforecast.models import NBEATSx

model = NBEATSx(
    h=12,
    input_size=24,
    n_harmonics=1,
    n_polynomials=2,
    x_s_n_hidden=0
)
```

**Rust**:
```rust
use neuro_divergent::models::NBEATSx;

let model = NBEATSx::builder()
    .horizon(12)
    .input_size(24)
    .n_harmonics(1)
    .n_polynomials(2)
    .x_s_n_hidden(0)
    .build()?;
```

### NHITS

**Python**:
```python
from neuralforecast.models import NHITS

model = NHITS(
    h=12,
    input_size=24,
    n_freq_downsample=[2, 1, 1],
    n_blocks=1
)
```

**Rust**:
```rust
use neuro_divergent::models::NHITS;

let model = NHITS::builder()
    .horizon(12)
    .input_size(24)
    .n_freq_downsample(vec![2, 1, 1])
    .n_blocks(1)
    .build()?;
```

## Transformer Models

### TFT (Temporal Fusion Transformer)

**Python**:
```python
from neuralforecast.models import TFT

model = TFT(
    h=12,
    input_size=24,
    hidden_size=128,
    n_head=4,
    attn_dropout=0.1,
    dropout=0.1,
    learning_rate=0.001
)
```

**Rust**:
```rust
use neuro_divergent::models::TFT;

let model = TFT::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .n_head(4)
    .attn_dropout(0.1)
    .dropout(0.1)
    .learning_rate(0.001)
    .build()?;
```

### Autoformer

**Python**:
```python
from neuralforecast.models import Autoformer

model = Autoformer(
    h=12,
    input_size=24,
    hidden_size=128,
    n_head=8,
    e_layers=2,
    d_layers=1,
    dropout=0.05
)
```

**Rust**:
```rust
use neuro_divergent::models::Autoformer;

let model = Autoformer::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .n_head(8)
    .e_layers(2)
    .d_layers(1)
    .dropout(0.05)
    .build()?;
```

### Informer

**Python**:
```python
from neuralforecast.models import Informer

model = Informer(
    h=12,
    input_size=24,
    hidden_size=128,
    n_head=8,
    e_layers=2,
    d_layers=1,
    factor=5
)
```

**Rust**:
```rust
use neuro_divergent::models::Informer;

let model = Informer::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .n_head(8)
    .e_layers(2)
    .d_layers(1)
    .factor(5)
    .build()?;
```

## Data Processing Functions

### DataFrame Operations

| Python (pandas) | Rust (polars) | Notes |
|-----------------|---------------|-------|
| `df.groupby('unique_id')` | `df.group_by("unique_id")` | Method name difference |
| `df.sort_values('ds')` | `df.sort("ds")` | Simpler API |
| `df.reset_index()` | Not needed | polars doesn't have index |
| `df.fillna(method='ffill')` | `df.fill_null(FillNullStrategy::Forward)` | Explicit strategy |
| `df.dropna()` | `df.drop_nulls()` | Method name |
| `df.to_csv('file.csv')` | `df.write_csv("file.csv")?` | Error handling |
| `pd.read_csv('file.csv')` | `LazyFrame::scan_csv("file.csv", Default::default())?` | Lazy loading |

### Data Validation

**Python**:
```python
def validate_data(df):
    assert 'unique_id' in df.columns
    assert 'ds' in df.columns
    assert 'y' in df.columns
    assert df['y'].notna().all()
    return True
```

**Rust**:
```rust
fn validate_data(df: &DataFrame) -> anyhow::Result<()> {
    let required_cols = ["unique_id", "ds", "y"];
    for col in required_cols {
        if !df.get_column_names().contains(&col) {
            anyhow::bail!("Missing required column: {}", col);
        }
    }
    
    let y_null_count = df.column("y")?.null_count();
    if y_null_count > 0 {
        anyhow::bail!("Column 'y' contains {} null values", y_null_count);
    }
    
    Ok(())
}
```

## Error Handling Patterns

### Python Error Handling
```python
try:
    model = LSTM(h=12, input_size=24)
    model.fit(df)
    forecasts = model.predict()
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Training failed: {e}")
```

### Rust Error Handling
```rust
use anyhow::Result;

fn train_model(df: DataFrame) -> Result<DataFrame> {
    let model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .build()?;
    
    let mut trained_model = model.fit(df)?;
    let forecasts = trained_model.predict()?;
    Ok(forecasts)
}

// Usage
match train_model(df) {
    Ok(forecasts) => println!("Training successful"),
    Err(e) => eprintln!("Training failed: {}", e),
}
```

## Type Conversions

### Common Type Mappings

| Python Type | Rust Type | Notes |
|-------------|-----------|-------|
| `int` | `usize` / `i32` / `i64` | Context dependent |
| `float` | `f32` / `f64` | Usually `f64` |
| `str` | `String` / `&str` | Owned vs borrowed |
| `bool` | `bool` | Same |
| `List[int]` | `Vec<usize>` | Generic vectors |
| `Dict[str, Any]` | `HashMap<String, Value>` | JSON-like values |
| `None` | `Option<T>` | Explicit optionality |
| `pd.DataFrame` | `polars::DataFrame` | Different libraries |
| `np.ndarray` | `ndarray::Array` | Optional dependency |

### Frequency Mappings

| Python | Rust | Notes |
|--------|------|-------|
| `'D'` | `Frequency::Daily` | Daily frequency |
| `'H'` | `Frequency::Hourly` | Hourly frequency |
| `'M'` | `Frequency::Monthly` | Monthly frequency |
| `'W'` | `Frequency::Weekly` | Weekly frequency |
| `'Y'` | `Frequency::Yearly` | Yearly frequency |
| `'Q'` | `Frequency::Quarterly` | Quarterly frequency |
| `'B'` | `Frequency::BusinessDaily` | Business days |
| `'T'` | `Frequency::Minutely` | Minutes |
| `'S'` | `Frequency::Secondly` | Seconds |

## Advanced Features

### Cross-Validation

**Python**:
```python
cv_results = nf.cross_validation(
    df=df,
    n_windows=3,
    h=12,
    step_size=1,
    fitted=True
)
```

**Rust**:
```rust
let cv_results = nf.cross_validation(CrossValidationConfig {
    data: df,
    n_windows: 3,
    horizon: 12,
    step_size: 1,
    fitted: true,
})?;
```

### Hyperparameter Tuning

**Python**:
```python
from neuralforecast.auto import AutoLSTM

auto_model = AutoLSTM(h=12, config={'hidden_size': [64, 128, 256]})
auto_model.fit(df)
```

**Rust**:
```rust
use neuro_divergent::auto::AutoLSTM;

let auto_model = AutoLSTM::builder()
    .horizon(12)
    .hidden_size_options(vec![64, 128, 256])
    .build()?;
auto_model.fit(df)?;
```

### Ensemble Methods

**Python**:
```python
models = [LSTM(h=12), NBEATS(h=12), TFT(h=12)]
nf = NeuralForecast(models=models, freq='D')
```

**Rust**:
```rust
let models: Vec<Box<dyn Model>> = vec![
    Box::new(LSTM::builder().horizon(12).build()?),
    Box::new(NBEATS::builder().horizon(12).build()?),
    Box::new(TFT::builder().horizon(12).build()?),
];

let nf = NeuralForecast::builder()
    .with_models(models)
    .with_frequency(Frequency::Daily)
    .build()?;
```

## Migration Utilities

### Automatic Conversion Functions

```rust
use neuro_divergent::migration::{convert_freq, convert_params};

// Convert Python frequency strings to Rust enums
let freq = convert_freq("D")?; // Returns Frequency::Daily

// Convert Python parameter dictionaries to Rust structs
let params = convert_params(python_dict)?;
```

### Validation Helpers

```rust
use neuro_divergent::migration::validate_conversion;

// Validate that Rust model produces equivalent results
validate_conversion(python_results, rust_results, tolerance=1e-6)?;
```

---

**Next**: Continue to [Code Conversion](code-conversion.md) for comprehensive code migration examples.