# Reference Data for Accuracy Validation

This directory contains reference outputs from Python's NeuralForecast library used to validate the accuracy of the Rust implementation.

## Generating Reference Data

To generate or update the reference data, run:

```bash
# From the neuro-divergent root directory
python scripts/generate_reference_data.py

# To generate data for specific models only
python scripts/generate_reference_data.py --models LSTM RNN GRU

# For quick testing with fewer iterations
python scripts/generate_reference_data.py --quick
```

## File Structure

- `train_data.csv` - Training dataset used for all models
- `test_data.csv` - Test dataset for evaluation
- `*_reference.json` - Model-specific reference outputs containing:
  - Model configuration
  - Loss values for different loss functions
  - Predictions on train and test sets
  - Evaluation metrics
  - Gradient information (when available)
- `edge_cases/` - Reference data for edge case scenarios
- `reference_summary.json` - Summary of all reference data generation

## Data Format

Each reference JSON file contains:
```json
{
  "model": "ModelName",
  "config": { ... },
  "losses": {
    "MAE": 0.123,
    "MSE": 0.456,
    ...
  },
  "predictions": {
    "MAE": {
      "train": { ... },
      "test": { ... }
    },
    ...
  },
  "metrics": {
    "MAE": {
      "mae": 0.123,
      "mse": 0.456,
      "rmse": 0.675,
      "mape": 1.23,
      "smape": 2.34
    },
    ...
  }
}
```

## Requirements

To generate reference data, you need:
- Python 3.8+
- NeuralForecast: `pip install neuralforecast`
- NumPy, Pandas: `pip install numpy pandas`

## Notes

- Reference data is generated with fixed random seeds for reproducibility
- All models use the same datasets for fair comparison
- Edge cases test numerical stability and error handling
- The data is versioned with the test suite