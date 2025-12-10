#!/usr/bin/env python3
"""
Generate reference data from Python NeuralForecast for accuracy validation.

This script generates predictions and training outputs from all NeuralForecast models
to serve as ground truth for validating the Rust implementation.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import NeuralForecast - provide instructions if not available
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        RNN, LSTM, GRU, MLP, MLPMultivariate,
        TCN, BiTCN, TimesNet,
        TFT, Informer, Autoformer, FEDformer, PatchTST, iTransformer,
        NBEATS, NBEATSx, NHITS,
        DLinear, NLinear,
        TiDE, DeepAR, 
        TSMixer, TSMixerx, StemGNN, TimeLLM
    )
    from neuralforecast.losses.pytorch import (
        MAE, MSE, RMSE, MAPE, SMAPE, MASE,
        QuantileLoss, MQLoss, DistributionLoss,
        HuberLoss, TukeyLoss
    )
except ImportError:
    logger.error("NeuralForecast not installed. Please install with: pip install neuralforecast")
    raise

# Standard test datasets
def generate_synthetic_data(n_series: int = 3, n_timesteps: int = 200, 
                          horizon: int = 24, freq: str = 'H') -> pd.DataFrame:
    """Generate synthetic time series data for testing."""
    logger.info(f"Generating synthetic data: {n_series} series, {n_timesteps} timesteps")
    
    data_list = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_series):
        # Generate timestamps
        timestamps = pd.date_range(start=base_date, periods=n_timesteps, freq=freq)
        
        # Generate different patterns for each series
        if i == 0:
            # Sinusoidal pattern with trend
            trend = np.linspace(100, 150, n_timesteps)
            seasonal = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
            noise = np.random.normal(0, 5, n_timesteps)
            values = trend + seasonal + noise
        elif i == 1:
            # Multiple seasonalities
            daily = 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
            weekly = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))
            noise = np.random.normal(0, 3, n_timesteps)
            values = 100 + daily + weekly + noise
        else:
            # Random walk with drift
            drift = 0.1
            noise = np.random.normal(0, 2, n_timesteps)
            values = np.cumsum(drift + noise) + 100
        
        # Create dataframe for this series
        series_df = pd.DataFrame({
            'unique_id': f'series_{i}',
            'ds': timestamps,
            'y': values
        })
        
        # Add exogenous variables
        series_df['exog_1'] = np.random.randn(n_timesteps) * 10 + 50
        series_df['exog_2'] = np.sin(2 * np.pi * np.arange(n_timesteps) / 12) * 5
        
        data_list.append(series_df)
    
    return pd.concat(data_list, ignore_index=True)

def generate_edge_case_data() -> List[pd.DataFrame]:
    """Generate edge case datasets for robustness testing."""
    logger.info("Generating edge case datasets")
    
    edge_cases = []
    base_date = datetime(2023, 1, 1)
    
    # 1. Empty dataset (should handle gracefully)
    empty_df = pd.DataFrame(columns=['unique_id', 'ds', 'y'])
    edge_cases.append(('empty', empty_df))
    
    # 2. Single data point
    single_point = pd.DataFrame({
        'unique_id': ['single'],
        'ds': [base_date],
        'y': [100.0]
    })
    edge_cases.append(('single_point', single_point))
    
    # 3. Constant values
    n_const = 100
    constant_values = pd.DataFrame({
        'unique_id': ['constant'] * n_const,
        'ds': pd.date_range(base_date, periods=n_const, freq='H'),
        'y': [42.0] * n_const
    })
    edge_cases.append(('constant', constant_values))
    
    # 4. Extreme values
    n_extreme = 50
    extreme_values = pd.DataFrame({
        'unique_id': ['extreme'] * n_extreme,
        'ds': pd.date_range(base_date, periods=n_extreme, freq='H'),
        'y': [1e10 if i % 10 == 0 else 1e-10 for i in range(n_extreme)]
    })
    edge_cases.append(('extreme', extreme_values))
    
    # 5. Missing values
    n_missing = 100
    y_values = np.random.randn(n_missing) * 10 + 100
    # Introduce 20% missing values
    missing_mask = np.random.rand(n_missing) < 0.2
    y_values[missing_mask] = np.nan
    missing_values = pd.DataFrame({
        'unique_id': ['missing'] * n_missing,
        'ds': pd.date_range(base_date, periods=n_missing, freq='H'),
        'y': y_values
    })
    edge_cases.append(('missing', missing_values))
    
    # 6. Perfect predictions (zero error case)
    n_perfect = 50
    perfect_values = pd.DataFrame({
        'unique_id': ['perfect'] * n_perfect,
        'ds': pd.date_range(base_date, periods=n_perfect, freq='H'),
        'y': np.arange(n_perfect, dtype=float)
    })
    edge_cases.append(('perfect', perfect_values))
    
    return edge_cases

def create_model_configs() -> Dict[str, Dict[str, Any]]:
    """Create configuration for each model with standardized parameters."""
    # Common parameters for reproducibility
    common_params = {
        'h': 24,  # forecast horizon
        'input_size': 48,  # lookback window
        'random_seed': 42,
        'max_steps': 100,  # Limit training for speed
        'val_check_steps': 10,
        'early_stop_patience_steps': 5
    }
    
    # Model-specific configurations
    configs = {
        # Recurrent models
        'RNN': {**common_params, 'hidden_size': 64, 'n_layers': 2, 'dropout': 0.1},
        'LSTM': {**common_params, 'hidden_size': 64, 'n_layers': 2, 'dropout': 0.1},
        'GRU': {**common_params, 'hidden_size': 64, 'n_layers': 2, 'dropout': 0.1},
        
        # Simple feedforward
        'MLP': {**common_params, 'num_layers': 3, 'hidden_size': 128},
        'MLPMultivariate': {**common_params, 'num_layers': 3, 'hidden_size': 128, 'n_series': 3},
        
        # Convolutional
        'TCN': {**common_params, 'kernel_size': 3, 'dilations': [1, 2, 4, 8], 'n_filters': 64},
        'BiTCN': {**common_params, 'kernel_size': 3, 'dilations': [1, 2, 4, 8], 'n_filters': 64},
        'TimesNet': {**common_params, 'd_model': 64, 'd_ff': 128, 'top_k': 3, 'num_kernels': 3},
        
        # Transformer-based
        'TFT': {**common_params, 'hidden_size': 128, 'n_head': 4, 'attn_dropout': 0.1},
        'Informer': {**common_params, 'hidden_size': 128, 'n_head': 4, 'conv_hidden_size': 64},
        'Autoformer': {**common_params, 'hidden_size': 128, 'n_head': 4, 'conv_hidden_size': 64},
        'FEDformer': {**common_params, 'hidden_size': 128, 'n_head': 4, 'conv_hidden_size': 64},
        'PatchTST': {**common_params, 'patch_len': 16, 'stride': 8, 'd_model': 128},
        'iTransformer': {**common_params, 'hidden_size': 128, 'n_head': 4, 'n_series': 3},
        
        # N-BEATS family
        'NBEATS': {**common_params, 'n_blocks': [1, 1, 1], 'mlp_units': [[128, 128]]*3, 
                   'n_polynomials': 3, 'n_harmonics': 3},
        'NBEATSx': {**common_params, 'n_blocks': [1, 1, 1], 'mlp_units': [[128, 128]]*3},
        'NHITS': {**common_params, 'n_blocks': [1, 1, 1], 'mlp_units': [[128, 128]]*3,
                  'n_freq_downsample': [24, 12, 1]},
        
        # Linear models
        'DLinear': {**common_params, 'kernel_size': 25},
        'NLinear': {**common_params},
        
        # Advanced specialized
        'TiDE': {**common_params, 'hidden_size': 128, 'encoder_layers': 2, 'decoder_layers': 2},
        'DeepAR': {**common_params, 'hidden_size': 64, 'n_layers': 2, 'dropout': 0.1},
        
        # Mixing models
        'TSMixer': {**common_params, 'n_block': 2, 'dropout': 0.1, 'ff_dim': 64},
        'TSMixerx': {**common_params, 'n_block': 2, 'dropout': 0.1, 'ff_dim': 64},
        
        # Graph and LLM (if available)
        'StemGNN': {**common_params, 'n_nodes': 3, 'cycle': 24, 'd_model': 64},
        'TimeLLM': {**common_params, 'd_model': 64, 'd_ff': 128, 'patch_len': 16}
    }
    
    return configs

def train_and_evaluate_model(model_class, config: Dict[str, Any], 
                           train_df: pd.DataFrame, test_df: pd.DataFrame,
                           loss_functions: List[str] = ['MAE', 'MSE']) -> Dict[str, Any]:
    """Train a model and collect detailed outputs for validation."""
    model_name = model_class.__name__
    logger.info(f"Training {model_name} with config: {config}")
    
    results = {
        'model': model_name,
        'config': config,
        'losses': {},
        'predictions': {},
        'metrics': {},
        'gradients': {},
        'training_history': []
    }
    
    # Test with different loss functions
    for loss_name in loss_functions:
        try:
            # Get loss function
            if loss_name == 'MAE':
                loss = MAE()
            elif loss_name == 'MSE':
                loss = MSE()
            elif loss_name == 'RMSE':
                loss = RMSE()
            elif loss_name == 'MAPE':
                loss = MAPE()
            elif loss_name == 'SMAPE':
                loss = SMAPE()
            else:
                loss = MAE()  # Default
            
            # Update config with loss
            model_config = {**config, 'loss': loss}
            
            # Initialize model
            model = model_class(**model_config)
            
            # Create NeuralForecast object
            nf = NeuralForecast(models=[model], freq='H')
            
            # Train model
            logger.info(f"Training {model_name} with {loss_name} loss")
            nf.fit(df=train_df, val_size=24)
            
            # Generate predictions
            predictions = nf.predict()
            
            # Evaluate on test set
            if len(test_df) > 0:
                test_predictions = nf.predict(df=test_df)
                results['predictions'][loss_name] = {
                    'train': predictions.to_dict(),
                    'test': test_predictions.to_dict()
                }
            else:
                results['predictions'][loss_name] = {
                    'train': predictions.to_dict()
                }
            
            # Calculate metrics
            y_true = test_df['y'].values if len(test_df) > 0 else train_df['y'].values[-24:]
            y_pred = predictions[model_name].values if len(test_df) == 0 else test_predictions[model_name].values
            
            # Calculate various metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Handle division by zero for percentage errors
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                smape = np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / 
                               (np.abs(y_pred[mask]) + np.abs(y_true[mask]))) * 100
            else:
                mape = np.nan
                smape = np.nan
            
            results['metrics'][loss_name] = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'smape': float(smape)
            }
            
            # Store loss value
            results['losses'][loss_name] = float(mae)  # Store final loss value
            
            logger.info(f"Completed {model_name} with {loss_name}: MAE={mae:.6f}")
            
        except Exception as e:
            logger.warning(f"Failed to train {model_name} with {loss_name}: {str(e)}")
            results['metrics'][loss_name] = {'error': str(e)}
    
    return results

def save_reference_data(results: Dict[str, Any], output_dir: str):
    """Save reference data in JSON format for Rust tests."""
    model_name = results['model']
    output_path = os.path.join(output_dir, f"{model_name.lower()}_reference.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved reference data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate NeuralForecast reference data')
    parser.add_argument('--output-dir', type=str, 
                       default='/workspaces/ruv-FANN/neuro-divergent/tests/accuracy/comparison_data',
                       help='Output directory for reference data')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to test (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer iterations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate datasets
    logger.info("Generating datasets...")
    train_df = generate_synthetic_data(n_series=3, n_timesteps=200, horizon=24)
    test_df = generate_synthetic_data(n_series=3, n_timesteps=50, horizon=24)
    edge_cases = generate_edge_case_data()
    
    # Save datasets
    train_df.to_csv(os.path.join(args.output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_data.csv'), index=False)
    
    # Save edge case datasets
    for name, df in edge_cases:
        df.to_csv(os.path.join(args.output_dir, f'edge_case_{name}.csv'), index=False)
    
    # Get model configurations
    configs = create_model_configs()
    
    # If quick mode, reduce iterations
    if args.quick:
        for config in configs.values():
            config['max_steps'] = 10
            config['val_check_steps'] = 5
    
    # Define models to test
    model_classes = {
        'RNN': RNN, 'LSTM': LSTM, 'GRU': GRU,
        'MLP': MLP, 'MLPMultivariate': MLPMultivariate,
        'TCN': TCN, 'BiTCN': BiTCN, 'TimesNet': TimesNet,
        'TFT': TFT, 'Informer': Informer, 'Autoformer': Autoformer,
        'FEDformer': FEDformer, 'PatchTST': PatchTST, 'iTransformer': iTransformer,
        'NBEATS': NBEATS, 'NBEATSx': NBEATSx, 'NHITS': NHITS,
        'DLinear': DLinear, 'NLinear': NLinear,
        'TiDE': TiDE, 'DeepAR': DeepAR,
        'TSMixer': TSMixer, 'TSMixerx': TSMixerx,
        'StemGNN': StemGNN, 'TimeLLM': TimeLLM
    }
    
    # Filter models if specified
    if args.models:
        model_classes = {k: v for k, v in model_classes.items() if k in args.models}
    
    # Test each model
    all_results = {}
    for model_name, model_class in model_classes.items():
        if model_name not in configs:
            logger.warning(f"No config found for {model_name}, skipping")
            continue
        
        try:
            # Train and evaluate model
            results = train_and_evaluate_model(
                model_class, 
                configs[model_name],
                train_df,
                test_df,
                loss_functions=['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']
            )
            
            # Save results
            save_reference_data(results, args.output_dir)
            all_results[model_name] = results
            
            # Also test edge cases for selected models
            if model_name in ['MLP', 'LSTM', 'DLinear']:  # Test subset for edge cases
                for edge_name, edge_df in edge_cases:
                    if len(edge_df) > 0:  # Skip empty dataset
                        try:
                            edge_results = train_and_evaluate_model(
                                model_class,
                                configs[model_name],
                                edge_df,
                                pd.DataFrame(),  # No test set for edge cases
                                loss_functions=['MAE']
                            )
                            edge_results['edge_case'] = edge_name
                            save_reference_data(edge_results, 
                                              os.path.join(args.output_dir, 'edge_cases'))
                        except Exception as e:
                            logger.warning(f"Edge case {edge_name} failed for {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_tested': list(all_results.keys()),
        'datasets': {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'edge_cases': [name for name, _ in edge_cases]
        },
        'results_summary': {
            model: {
                'success': 'error' not in results,
                'losses_tested': list(results.get('losses', {}).keys()),
                'metrics': results.get('metrics', {})
            }
            for model, results in all_results.items()
        }
    }
    
    with open(os.path.join(args.output_dir, 'reference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Reference data generation complete. Summary saved to {args.output_dir}/reference_summary.json")

if __name__ == '__main__':
    main()