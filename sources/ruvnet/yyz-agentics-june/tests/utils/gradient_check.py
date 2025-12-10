import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
import warnings


class GradientChecker:
    """Utilities for numerical gradient checking and validation."""
    
    def __init__(self, epsilon: float = 1e-5, tolerance: float = 1e-7):
        """
        Initialize gradient checker.
        
        Args:
            epsilon: Perturbation size for finite differences
            tolerance: Tolerance for gradient comparison
        """
        self.epsilon = epsilon
        self.tolerance = tolerance
        
    def compute_numerical_gradient(self, 
                                 func: Callable,
                                 x: np.ndarray,
                                 verbose: bool = False) -> np.ndarray:
        """
        Compute numerical gradient using central finite differences.
        
        Args:
            func: Function that takes x and returns scalar loss
            x: Input array
            verbose: Print progress for large arrays
            
        Returns:
            Numerical gradient array of same shape as x
        """
        grad = np.zeros_like(x, dtype=np.float64)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        total_elements = x.size
        i = 0
        
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx].copy()
            
            # Forward perturbation
            x[idx] = old_value + self.epsilon
            f_plus = func(x)
            
            # Backward perturbation
            x[idx] = old_value - self.epsilon
            f_minus = func(x)
            
            # Central difference
            grad[idx] = (f_plus - f_minus) / (2 * self.epsilon)
            
            # Restore original value
            x[idx] = old_value
            
            if verbose and i % 100 == 0:
                print(f"Gradient check progress: {i}/{total_elements}")
                
            i += 1
            it.iternext()
            
        return grad
        
    def check_gradient(self,
                      func: Callable,
                      grad_func: Callable,
                      x: np.ndarray,
                      verbose: bool = True) -> Tuple[bool, float, Dict]:
        """
        Check analytical gradient against numerical gradient.
        
        Args:
            func: Loss function
            grad_func: Analytical gradient function
            x: Input point
            verbose: Print detailed results
            
        Returns:
            Tuple of (passed, relative_error, detailed_results)
        """
        # Compute gradients
        analytical_grad = grad_func(x)
        numerical_grad = self.compute_numerical_gradient(func, x)
        
        # Compute relative error
        diff = analytical_grad - numerical_grad
        norm_diff = np.linalg.norm(diff)
        norm_sum = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
        
        if norm_sum == 0:
            relative_error = 0 if norm_diff == 0 else float('inf')
        else:
            relative_error = norm_diff / norm_sum
            
        passed = relative_error < self.tolerance
        
        # Detailed analysis
        results = {
            'passed': passed,
            'relative_error': relative_error,
            'max_absolute_diff': np.max(np.abs(diff)),
            'mean_absolute_diff': np.mean(np.abs(diff)),
            'analytical_norm': np.linalg.norm(analytical_grad),
            'numerical_norm': np.linalg.norm(numerical_grad),
        }
        
        if verbose:
            print(f"Gradient check {'PASSED' if passed else 'FAILED'}")
            print(f"Relative error: {relative_error:.2e}")
            print(f"Tolerance: {self.tolerance:.2e}")
            print(f"Max absolute difference: {results['max_absolute_diff']:.2e}")
            
        # Find largest differences
        if not passed and verbose:
            flat_idx = np.argmax(np.abs(diff))
            idx = np.unravel_index(flat_idx, diff.shape)
            print(f"\nLargest difference at index {idx}:")
            print(f"Analytical: {analytical_grad[idx]:.6f}")
            print(f"Numerical: {numerical_grad[idx]:.6f}")
            print(f"Difference: {diff[idx]:.2e}")
            
        return passed, relative_error, results
        
    def check_layer_gradients(self,
                            layer_forward: Callable,
                            layer_backward: Callable,
                            input_shape: Tuple,
                            param_shapes: Dict[str, Tuple],
                            verbose: bool = True) -> Dict[str, Dict]:
        """
        Check gradients for a neural network layer.
        
        Args:
            layer_forward: Forward pass function
            layer_backward: Backward pass function
            input_shape: Shape of layer input
            param_shapes: Dictionary of parameter names to shapes
            verbose: Print detailed results
            
        Returns:
            Dictionary of gradient check results per parameter
        """
        # Generate random inputs and parameters
        X = np.random.randn(*input_shape)
        params = {name: np.random.randn(*shape) 
                 for name, shape in param_shapes.items()}
        
        # Forward pass
        output = layer_forward(X, params)
        
        # Random gradient from upstream
        grad_output = np.random.randn(*output.shape)
        
        # Define loss function for numerical gradient
        def loss_wrt_param(param_name):
            def loss_func(param_value):
                params_copy = params.copy()
                params_copy[param_name] = param_value
                out = layer_forward(X, params_copy)
                return np.sum(out * grad_output)
            return loss_func
            
        def loss_wrt_input(x):
            out = layer_forward(x, params)
            return np.sum(out * grad_output)
            
        # Backward pass
        grads = layer_backward(X, params, grad_output)
        
        results = {}
        
        # Check parameter gradients
        for param_name in params:
            if verbose:
                print(f"\nChecking gradient for parameter: {param_name}")
                
            loss_func = loss_wrt_param(param_name)
            analytical_grad = grads.get(f'd{param_name}', grads.get(param_name))
            
            passed, rel_error, details = self.check_gradient(
                loss_func,
                lambda p: analytical_grad,
                params[param_name],
                verbose=verbose
            )
            
            results[param_name] = details
            
        # Check input gradient
        if 'dX' in grads or 'dx' in grads:
            if verbose:
                print(f"\nChecking gradient for input")
                
            analytical_grad = grads.get('dX', grads.get('dx'))
            passed, rel_error, details = self.check_gradient(
                loss_wrt_input,
                lambda x: analytical_grad,
                X,
                verbose=verbose
            )
            
            results['input'] = details
            
        return results


class NumericalStabilityChecker:
    """Check numerical stability of neural network operations."""
    
    @staticmethod
    def check_forward_stability(layer_func: Callable,
                              input_ranges: List[Tuple[float, float]],
                              input_shape: Tuple) -> Dict[str, List]:
        """
        Check if layer produces stable outputs for various input ranges.
        
        Args:
            layer_func: Layer forward function
            input_ranges: List of (min, max) ranges to test
            input_shape: Shape of input tensor
            
        Returns:
            Dictionary of stability test results
        """
        results = {
            'input_range': [],
            'has_nan': [],
            'has_inf': [],
            'output_mean': [],
            'output_std': [],
            'output_min': [],
            'output_max': [],
        }
        
        for min_val, max_val in input_ranges:
            # Generate input in specified range
            if min_val == 0 and max_val == 0:
                X = np.zeros(input_shape)
            else:
                X = np.random.uniform(min_val, max_val, size=input_shape)
                
            # Forward pass
            try:
                output = layer_func(X)
                
                results['input_range'].append((min_val, max_val))
                results['has_nan'].append(np.any(np.isnan(output)))
                results['has_inf'].append(np.any(np.isinf(output)))
                results['output_mean'].append(np.mean(output))
                results['output_std'].append(np.std(output))
                results['output_min'].append(np.min(output))
                results['output_max'].append(np.max(output))
                
            except Exception as e:
                results['input_range'].append((min_val, max_val))
                results['has_nan'].append(True)
                results['has_inf'].append(True)
                results['output_mean'].append(np.nan)
                results['output_std'].append(np.nan)
                results['output_min'].append(np.nan)
                results['output_max'].append(np.nan)
                warnings.warn(f"Layer failed for input range {(min_val, max_val)}: {e}")
                
        return results
        
    @staticmethod
    def check_gradient_stability(grad_func: Callable,
                               loss_scales: List[float],
                               param_shape: Tuple) -> Dict[str, List]:
        """
        Check if gradients remain stable under different loss scales.
        
        Args:
            grad_func: Gradient computation function
            loss_scales: List of loss scaling factors to test
            param_shape: Shape of parameter tensor
            
        Returns:
            Dictionary of gradient stability results
        """
        results = {
            'loss_scale': [],
            'grad_norm': [],
            'grad_mean': [],
            'grad_std': [],
            'has_nan': [],
            'has_inf': [],
        }
        
        base_grad = np.random.randn(*param_shape)
        
        for scale in loss_scales:
            scaled_grad = base_grad * scale
            
            results['loss_scale'].append(scale)
            results['grad_norm'].append(np.linalg.norm(scaled_grad))
            results['grad_mean'].append(np.mean(scaled_grad))
            results['grad_std'].append(np.std(scaled_grad))
            results['has_nan'].append(np.any(np.isnan(scaled_grad)))
            results['has_inf'].append(np.any(np.isinf(scaled_grad)))
            
        return results


def gradient_check_network(network_forward: Callable,
                         network_backward: Callable,
                         input_shape: Tuple,
                         num_samples: int = 5,
                         verbose: bool = True) -> List[Dict]:
    """
    Perform gradient checking on entire neural network.
    
    Args:
        network_forward: Network forward pass function
        network_backward: Network backward pass function
        input_shape: Shape of network input
        num_samples: Number of random samples to test
        verbose: Print detailed results
        
    Returns:
        List of gradient check results for each sample
    """
    checker = GradientChecker()
    results = []
    
    for i in range(num_samples):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Gradient check sample {i+1}/{num_samples}")
            print('='*50)
            
        # Random input
        X = np.random.randn(*input_shape)
        
        # Forward pass
        output = network_forward(X)
        
        # Random target for loss computation
        target = np.random.randn(*output.shape)
        
        # Define loss function
        def loss_func(x):
            out = network_forward(x)
            return np.mean((out - target)**2)  # MSE loss
            
        # Analytical gradient via backprop
        grad_output = 2 * (output - target) / output.size
        analytical_grad = network_backward(X, grad_output)
        
        # Check gradient
        passed, rel_error, details = checker.check_gradient(
            loss_func,
            lambda x: analytical_grad,
            X,
            verbose=verbose
        )
        
        results.append({
            'sample': i,
            'passed': passed,
            'relative_error': rel_error,
            'details': details
        })
        
    # Summary
    if verbose:
        print(f"\n{'='*50}")
        print("GRADIENT CHECK SUMMARY")
        print('='*50)
        
        passed_count = sum(r['passed'] for r in results)
        print(f"Passed: {passed_count}/{num_samples}")
        
        avg_error = np.mean([r['relative_error'] for r in results])
        print(f"Average relative error: {avg_error:.2e}")
        
    return results


if __name__ == '__main__':
    # Example usage
    print("Gradient checking utilities loaded successfully")