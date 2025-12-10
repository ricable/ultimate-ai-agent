"""
Advanced RTB Template Processor
Supports conditional logic ($cond), evaluation ($eval), and custom Python functions
with cognitive automation capabilities
"""

import ast
import operator
import json
import importlib.util
import sys
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from rtb_schema import (
    RTBTemplate, CustomFunction, ConditionalOperator,
    EvaluationOperator, RTBMeta
)


class SafeExpressionEvaluator:
    """Safe expression evaluator using AST parsing"""

    # Allowed operators and functions
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.FloorDiv: operator.floordiv,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: operator.contains,
        ast.NotIn: lambda x, y: not operator.contains(x, y),
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        'len': len,
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'sum': sum,
        'any': any,
        'all': all,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        # Math functions
        'sqrt': np.sqrt,
        'log': np.log,
        'log10': np.log10,
        'exp': np.exp,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        # String functions
        'lower': str.lower,
        'upper': str.upper,
        'strip': str.strip,
        'split': str.split,
        'join': str.join,
        # Date functions
        'now': datetime.now,
        'today': datetime.today,
    }

    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.context.update(self.ALLOWED_FUNCTIONS)

    def evaluate(self, expression: str) -> Any:
        """Safely evaluate an expression using AST"""
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.context:
                return self.context[node.id]
            raise NameError(f"Name '{node.id}' is not defined")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = self.ALLOWED_OPERATORS.get(type(node.op))
            if op_func:
                return op_func(operand)
            raise ValueError(f"Unsupported unary operator {type(node.op)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self.ALLOWED_OPERATORS.get(type(node.op))
            if op_func:
                return op_func(left, right)
            raise ValueError(f"Unsupported binary operator {type(node.op)}")
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, right in zip(node.ops, node.comparators):
                right_val = self._eval_node(right)
                op_func = self.ALLOWED_OPERATORS.get(type(op))
                if not op_func:
                    raise ValueError(f"Unsupported comparison operator {type(op)}")
                if not op_func(left, right_val):
                    return False
                left = right_val
            return True
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            index = self._eval_node(node.slice)
            return value[index]
        elif isinstance(node, ast.Attribute):
            value = self._eval_node(node.value)
            return getattr(value, node.attr)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")


class RTBCustomFunctionExecutor:
    """Execute custom Python functions in a sandboxed environment"""

    def __init__(self):
        self.functions = {}
        self.function_cache = {}
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }

    def register_function(self, func_def: CustomFunction) -> None:
        """Register a custom function from template definition"""
        # Build function code
        func_code = self._build_function_code(func_def)

        # Create function in safe namespace
        safe_globals = self._get_safe_globals()

        try:
            exec(func_code, safe_globals)
            if func_def.name in safe_globals:
                self.functions[func_def.name] = safe_globals[func_def.name]
        except Exception as e:
            raise ValueError(f"Failed to register function '{func_def.name}': {e}")

    def _build_function_code(self, func_def: CustomFunction) -> str:
        """Build Python code for a custom function"""
        args_str = ', '.join(func_def.args)
        func_def_lines = func_def.body

        # Check if function returns a value
        has_return = any('return ' in line for line in func_def_lines)
        if not has_return and func_def_lines:
            # Add return statement if missing
            last_line = func_def_lines[-1].strip()
            if not last_line.startswith('return '):
                func_def_lines = func_def_lines[:-1] + [f"return {last_line}"]

        # Indent function body
        indented_body = '\n'.join(f"    {line}" for line in func_def_lines)

        # Combine into complete function
        func_code = f"def {func_def.name}({args_str}):\n{indented_body}"

        return func_code

    def _get_safe_globals(self) -> Dict[str, Any]:
        """Get safe global namespace for function execution"""
        safe_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'set': set,
                'tuple': tuple, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'min': min, 'max': max, 'abs': abs, 'round': round,
                'sum': sum, 'any': any, 'all': all, 'sorted': sorted,
                'reversed': reversed,
                # Math functions
                'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10,
                'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                'tan': np.tan, 'pi': np.pi, 'e': np.e,
                # String functions
                'lower': str.lower, 'upper': str.upper,
                'strip': str.strip, 'split': str.split,
                'join': str.join, 'replace': str.replace,
                # Date functions
                'datetime': datetime,
            },
            # Utility modules
            'np': np,
            'pd': pd,
            # Constants
            'TRUE': True,
            'FALSE': False,
            'NULL': None,
        }

        return safe_globals

    def execute_function(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered custom function"""
        self.execution_stats['total_executions'] += 1

        if name not in self.functions:
            self.execution_stats['failed_executions'] += 1
            raise ValueError(f"Function '{name}' is not registered")

        try:
            # Check cache for pure functions
            cache_key = None
            if args and not kwargs:
                cache_key = (name, args)
                if cache_key in self.function_cache:
                    self.execution_stats['successful_executions'] += 1
                    return self.function_cache[cache_key]

            # Execute function
            result = self.functions[name](*args, **kwargs)

            # Cache result for pure functions
            if cache_key and self._is_pure_function(name):
                self.function_cache[cache_key] = result

            self.execution_stats['successful_executions'] += 1
            return result

        except Exception as e:
            self.execution_stats['failed_executions'] += 1
            raise RuntimeError(f"Failed to execute function '{name}': {e}")

    def _is_pure_function(self, name: str) -> bool:
        """Check if a function is pure (same input always produces same output)"""
        # Simple heuristic: functions without side effects
        pure_functions = {
            'calculate', 'compute', 'get', 'find', 'check',
            'validate', 'convert', 'transform', 'calculateOffset'
        }
        return any(keyword in name.lower() for keyword in pure_functions)


class RTBConditionalProcessor:
    """Process conditional logic ($cond) and evaluation ($eval) operators"""

    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.evaluator = SafeExpressionEvaluator(context)
        self.condition_cache = {}

    def process_conditionals(self, conditionals: Dict[str, ConditionalOperator]) -> Dict[str, Any]:
        """Process all conditional operators"""
        result = {}

        for field, cond_op in conditionals.items():
            try:
                # Check cache first
                cache_key = (field, cond_op.condition)
                if cache_key in self.condition_cache:
                    condition_result = self.condition_cache[cache_key]
                else:
                    condition_result = self.evaluator.evaluate(cond_op.condition)
                    self.condition_cache[cache_key] = condition_result

                # Apply conditional logic
                if condition_result:
                    result[field] = self._resolve_value(cond_op.then_value)
                elif cond_op.else_value != "__ignore__":
                    result[field] = self._resolve_value(cond_op.else_value)

            except Exception as e:
                print(f"Warning: Failed to process conditional for '{field}': {e}")
                # Skip or use default
                if cond_op.else_value != "__ignore__":
                    result[field] = self._resolve_value(cond_op.else_value)

        return result

    def process_evaluations(self, evaluations: Dict[str, EvaluationOperator],
                          function_executor: RTBCustomFunctionExecutor) -> Dict[str, Any]:
        """Process all evaluation operators"""
        result = {}

        for field, eval_op in evaluations.items():
            try:
                # Prepare arguments
                args = []
                kwargs = {}

                if eval_op.parameters:
                    # Handle positional and keyword arguments
                    if 'args' in eval_op.parameters:
                        args = [
                            self._resolve_value(arg)
                            for arg in eval_op.parameters['args']
                        ]
                    if 'kwargs' in eval_op.parameters:
                        kwargs = {
                            k: self._resolve_value(v)
                            for k, v in eval_op.parameters['kwargs'].items()
                        }

                # Execute function
                func_result = function_executor.execute_function(
                    eval_op.function, *args, **kwargs
                )
                result[field] = func_result

            except Exception as e:
                print(f"Warning: Failed to evaluate '{eval_op.function}' for '{field}': {e}")

        return result

    def _resolve_value(self, value: Any) -> Any:
        """Resolve a value, handling template variables"""
        if isinstance(value, str):
            # Check if it's a template variable
            if value in self.context:
                return self.context[value]
            # Check if it's an expression
            if value.startswith('$') and value.endswith('$'):
                expr = value[1:-1]
                return self.evaluator.evaluate(expr)
        return value


class RTBTemplateProcessor:
    """Advanced RTB Template Processor with full cognitive automation support"""

    def __init__(self, template: RTBTemplate, enable_caching: bool = True):
        self.template = template
        self.enable_caching = enable_caching

        # Initialize processors
        self.function_executor = RTBCustomFunctionExecutor()
        self.conditional_processor = RTBConditionalProcessor({})

        # Register custom functions
        if template.custom_functions:
            for func_def in template.custom_functions:
                self.function_executor.register_function(func_def)

        # Processing metrics
        self.metrics = {
            'templates_processed': 0,
            'conditions_processed': 0,
            'functions_executed': 0,
            'processing_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Cache for processed templates
        self.template_cache = {} if enable_caching else None

    def process_template(self, context: Optional[Dict[str, Any]] = None,
                         optimize: bool = True) -> Dict[str, Any]:
        """Process RTB template with given context and return JSON configuration"""
        import time
        start_time = time.time()

        # Merge context with template defaults
        merged_context = self._merge_context(context or {})

        # Update conditional processor context
        self.conditional_processor.context = merged_context
        self.conditional_processor.evaluator.context = merged_context

        # Check cache
        cache_key = self._get_cache_key(merged_context)
        if self.enable_caching and cache_key in self.template_cache:
            self.metrics['cache_hits'] += 1
            return self.template_cache[cache_key]

        self.metrics['cache_misses'] += 1

        # Extract static configuration
        result = self._extract_static_config()

        # Process conditional logic
        if self.template.conditional_logic:
            conditional_result = self.conditional_processor.process_conditionals(
                self.template.conditional_logic
            )
            result.update(conditional_result)
            self.metrics['conditions_processed'] = len(self.template.conditional_logic)

        # Process evaluation logic
        if self.template.evaluation_logic:
            evaluation_result = self.conditional_processor.process_evaluations(
                self.template.evaluation_logic,
                self.function_executor
            )
            result.update(evaluation_result)
            self.metrics['functions_executed'] = len(self.template.evaluation_logic)

        # Apply optimizations if enabled
        if optimize:
            result = self._optimize_configuration(result)

        # Add metadata
        if self.template.meta:
            result['$meta'] = self.template.meta.dict()
            result['$meta']['processed_at'] = datetime.now().isoformat()

        # Cache result
        if self.enable_caching:
            self.template_cache[cache_key] = result

        # Update metrics
        self.metrics['templates_processed'] += 1
        self.metrics['processing_time_ms'] = int((time.time() - start_time) * 1000)

        return result

    def process_template_batch(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process template for multiple contexts in batch"""
        results = []
        for context in contexts:
            result = self.process_template(context)
            results.append(result)
        return results

    def generate_json(self, context: Optional[Dict[str, Any]] = None,
                      optimize: bool = True, indent: int = 2) -> str:
        """Generate JSON configuration from template"""
        result = self.process_template(context, optimize)
        return json.dumps(result, indent=indent, ensure_ascii=False, cls=RTBJSONEncoder)

    def _merge_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge context with template defaults"""
        merged = {}

        # Add template defaults
        if self.template.meta:
            merged['template_version'] = self.template.meta.version
            merged['template_author'] = self.template.meta.author

        # Add network element defaults
        if self.template.gnbCucp:
            merged['gnb_name'] = self.template.gnbCucp.gNBCUName
            merged['gnb_id'] = self.template.gnbCucp.gNBId

        # Override with provided context
        merged.update(context)

        return merged

    def _extract_static_config(self) -> Dict[str, Any]:
        """Extract static configuration from template"""
        config = {}

        # Convert models to dictionaries
        if self.template.managedElement:
            config['managedElement'] = self.template.managedElement.dict()

        if self.template.gnbCucp:
            config['gnbCucp'] = self.template.gnbCucp.dict()

        if self.template.eUtranCells:
            config['eUtranCells'] = [cell.dict() for cell in self.template.eUtranCells]

        if self.template.nrCells:
            config['nrCells'] = [cell.dict() for cell in self.template.nrCells]

        if self.template.qciProfiles:
            config['qciProfiles'] = [qci.dict() for qci in self.template.qciProfiles]

        if self.template.neighborRelations:
            config['neighborRelations'] = [rel.dict() for rel in self.template.neighborRelations]

        if self.template.anrFunction:
            config['anrFunction'] = self.template.anrFunction.dict()

        return config

    def _optimize_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cognitive optimizations to configuration"""
        # Remove empty values
        config = {k: v for k, v in config.items() if v is not None and v != {}}

        # Optimize nested structures
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = {k: v for k, v in value.items() if v is not None}
            elif isinstance(value, list):
                config[key] = [item for item in value if item is not None]

        return config

    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for context"""
        import hashlib
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        metrics = self.metrics.copy()
        metrics.update(self.function_executor.execution_stats)
        metrics['cache_size'] = len(self.template_cache) if self.template_cache else 0
        return metrics

    def clear_cache(self) -> None:
        """Clear all caches"""
        if self.template_cache:
            self.template_cache.clear()
        self.function_executor.function_cache.clear()
        self.conditional_processor.condition_cache.clear()


class RTBJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for RTB configurations"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# PREDEFINED CUSTOM FUNCTIONS
# ============================================================================

class RTBPredefinedFunctions:
    """Collection of predefined custom functions for RTB processing"""

    @staticmethod
    def calculate_rsrp_offset(target_rsrp: int, cell_type: str,
                           environment: str = "urban") -> int:
        """Calculate RSRP offset based on cell type and environment"""
        base_offset = {
            "urban": {"macro": 2, "pico": -2, "femto": -4},
            "suburban": {"macro": 1, "pico": -1, "femto": -3},
            "rural": {"macro": 0, "pico": -2, "femto": -4}
        }

        if environment in base_offset and cell_type in base_offset[environment]:
            return target_rsrp + base_offset[environment][cell_type]
        return target_rsrp

    @staticmethod
    def optimize_power_control(load: float, distance: float,
                             max_power: int = 43) -> int:
        """Optimize power control based on load and distance"""
        # Simple power control algorithm
        load_factor = 1.0 - (load / 100.0)
        distance_factor = max(0.5, 1.0 - (distance / 10.0))

        optimized_power = int(max_power * load_factor * distance_factor)
        return max(20, min(optimized_power, max_power))

    @staticmethod
    def determine_qci(service_type: str, priority: int,
                      gbr_required: bool = False) -> int:
        """Determine appropriate QCI based on service requirements"""
        qci_mapping = {
            ("conversational", 1, True): 1,
            ("conversational", 2, True): 2,
            ("conversational", 3, False): 3,
            ("streaming", 4, True): 4,
            ("streaming", 5, False): 5,
            ("interactive", 6, False): 6,
            ("interactive", 7, False): 7,
            ("background", 8, False): 8,
            ("background", 9, False): 9,
        }

        key = (service_type.lower(), priority, gbr_required)
        return qci_mapping.get(key, 9)

    @staticmethod
    def calculate_handover_margin(cell_load: float, neighbor_load: float,
                                 user_velocity: float) -> int:
        """Calculate handover margin based on load and user velocity"""
        load_balance_factor = (cell_load - neighbor_load) / 100.0
        velocity_factor = min(1.0, user_velocity / 120.0)  # 120 km/h max

        margin = int(3 + load_balance_factor * 2 + velocity_factor * 5)
        return max(0, min(margin, 15))

    @staticmethod
    def get_optimal_bandwidth(expected_users: int,
                           service_mix: Dict[str, float]) -> str:
        """Get optimal bandwidth based on user count and service mix"""
        total_bandwidth = expected_users * 10  # MHz per user estimate

        if total_bandwidth < 20:
            return "5"
        elif total_bandwidth < 40:
            return "10"
        elif total_bandwidth < 80:
            return "20"
        elif total_bandwidth < 160:
            return "40"
        elif total_bandwidth < 320:
            return "80"
        else:
            return "100"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_template_from_file(file_path: Union[str, Path]) -> RTBTemplate:
    """Load RTB template from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return RTBTemplate(**data)

def save_template_to_file(template: RTBTemplate,
                          file_path: Union[str, Path],
                          indent: int = 2) -> None:
    """Save RTB template to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(template.dict(), f, indent=indent, ensure_ascii=False,
                  cls=RTBJSONEncoder)

def create_template_from_config(config: Dict[str, Any]) -> RTBTemplate:
    """Create RTBTemplate from configuration dictionary"""
    # Extract metadata
    meta = None
    if '$meta' in config:
        meta = RTBMeta(**config['$meta'])
        del config['$meta']

    # Extract custom functions
    custom_funcs = None
    if '$custom' in config:
        custom_funcs = [CustomFunction(**func) for func in config['$custom']]
        del config['$custom']

    # Extract conditional logic
    conditionals = None
    if '$cond' in config:
        conditionals = {
            k: ConditionalOperator(condition=v.get('if', ''),
                                  then_value=v.get('then'),
                                  else_value=v.get('else', '__ignore__'))
            for k, v in config['$cond'].items()
        }
        del config['$cond']

    # Extract evaluation logic
    evaluations = None
    if '$eval' in config:
        evaluations = {
            k: EvaluationOperator(function=v.get('eval'),
                                parameters=v)
            for k, v in config['$eval'].items()
        }
        del config['$eval']

    # Create template
    template = RTBTemplate(
        meta=meta,
        custom_functions=custom_funcs,
        conditional_logic=conditionals,
        evaluation_logic=evaluations,
        **config
    )

    return template