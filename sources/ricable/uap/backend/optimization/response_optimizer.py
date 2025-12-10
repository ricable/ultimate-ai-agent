# File: backend/optimization/response_optimizer.py
"""
API response optimization for UAP platform.
Provides compression, field selection, pagination, and response batching.
"""

import gzip
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict
import mimetypes

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for response compression"""
    enabled: bool = True
    min_size: int = 1024  # Only compress responses larger than 1KB
    compression_level: int = 6  # gzip compression level (1-9)
    compressible_types: Set[str] = None
    
    def __post_init__(self):
        if self.compressible_types is None:
            self.compressible_types = {
                'application/json',
                'text/plain',
                'text/html',
                'text/css',
                'text/javascript',
                'application/javascript',
                'application/xml',
                'text/xml'
            }

@dataclass
class PaginationConfig:
    """Configuration for pagination optimization"""
    default_page_size: int = 50
    max_page_size: int = 1000
    enable_cursor_pagination: bool = True
    enable_total_count: bool = True
    cache_paginated_results: bool = True
    cache_ttl: int = 300  # 5 minutes

class ResponseCompressor:
    """Response compression handler"""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.compression_stats = {
            'total_responses': 0,
            'compressed_responses': 0,
            'bytes_saved': 0,
            'compression_time': 0.0
        }
    
    def should_compress(self, content: Union[str, bytes], content_type: str = None) -> bool:
        """Determine if content should be compressed"""
        if not self.config.enabled:
            return False
        
        # Check size threshold
        content_size = len(content.encode('utf-8') if isinstance(content, str) else content)
        if content_size < self.config.min_size:
            return False
        
        # Check content type
        if content_type and content_type not in self.config.compressible_types:
            return False
        
        return True
    
    def compress_response(self, content: Union[str, bytes], content_type: str = None) -> Dict[str, Any]:
        """Compress response content"""
        start_time = time.time()
        self.compression_stats['total_responses'] += 1
        
        if not self.should_compress(content, content_type):
            return {
                'content': content,
                'compressed': False,
                'original_size': len(content.encode('utf-8') if isinstance(content, str) else content),
                'compressed_size': len(content.encode('utf-8') if isinstance(content, str) else content),
                'compression_ratio': 1.0
            }
        
        try:
            # Convert to bytes if string
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = content
            
            original_size = len(content_bytes)
            
            # Compress content
            compressed_content = gzip.compress(content_bytes, compresslevel=self.config.compression_level)
            compressed_size = len(compressed_content)
            
            compression_time = time.time() - start_time
            self.compression_stats['compressed_responses'] += 1
            self.compression_stats['bytes_saved'] += (original_size - compressed_size)
            self.compression_stats['compression_time'] += compression_time
            
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            return {
                'content': compressed_content,
                'compressed': True,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'compression_time': compression_time
            }
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return {
                'content': content,
                'compressed': False,
                'original_size': len(content.encode('utf-8') if isinstance(content, str) else content),
                'compressed_size': len(content.encode('utf-8') if isinstance(content, str) else content),
                'compression_ratio': 1.0,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        total = self.compression_stats['total_responses']
        compressed = self.compression_stats['compressed_responses']
        compression_rate = (compressed / total * 100) if total > 0 else 0
        
        avg_compression_time = (self.compression_stats['compression_time'] / compressed 
                              if compressed > 0 else 0)
        
        return {
            'total_responses': total,
            'compressed_responses': compressed,
            'compression_rate_percent': round(compression_rate, 2),
            'bytes_saved': self.compression_stats['bytes_saved'],
            'average_compression_time_ms': round(avg_compression_time * 1000, 2),
            'config': {
                'enabled': self.config.enabled,
                'min_size': self.config.min_size,
                'compression_level': self.config.compression_level
            }
        }

class FieldSelector:
    """Field selection for API responses"""
    
    def __init__(self):
        self.selection_stats = defaultdict(int)
    
    def select_fields(self, data: Any, fields: Optional[List[str]] = None, 
                     exclude_fields: Optional[List[str]] = None) -> Any:
        """Select specific fields from response data"""
        if not fields and not exclude_fields:
            return data
        
        if isinstance(data, dict):
            return self._select_dict_fields(data, fields, exclude_fields)
        elif isinstance(data, list):
            return [self.select_fields(item, fields, exclude_fields) for item in data]
        else:
            return data
    
    def _select_dict_fields(self, data: Dict[str, Any], fields: Optional[List[str]], 
                           exclude_fields: Optional[List[str]]) -> Dict[str, Any]:
        """Select fields from dictionary"""
        result = {}
        
        if fields:
            # Include only specified fields
            for field in fields:
                if '.' in field:
                    # Nested field access
                    parts = field.split('.')
                    if parts[0] in data:
                        if parts[0] not in result:
                            result[parts[0]] = {}
                        self._set_nested_field(result[parts[0]], parts[1:], 
                                             self._get_nested_field(data[parts[0]], parts[1:]))
                elif field in data:
                    result[field] = data[field]
                    
                self.selection_stats[f'included_{field}'] += 1
        else:
            # Include all fields
            result = data.copy()
        
        if exclude_fields:
            # Remove excluded fields
            for field in exclude_fields:
                if '.' in field:
                    # Nested field removal
                    parts = field.split('.')
                    if parts[0] in result:
                        self._remove_nested_field(result[parts[0]], parts[1:])
                elif field in result:
                    del result[field]
                    
                self.selection_stats[f'excluded_{field}'] += 1
        
        return result
    
    def _get_nested_field(self, data: Any, path: List[str]) -> Any:
        """Get nested field value"""
        if not path or not isinstance(data, dict):
            return data
        
        key = path[0]
        if key not in data:
            return None
        
        if len(path) == 1:
            return data[key]
        else:
            return self._get_nested_field(data[key], path[1:])
    
    def _set_nested_field(self, data: Dict[str, Any], path: List[str], value: Any):
        """Set nested field value"""
        if not path:
            return
        
        key = path[0]
        if len(path) == 1:
            data[key] = value
        else:
            if key not in data:
                data[key] = {}
            self._set_nested_field(data[key], path[1:], value)
    
    def _remove_nested_field(self, data: Any, path: List[str]):
        """Remove nested field"""
        if not path or not isinstance(data, dict):
            return
        
        key = path[0]
        if len(path) == 1:
            data.pop(key, None)
        else:
            if key in data:
                self._remove_nested_field(data[key], path[1:])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get field selection statistics"""
        return {
            'field_operations': dict(self.selection_stats),
            'total_operations': sum(self.selection_stats.values())
        }

class ResponsePaginator:
    """Response pagination handler"""
    
    def __init__(self, config: PaginationConfig = None):
        self.config = config or PaginationConfig()
        self.pagination_stats = {
            'total_requests': 0,
            'paginated_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def paginate_response(self, data: List[Any], page: int = 1, page_size: Optional[int] = None, 
                         total_count: Optional[int] = None) -> Dict[str, Any]:
        """Paginate response data"""
        self.pagination_stats['total_requests'] += 1
        
        if not isinstance(data, list):
            return {'data': data, 'pagination': None}
        
        # Validate and set page size
        if page_size is None:
            page_size = self.config.default_page_size
        page_size = min(page_size, self.config.max_page_size)
        
        # Calculate pagination
        total_items = total_count if total_count is not None else len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        # Validate page number
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        # Calculate slice indices
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        
        # Slice data
        paginated_data = data[start_index:end_index]
        
        self.pagination_stats['paginated_requests'] += 1
        
        pagination_info = {
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'previous_page': page - 1 if page > 1 else None
        }
        
        if self.config.enable_total_count:
            pagination_info['total_items'] = total_items
        
        if self.config.enable_cursor_pagination and paginated_data:
            # Add cursor information for cursor-based pagination
            pagination_info['cursor'] = {
                'start': start_index,
                'end': min(end_index, total_items)
            }
        
        return {
            'data': paginated_data,
            'pagination': pagination_info
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pagination statistics"""
        total = self.pagination_stats['total_requests']
        paginated = self.pagination_stats['paginated_requests']
        pagination_rate = (paginated / total * 100) if total > 0 else 0
        
        return {
            'total_requests': total,
            'paginated_requests': paginated,
            'pagination_rate_percent': round(pagination_rate, 2),
            'cache_hits': self.pagination_stats['cache_hits'],
            'cache_misses': self.pagination_stats['cache_misses'],
            'config': {
                'default_page_size': self.config.default_page_size,
                'max_page_size': self.config.max_page_size,
                'cursor_pagination_enabled': self.config.enable_cursor_pagination
            }
        }

class RequestBatcher:
    """Batch similar requests for efficiency"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'batch_count': 0,
            'average_batch_size': 0.0
        }
    
    async def add_to_batch(self, batch_key: str, request_data: Dict[str, Any], 
                          processor_func: callable) -> Any:
        """Add request to batch for processing"""
        self.batch_stats['total_requests'] += 1
        
        # Add request to batch
        batch_item = {
            'data': request_data,
            'timestamp': time.time(),
            'future': asyncio.Future()
        }
        self.pending_batches[batch_key].append(batch_item)
        
        # Check if batch is ready for processing
        batch = self.pending_batches[batch_key]
        if len(batch) >= self.batch_size:
            await self._process_batch(batch_key, processor_func)
        else:
            # Set timeout for batch processing
            asyncio.create_task(self._batch_timeout_handler(batch_key, processor_func))
        
        return await batch_item['future']
    
    async def _batch_timeout_handler(self, batch_key: str, processor_func: callable):
        """Handle batch timeout"""
        await asyncio.sleep(self.batch_timeout)
        
        if batch_key in self.pending_batches and self.pending_batches[batch_key]:
            await self._process_batch(batch_key, processor_func)
    
    async def _process_batch(self, batch_key: str, processor_func: callable):
        """Process a batch of requests"""
        if batch_key not in self.pending_batches:
            return
        
        batch = self.pending_batches[batch_key]
        if not batch:
            return
        
        # Remove batch from pending
        del self.pending_batches[batch_key]
        
        self.batch_stats['batch_count'] += 1
        self.batch_stats['batched_requests'] += len(batch)
        
        # Update average batch size
        total_batched = self.batch_stats['batched_requests']
        batch_count = self.batch_stats['batch_count']
        self.batch_stats['average_batch_size'] = total_batched / batch_count
        
        try:
            # Extract data from batch
            batch_data = [item['data'] for item in batch]
            
            # Process batch
            results = await processor_func(batch_data)
            
            # Set results for futures
            for i, item in enumerate(batch):
                if i < len(results):
                    item['future'].set_result(results[i])
                else:
                    item['future'].set_exception(IndexError(f"Result not found for batch item {i}"))
                    
        except Exception as e:
            # Set exception for all futures in batch
            for item in batch:
                if not item['future'].done():
                    item['future'].set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        total = self.batch_stats['total_requests']
        batched = self.batch_stats['batched_requests']
        batching_rate = (batched / total * 100) if total > 0 else 0
        
        return {
            'total_requests': total,
            'batched_requests': batched,
            'batching_rate_percent': round(batching_rate, 2),
            'batch_count': self.batch_stats['batch_count'],
            'average_batch_size': round(self.batch_stats['average_batch_size'], 2),
            'pending_batches': {k: len(v) for k, v in self.pending_batches.items()},
            'config': {
                'batch_size': self.batch_size,
                'batch_timeout': self.batch_timeout
            }
        }

class ResponseOptimizer:
    """Main response optimization coordinator"""
    
    def __init__(self, compression_config: CompressionConfig = None, 
                 pagination_config: PaginationConfig = None):
        self.compressor = ResponseCompressor(compression_config)
        self.field_selector = FieldSelector()
        self.paginator = ResponsePaginator(pagination_config)
        self.batcher = RequestBatcher()
        
        self.optimization_stats = {
            'total_optimizations': 0,
            'compression_enabled': 0,
            'field_selection_enabled': 0,
            'pagination_enabled': 0,
            'batching_enabled': 0
        }
    
    async def optimize_response(self, data: Any, 
                              compress: bool = True,
                              content_type: str = 'application/json',
                              fields: Optional[List[str]] = None,
                              exclude_fields: Optional[List[str]] = None,
                              paginate: bool = False,
                              page: int = 1,
                              page_size: Optional[int] = None) -> Dict[str, Any]:
        """Apply comprehensive response optimization"""
        self.optimization_stats['total_optimizations'] += 1
        optimization_start = time.time()
        
        # Field selection
        if fields or exclude_fields:
            data = self.field_selector.select_fields(data, fields, exclude_fields)
            self.optimization_stats['field_selection_enabled'] += 1
        
        # Pagination
        pagination_info = None
        if paginate and isinstance(data, list):
            paginated_result = self.paginator.paginate_response(data, page, page_size)
            data = paginated_result['data']
            pagination_info = paginated_result['pagination']
            self.optimization_stats['pagination_enabled'] += 1
        
        # Prepare response
        response_data = {
            'data': data,
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'optimized': True
            }
        }
        
        if pagination_info:
            response_data['pagination'] = pagination_info
        
        # Serialize to JSON for compression
        if compress:
            json_content = json.dumps(response_data, default=str)
            compression_result = self.compressor.compress_response(json_content, content_type)
            
            response_data['compression'] = {
                'compressed': compression_result['compressed'],
                'original_size': compression_result['original_size'],
                'compressed_size': compression_result['compressed_size'],
                'compression_ratio': compression_result['compression_ratio']
            }
            
            if compression_result['compressed']:
                self.optimization_stats['compression_enabled'] += 1
                response_data['content'] = compression_result['content']
            else:
                response_data['content'] = json_content
        
        optimization_time = time.time() - optimization_start
        response_data['metadata']['optimization_time_ms'] = round(optimization_time * 1000, 2)
        
        return response_data
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'response_optimizer': {
                'total_optimizations': self.optimization_stats['total_optimizations'],
                'compression_usage_rate': (self.optimization_stats['compression_enabled'] / 
                                          max(1, self.optimization_stats['total_optimizations']) * 100),
                'field_selection_usage_rate': (self.optimization_stats['field_selection_enabled'] / 
                                              max(1, self.optimization_stats['total_optimizations']) * 100),
                'pagination_usage_rate': (self.optimization_stats['pagination_enabled'] / 
                                         max(1, self.optimization_stats['total_optimizations']) * 100)
            },
            'compression': self.compressor.get_stats(),
            'field_selection': self.field_selector.get_stats(),
            'pagination': self.paginator.get_stats(),
            'batching': self.batcher.get_stats()
        }

# Global response optimizer instance
response_optimizer = ResponseOptimizer()

# Export components
__all__ = [
    'ResponseOptimizer',
    'ResponseCompressor',
    'FieldSelector', 
    'ResponsePaginator',
    'RequestBatcher',
    'CompressionConfig',
    'PaginationConfig',
    'response_optimizer'
]
