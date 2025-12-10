# File: backend/config/cdn_config.py
"""
CDN configuration and management for UAP platform.
Provides static asset delivery and document caching via CDN.
"""

import os
import mimetypes
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CloudFrontConfig:
    """AWS CloudFront CDN configuration"""
    distribution_id: str = os.getenv("CLOUDFRONT_DISTRIBUTION_ID", "")
    domain_name: str = os.getenv("CLOUDFRONT_DOMAIN", "")
    invalidation_enabled: bool = True
    price_class: str = "PriceClass_100"  # All, 100, 200
    
    # Cache behaviors
    default_ttl: int = 86400  # 24 hours
    max_ttl: int = 31536000   # 1 year
    min_ttl: int = 0
    
    # Origins
    origins: Dict[str, str] = None
    
    def __post_init__(self):
        if self.origins is None:
            self.origins = {
                "api": os.getenv("UAP_API_ORIGIN", "api.uap.ai"),
                "static": os.getenv("UAP_STATIC_ORIGIN", "static.uap.ai"),
                "documents": os.getenv("UAP_DOCS_ORIGIN", "docs.uap.ai")
            }

@dataclass
class CloudflareConfig:
    """Cloudflare CDN configuration"""
    zone_id: str = os.getenv("CLOUDFLARE_ZONE_ID", "")
    api_token: str = os.getenv("CLOUDFLARE_API_TOKEN", "")
    domain_name: str = os.getenv("CLOUDFLARE_DOMAIN", "")
    
    # Cache settings
    cache_level: str = "aggressive"  # basic, simplified, aggressive
    browser_ttl: int = 86400
    edge_ttl: int = 604800  # 1 week
    
    # Performance features
    minify_js: bool = True
    minify_css: bool = True
    minify_html: bool = True
    brotli_compression: bool = True

@dataclass
class FastlyConfig:
    """Fastly CDN configuration"""
    service_id: str = os.getenv("FASTLY_SERVICE_ID", "")
    api_token: str = os.getenv("FASTLY_API_TOKEN", "")
    domain_name: str = os.getenv("FASTLY_DOMAIN", "")
    
    # Cache settings
    default_ttl: int = 3600
    stale_while_revalidate: int = 86400
    stale_if_error: int = 259200  # 3 days

class CDNManager:
    """CDN management and asset optimization"""
    
    def __init__(self, provider: str = "cloudfront"):
        self.provider = provider.lower()
        self.config = self._get_provider_config()
        self.asset_mappings = self._setup_asset_mappings()
        self.cache_headers = self._setup_cache_headers()
    
    def _get_provider_config(self):
        """Get provider-specific configuration"""
        if self.provider == "cloudfront":
            return CloudFrontConfig()
        elif self.provider == "cloudflare":
            return CloudflareConfig()
        elif self.provider == "fastly":
            return FastlyConfig()
        else:
            raise ValueError(f"Unsupported CDN provider: {self.provider}")
    
    def _setup_asset_mappings(self) -> Dict[str, str]:
        """Setup asset type to cache behavior mappings"""
        return {
            # Static assets - long cache
            ".js": "static-long",
            ".css": "static-long", 
            ".woff": "static-long",
            ".woff2": "static-long",
            ".ttf": "static-long",
            ".eot": "static-long",
            
            # Images - medium cache
            ".png": "images-medium",
            ".jpg": "images-medium",
            ".jpeg": "images-medium",
            ".gif": "images-medium",
            ".svg": "images-medium",
            ".webp": "images-medium",
            ".ico": "images-medium",
            
            # Documents - short cache (may change)
            ".pdf": "documents-short",
            ".docx": "documents-short",
            ".xlsx": "documents-short",
            ".pptx": "documents-short",
            
            # API responses - very short cache
            ".json": "api-short",
            ".xml": "api-short"
        }
    
    def _setup_cache_headers(self) -> Dict[str, Dict[str, str]]:
        """Setup cache headers for different asset types"""
        return {
            "static-long": {
                "Cache-Control": "public, max-age=31536000, immutable",  # 1 year
                "Expires": "Thu, 31 Dec 2025 23:59:59 GMT"
            },
            "images-medium": {
                "Cache-Control": "public, max-age=604800",  # 1 week
                "Vary": "Accept-Encoding"
            },
            "documents-short": {
                "Cache-Control": "public, max-age=3600",  # 1 hour
                "Vary": "Accept-Encoding"
            },
            "api-short": {
                "Cache-Control": "public, max-age=300",  # 5 minutes
                "Vary": "Accept-Encoding, Authorization"
            },
            "no-cache": {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        }
    
    def get_cdn_url(self, asset_path: str, asset_type: str = "static") -> str:
        """Generate CDN URL for an asset"""
        if not hasattr(self.config, 'domain_name') or not self.config.domain_name:
            return asset_path  # Return original path if CDN not configured
        
        # Determine subdomain based on asset type
        if asset_type == "documents":
            subdomain = "docs"
        elif asset_type == "api":
            subdomain = "api"
        else:
            subdomain = "static"
        
        # Clean asset path
        clean_path = asset_path.lstrip('/')
        
        return f"https://{subdomain}.{self.config.domain_name}/{clean_path}"
    
    def get_cache_headers(self, file_path: str) -> Dict[str, str]:
        """Get appropriate cache headers for a file"""
        file_ext = Path(file_path).suffix.lower()
        cache_type = self.asset_mappings.get(file_ext, "api-short")
        return self.cache_headers.get(cache_type, self.cache_headers["no-cache"])
    
    def should_cache(self, file_path: str) -> bool:
        """Determine if a file should be cached"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.asset_mappings
    
    def get_optimized_headers(self, file_path: str, file_size: int = 0) -> Dict[str, str]:
        """Get optimized headers for a file"""
        headers = self.get_cache_headers(file_path)
        
        # Add compression headers for compressible content
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and self._is_compressible(mime_type):
            headers["Content-Encoding"] = "gzip"
            headers["Vary"] = headers.get("Vary", "") + ", Accept-Encoding"
        
        # Add security headers
        headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block"
        })
        
        # Add CORS headers for cross-origin requests
        if file_path.endswith(('.js', '.css', '.woff', '.woff2')):
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        
        return headers
    
    def _is_compressible(self, mime_type: str) -> bool:
        """Check if content type is compressible"""
        compressible_types = [
            'text/', 'application/json', 'application/javascript',
            'application/xml', 'image/svg+xml'
        ]
        return any(mime_type.startswith(ct) for ct in compressible_types)
    
    async def invalidate_cache(self, paths: List[str]) -> Dict[str, Any]:
        """Invalidate CDN cache for specific paths"""
        if self.provider == "cloudfront":
            return await self._invalidate_cloudfront(paths)
        elif self.provider == "cloudflare":
            return await self._invalidate_cloudflare(paths)
        elif self.provider == "fastly":
            return await self._invalidate_fastly(paths)
        else:
            return {"success": False, "message": "Provider not supported"}
    
    async def _invalidate_cloudfront(self, paths: List[str]) -> Dict[str, Any]:
        """Invalidate CloudFront cache"""
        try:
            import boto3
            
            cloudfront = boto3.client('cloudfront')
            response = cloudfront.create_invalidation(
                DistributionId=self.config.distribution_id,
                InvalidationBatch={
                    'Paths': {
                        'Quantity': len(paths),
                        'Items': paths
                    },
                    'CallerReference': f"uap-{int(os.time())}"
                }
            )
            
            return {
                "success": True,
                "invalidation_id": response['Invalidation']['Id'],
                "status": response['Invalidation']['Status']
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _invalidate_cloudflare(self, paths: List[str]) -> Dict[str, Any]:
        """Invalidate Cloudflare cache"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.cloudflare.com/client/v4/zones/{self.config.zone_id}/purge_cache",
                    headers={
                        "Authorization": f"Bearer {self.config.api_token}",
                        "Content-Type": "application/json"
                    },
                    json={"files": paths}
                )
                
                result = response.json()
                return {
                    "success": result.get("success", False),
                    "result": result.get("result"),
                    "errors": result.get("errors", [])
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _invalidate_fastly(self, paths: List[str]) -> Dict[str, Any]:
        """Invalidate Fastly cache"""
        try:
            import httpx
            
            results = []
            async with httpx.AsyncClient() as client:
                for path in paths:
                    response = await client.post(
                        f"https://api.fastly.com/service/{self.config.service_id}/purge{path}",
                        headers={
                            "Fastly-Token": self.config.api_token,
                            "Accept": "application/json"
                        }
                    )
                    results.append({
                        "path": path,
                        "status": response.status_code,
                        "success": response.status_code == 200
                    })
                
            return {
                "success": all(r["success"] for r in results),
                "results": results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CDN configuration and stats"""
        return {
            "provider": self.provider,
            "enabled": bool(getattr(self.config, 'domain_name', '')),
            "domain": getattr(self.config, 'domain_name', ''),
            "asset_types": len(self.asset_mappings),
            "cache_behaviors": len(self.cache_headers),
            "compression_enabled": True,
            "invalidation_supported": True
        }

# CDN middleware for FastAPI
class CDNMiddleware:
    """Middleware to add CDN headers and optimization"""
    
    def __init__(self, cdn_manager: CDNManager):
        self.cdn_manager = cdn_manager
    
    async def __call__(self, request, call_next):
        response = await call_next(request)
        
        # Add CDN headers for static assets
        if request.url.path.startswith('/static/') or self._is_static_asset(request.url.path):
            headers = self.cdn_manager.get_optimized_headers(request.url.path)
            for key, value in headers.items():
                response.headers[key] = value
        
        return response
    
    def _is_static_asset(self, path: str) -> bool:
        """Check if path is a static asset"""
        return any(path.endswith(ext) for ext in self.cdn_manager.asset_mappings.keys())

# Global CDN manager instance
cdn_manager = CDNManager(provider=os.getenv("CDN_PROVIDER", "cloudfront"))

# Export configuration
__all__ = [
    'CDNManager',
    'CDNMiddleware',
    'CloudFrontConfig',
    'CloudflareConfig',
    'FastlyConfig',
    'cdn_manager'
]