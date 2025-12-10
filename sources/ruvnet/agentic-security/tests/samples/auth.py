import bcrypt
import ipaddress
import jwt
import re
import secrets
import hmac
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union
from functools import wraps
from time import time

class TokenError(Exception):
    """Custom exception for token-related errors"""
    pass

class AuthenticationError(Exception):
    """Custom exception for authentication-related errors"""
    pass

import logging
from redis import Redis
from typing import Optional

# Secure configuration
_MAX_ATTEMPTS = 5
_LOCKOUT_TIME = 300  # 5 minutes in seconds
_REDIS_KEY_PREFIX = "auth_attempt:"

# Setup secure logging
logger = logging.getLogger('security')
logger.setLevel(logging.INFO)

from redis.connection import ConnectionPool
from typing import Dict, Optional

from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

# Redis connection pool for rate limiting with strict limits
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None
_MAX_POOL_SIZE = 10
_MAX_POOL_TIMEOUT = 5

def _get_redis() -> Redis:
    """Get Redis connection with enhanced security and monitoring"""
    global _redis_pool, _redis_client
    
    if _redis_pool is None:
        retry = Retry(ExponentialBackoff(cap=10.0, base=1.5), 5)
        
        # Load config from environment with secure defaults
        redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD'),  # Required password
            'ssl': True,
            'ssl_cert_reqs': 'required',
            'ssl_ca_certs': os.getenv('REDIS_CA_CERTS', '/etc/ssl/certs/ca-certificates.crt'),
            'ssl_certfile': os.getenv('REDIS_CERT_FILE'),
            'ssl_keyfile': os.getenv('REDIS_KEY_FILE'),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', str(_MAX_POOL_SIZE))),
            'socket_timeout': float(os.getenv('REDIS_TIMEOUT', '3.0')),
            'socket_connect_timeout': float(os.getenv('REDIS_CONNECT_TIMEOUT', '2.0')),
            'socket_keepalive': True,
            'retry_on_timeout': True,
            'retry': retry,
            'health_check_interval': 15
        }
        
        if not redis_config['password']:
            raise AuthenticationError("Redis password not configured")
            
        _redis_pool = ConnectionPool(**redis_config)
    
    try:
        if _redis_client is None:
            _redis_client = Redis(connection_pool=_redis_pool)
            # Test connection
            _redis_client.ping()
        return _redis_client
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Redis connection failed: {str(e)}")
        raise AuthenticationError("Authentication service temporarily unavailable")

def _check_rate_limit(username: str, ip_address: str) -> None:
    """Check if authentication attempts are within allowed limits using Redis"""
    if not isinstance(username, str) or not username.strip():
        raise AuthenticationError("Invalid username format")
    if not isinstance(ip_address, str) or not ip_address.strip():
        raise AuthenticationError("Invalid IP address format")
    
    # Sanitize inputs
    username = username.lower().strip()
    ip_address = ip_address.strip()
    
    # Validate IP address format
    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        raise AuthenticationError("Invalid IP address format")
    
    # Use multiple time windows for rate limiting
    windows = {
        'minute': 60,
        'hour': 3600,
        'day': 86400
    }
    limits = {
        'minute': 3,
        'hour': 10,
        'day': 20
    }
    
    # Define Redis keys for rate limiting
    user_key = f"{_REDIS_KEY_PREFIX}user:{username}"
    ip_key = f"{_REDIS_KEY_PREFIX}ip:{ip_address}"
    
    current_time = int(time())
    redis = _get_redis()
    
    try:
        redis = _get_redis()
        pipe = redis.pipeline()
        
        # Check both user and IP limits
        user_attempts = int(redis.get(user_key) or 0)
        ip_attempts = int(redis.get(ip_key) or 0)
        
        if user_attempts >= _MAX_ATTEMPTS:
            lockout_remaining = redis.ttl(user_key)
            if lockout_remaining > 0:
                logger.warning(f"User rate limit exceeded: {username}")
                raise AuthenticationError(
                    f"Account temporarily locked. Try again in {lockout_remaining} seconds."
                )
                
        if ip_attempts >= _MAX_ATTEMPTS * 2:  # Higher limit for IPs
            lockout_remaining = redis.ttl(ip_key)
            if lockout_remaining > 0:
                logger.warning(f"IP rate limit exceeded: {ip_address}")
                raise AuthenticationError("Too many attempts from this IP address")
        
        # Increment both counters atomically
        pipe.incr(user_key)
        pipe.expire(user_key, _LOCKOUT_TIME)
        pipe.incr(ip_key)
        pipe.expire(ip_key, _LOCKOUT_TIME)
        pipe.execute()
        
    except (ValueError, ConnectionError) as e:
        logger.error(f"Rate limiting error: {str(e)}")
        raise AuthenticationError("Service temporarily unavailable")

def authenticate_user(username: str, password: str, stored_hash: bytes, ip_address: str) -> bool:
    """Secure authentication with bcrypt and rate limiting"""
    if not all(isinstance(x, str) for x in (username, password, ip_address)):
        raise AuthenticationError("Invalid input types")
    if not isinstance(stored_hash, bytes):
        raise AuthenticationError("Invalid hash format")
    
    # Validate IP address format before rate limiting
    try:
        ip = ipaddress.ip_address(ip_address)
        if ip.is_private or ip.is_loopback or ip.is_multicast:
            raise AuthenticationError("Invalid IP address")
    except ValueError:
        raise AuthenticationError("Invalid IP address format")
        
    try:
        _check_rate_limit(username, str(ip))
        
        # Add pepper before comparing
        pepper = os.getenv('PASSWORD_PEPPER')
        if not pepper:
            raise AuthenticationError("Security configuration error")
            
        # Use constant-time comparison with pepper
        salted_pass = f"{password}{pepper}".encode('utf-8')
        computed_hash = bcrypt.hashpw(salted_pass, stored_hash)
        
        if not hmac.compare_digest(computed_hash, stored_hash):
            logger.warning(f"Failed login attempt for user: {username}")
            raise AuthenticationError("Invalid credentials")
            
        logger.info(f"Successful authentication for user: {username}")
        return True
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise AuthenticationError("Authentication failed")

def hash_password(password: str) -> bytes:
    """Secure password hashing with bcrypt and validation"""
    if not isinstance(password, str):
        raise ValueError("Password must be a string")
        
    # Validate password complexity
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")
    if not re.search(r'[A-Z]', password):
        raise ValueError("Password must contain uppercase letters")
    if not re.search(r'[a-z]', password):
        raise ValueError("Password must contain lowercase letters")
    if not re.search(r'\d', password):
        raise ValueError("Password must contain numbers")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValueError("Password must contain special characters")
        
    # Use stronger work factor for bcrypt and secure pepper handling
    pepper = os.getenv('PASSWORD_PEPPER')
    if not pepper:
        raise RuntimeError("PASSWORD_PEPPER environment variable not set")
    
    # Add pepper before hashing
    salted_pass = f"{password}{pepper}".encode('utf-8')
    
    # Use work factor 16 for increased security (industry recommended minimum)
    salt = bcrypt.gensalt(rounds=16)
    hashed = bcrypt.hashpw(salted_pass, salt)
    
    # Verify the hash can be validated
    try:
        bcrypt.checkpw(salted_pass, hashed)
    except Exception as e:
        logger.error(f"Hash verification failed: {str(e)}")
        raise ValueError("Generated hash failed verification")
        
    return hashed

def generate_token(
    user_data: Dict,
    expiry_hours: int = 1,
    audience: str = None,
    device_info: Optional[Dict] = None,
    ip_address: str = None
) -> Tuple[str, str]:
    
    # Rate limit token generation by IP
    if ip_address:
        redis = _get_redis()
        token_key = f"token_gen:{ip_address}"
        token_count = redis.incr(token_key)
        if token_count == 1:
            redis.expire(token_key, 3600)  # 1 hour window
        if token_count > 10:  # Max 10 tokens per hour per IP
            raise TokenError("Token generation rate limit exceeded")
    """
    Secure token generation with enhanced security measures and claims
    
    Args:
        user_data: User information dictionary
        expiry_hours: Token validity period in hours
        audience: Expected token audience (e.g., 'web', 'mobile')
    """
    if not isinstance(user_data, dict):
        raise ValueError("User data must be a dictionary")
    if not isinstance(expiry_hours, int) or expiry_hours <= 0:
        raise ValueError("Expiry hours must be a positive integer")
    if audience and not isinstance(audience, str):
        raise ValueError("Audience must be a string")
        
    # Create a safe copy of user data with strict validation
    required_fields = {'user_id', 'username', 'role'}
    if not all(field in user_data for field in required_fields):
        raise ValueError("Missing required user data fields")
    
    current_time = datetime.utcnow()
    safe_data = {
        'user_id': str(user_data['user_id']),
        'username': str(user_data['username']),
        'role': str(user_data['role']),
        'jti': secrets.token_hex(32),  # Increased to 256-bit unique ID
        'iat': current_time,
        'exp': current_time + timedelta(hours=expiry_hours),
        'nbf': current_time,
        'iss': 'auth_service',  # Token issuer
        'aud': audience or 'default',  # Token audience
        'sub': str(user_data['user_id']),  # Token subject
        'ip': user_data.get('ip_address'),  # Client IP if available
        'device': user_data.get('device_info'),  # Device information
    }
    
    # Use secure key management
    secret = _get_or_generate_secret()
    
    # Generate secure nonce
    nonce = secrets.token_urlsafe(32)
    
    # Enhanced headers with additional security measures
    headers = {
        'kid': secrets.token_hex(32),  # Increased key ID length
        'typ': 'JWT',
        'cty': 'JWT',
        'alg': 'HS512',
        'enc': 'none',
        'zip': 'none',
        'x5t': secrets.token_urlsafe(32),
        'jku': None,
        'x5u': None,
        'nonce': nonce,  # Add nonce for replay protection
        'crit': ['alg', 'typ', 'nonce']  # Critical headers that must be validated
    }
    
    # Add security-relevant claims
    safe_data.update({
        'nonce': nonce,
        'auth_time': int(time()),  # Add authentication time
        'sid': secrets.token_urlsafe(32),  # Add session ID
        'amr': ['pwd'],  # Authentication method reference
        'azp': audience or 'default'  # Authorized party
    })
    
    token = jwt.encode(
        safe_data,
        secret,
        algorithm='HS512',
        headers=headers
    )
    return token, secret

def verify_token(token: str, secret: str, audience: str = None) -> Optional[Dict]:
    """
    Secure token verification with enhanced security checks
    
    Args:
        token: JWT token string
        secret: Secret key used for token verification
        audience: Expected token audience to validate against
        
    Returns:
        Dict containing verified token data or None if verification fails
        
    Raises:
        TokenError: If token format or content is invalid
    """
    if not all(isinstance(x, str) for x in filter(None, [token, secret, audience])):
        raise TokenError("Invalid input types")
        
    # Rate limit token verification attempts
    redis = _get_redis()
    verify_key = f"token_verify:{datetime.utcnow().strftime('%Y%m%d%H')}"
    if int(redis.incr(verify_key)) > 1000:  # Max 1000 verifications per hour
        raise TokenError("Token verification rate limit exceeded")
        
    try:
        # First verify signature in constant time
        try:
            jwt.get_unverified_header(token)
        except jwt.InvalidTokenError:
            return None
            
        # Then decode with explicit verification
        data = jwt.decode(
            token, 
            secret, 
            algorithms=['HS512'],
            options={
                'verify_signature': True,
                'verify_exp': True,
                'verify_nbf': True,
                'verify_iat': True,
                'verify_aud': True,
                'require': ['exp', 'iat', 'nbf']
            }
        )
        
        # Validate expected data structure
        if not isinstance(data, dict):
            raise TokenError("Invalid token payload structure")
            
        # Verify required fields and types
        required_fields = {
            'user_id': str,
            'username': str,
            'role': str,
            'exp': (int, float)  # exp can be int or float timestamp
        }
        
        for field, expected_type in required_fields.items():
            if field not in data:
                raise TokenError(f"Missing required field: {field}")
            if not isinstance(data[field], expected_type):
                raise TokenError(f"Invalid type for field: {field}")
                
        return data
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception as e:
        raise TokenError(f"Token verification failed: {str(e)}")
import os
import secrets
from cryptography.fernet import Fernet
from pathlib import Path

class KeyManager:
    """Secure key management for authentication tokens"""
    
    _instance = None
    _KEY_FILE = "secret.key"
    _KEY_DIR = Path("/opt/secure/keys")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._key_path = self._KEY_DIR / self._KEY_FILE
        self._current_key = None
        self._fernet = None
        self._initialize()
    
    def _initialize(self):
        """Initialize key storage with secure permissions"""
        if not self._KEY_DIR.exists():
            self._KEY_DIR.mkdir(parents=True, mode=0o700)
        
        if not self._key_path.exists():
            self._generate_new_key()
        else:
            self._load_key()
    
    def _generate_new_key(self):
        """Generate and store a new secret key"""
        key = secrets.token_bytes(64)  # 512-bit key
        
        # Encrypt the key before storing
        if not self._fernet:
            self._fernet = Fernet(Fernet.generate_key())
        
        encrypted_key = self._fernet.encrypt(key)
        
        # Secure file permissions
        with open(self._key_path, 'wb') as f:
            f.write(encrypted_key)
        os.chmod(self._key_path, 0o600)
        
        self._current_key = key
    
    def _load_key(self):
        """Load the existing secret key"""
        try:
            with open(self._key_path, 'rb') as f:
                encrypted_key = f.read()
            
            if not self._fernet:
                self._fernet = Fernet(Fernet.generate_key())
                
            self._current_key = self._fernet.decrypt(encrypted_key)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load secret key: {str(e)}")
    
    def get_current_key(self) -> bytes:
        """Get the current secret key"""
        if not self._current_key:
            self._load_key()
        return self._current_key
    
    def rotate_key(self) -> None:
        """Rotate the secret key with enhanced security checks"""
        from auth_config import (KEY_ROTATION_INTERVAL, MINIMUM_KEY_AGE, 
                               MAX_KEY_AGE, KEY_LENGTH_BYTES)
        
        current_time = time.time()
        
        try:
            key_age = current_time - os.path.getmtime(self._key_path)
            key_stats = os.stat(self._key_path)
            
            # Verify key file permissions and ownership
            if key_stats.st_mode & 0o777 != 0o600:
                logger.error("Invalid key file permissions")
                self._generate_new_key()
                return
                
            # Check if key is too new to rotate
            if key_age < MINIMUM_KEY_AGE:
                logger.debug("Key too new for rotation")
                return
                
            # Force rotation if key is too old
            if key_age > MAX_KEY_AGE:
                logger.warning("Forcing key rotation due to age")
                self._generate_new_key()
                return
                
            # Normal rotation after interval
            if key_age > KEY_ROTATION_INTERVAL:
                logger.info("Performing scheduled key rotation")
                self._generate_new_key()
                
        except OSError as e:
            logger.error(f"Key file error during rotation: {str(e)}")
            self._generate_new_key()

# Global key manager instance
_key_manager = KeyManager()

def _get_or_generate_secret() -> bytes:
    """Get the current secret key or generate a new one"""
    return _key_manager.get_current_key()
"""Security configuration settings"""

import os
from datetime import timedelta

# Redis rate limiting settings
RATE_LIMIT_WINDOWS = {
    'minute': 60,
    'hour': 3600,
    'day': 86400
}

RATE_LIMIT_ATTEMPTS = {
    'minute': 3,
    'hour': 10,
    'day': 20
}

# Password security
MIN_PASSWORD_LENGTH = 12
PASSWORD_ROUNDS = 14
PASSWORD_PEPPER = os.getenv('PASSWORD_PEPPER', None)

# Token settings
TOKEN_EXPIRY = timedelta(hours=24)
TOKEN_ALGORITHM = 'HS512'
REQUIRED_CLAIMS = ['exp', 'iat', 'nbf', 'aud', 'sub']

# Redis connection
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'ssl': True,
    'ssl_cert_reqs': 'required',
    'ssl_ca_certs': '/etc/ssl/certs/ca-certificates.crt',
    'socket_timeout': 5.0,
    'socket_keepalive': True,
    'health_check_interval': 30
}
"""Security configuration constants"""

from typing import Dict, Final

# Rate limiting windows and attempts
RATE_LIMIT_WINDOWS: Final[Dict[str, int]] = {
    'minute': 60,
    'hour': 3600,
    'day': 86400,
    'week': 604800
}

RATE_LIMIT_ATTEMPTS: Final[Dict[str, int]] = {
    'minute': 3,
    'hour': 10,
    'day': 20,
    'week': 50
}

# Redis security settings
REDIS_MAX_POOL_SIZE: Final[int] = 10
REDIS_POOL_TIMEOUT: Final[int] = 5
REDIS_CONNECT_TIMEOUT: Final[float] = 2.0
REDIS_SOCKET_TIMEOUT: Final[float] = 3.0

# Password security
MIN_PASSWORD_LENGTH: Final[int] = 16
BCRYPT_ROUNDS: Final[int] = 16
PASSWORD_COMPLEXITY: Final[Dict[str, str]] = {
    'uppercase': r'[A-Z]',
    'lowercase': r'[a-z]', 
    'numbers': r'\d',
    'special': r'[!@#$%^&*(),.?":{}|<>]'
}

# Token security
TOKEN_EXPIRY_HOURS: Final[int] = 1
TOKEN_ALGORITHM: Final[str] = 'HS512'
REQUIRED_TOKEN_CLAIMS: Final[tuple] = (
    'exp', 'iat', 'nbf', 'aud', 'sub', 'jti',
    'auth_time', 'nonce', 'sid'
)

# Key rotation
KEY_ROTATION_HOURS: Final[int] = 24
KEY_LENGTH_BYTES: Final[int] = 64  # 512 bits
"""Security configuration constants"""

from typing import Dict, Final
import os

# Token generation rate limits
TOKEN_RATE_LIMIT_WINDOW: Final[int] = 3600  # 1 hour
TOKEN_RATE_LIMIT_MAX: Final[int] = 10  # tokens per window

# Key rotation settings
KEY_ROTATION_INTERVAL: Final[int] = 3600  # 1 hour
MINIMUM_KEY_AGE: Final[int] = 300  # 5 minutes
MAX_KEY_AGE: Final[int] = 7200  # 2 hours

# Redis security settings
REDIS_SECURITY_CONFIG: Final[Dict] = {
    'ssl': True,
    'ssl_cert_reqs': 'required',
    'ssl_ca_certs': os.getenv('REDIS_CA_CERTS', '/etc/ssl/certs/ca-certificates.crt'),
    'socket_timeout': 3.0,
    'socket_connect_timeout': 2.0,
    'health_check_interval': 30,
    'retry_on_timeout': True
}

# Token security settings
TOKEN_SECURITY_CONFIG: Final[Dict] = {
    'algorithms': ['HS512'],
    'verify_signature': True,
    'verify_exp': True,
    'verify_nbf': True,
    'verify_iat': True,
    'verify_aud': True,
    'require': ['exp', 'iat', 'nbf', 'aud', 'sub', 'jti']
}
"""Centralized security configuration"""

from typing import Dict, Final
import os
from datetime import timedelta

# Rate limiting configuration
RATE_LIMIT_WINDOWS: Final[Dict[str, int]] = {
    'minute': 60,
    'hour': 3600,
    'day': 86400,
    'week': 604800
}

RATE_LIMIT_ATTEMPTS: Final[Dict[str, int]] = {
    'minute': 3,
    'hour': 10, 
    'day': 20,
    'week': 50
}

# Redis security configuration
REDIS_CONFIG: Final[Dict] = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'password': os.getenv('REDIS_PASSWORD'),
    'ssl': True,
    'ssl_cert_reqs': 'required',
    'ssl_ca_certs': os.getenv('REDIS_CA_CERTS', '/etc/ssl/certs/ca-certificates.crt'),
    'ssl_certfile': os.getenv('REDIS_CERT_FILE'),
    'ssl_keyfile': os.getenv('REDIS_KEY_FILE'),
    'socket_timeout': 3.0,
    'socket_connect_timeout': 2.0,
    'socket_keepalive': True,
    'health_check_interval': 30,
    'retry_on_timeout': True,
    'max_connections': 10,
    'retry_on_error': [TimeoutError, ConnectionError]
}

# Password security settings
MIN_PASSWORD_LENGTH: Final[int] = 16
BCRYPT_ROUNDS: Final[int] = 16
PASSWORD_COMPLEXITY: Final[Dict[str, str]] = {
    'uppercase': r'[A-Z]',
    'lowercase': r'[a-z]',
    'numbers': r'\d',
    'special': r'[!@#$%^&*(),.?":{}|<>]'
}

# Token security settings
TOKEN_EXPIRY: Final[timedelta] = timedelta(hours=1)
TOKEN_ALGORITHM: Final[str] = 'HS512'
REQUIRED_TOKEN_CLAIMS: Final[tuple] = (
    'exp', 'iat', 'nbf', 'aud', 'sub', 'jti',
    'auth_time', 'nonce', 'sid'
)
TOKEN_RATE_LIMIT_WINDOW: Final[int] = 3600  # 1 hour
TOKEN_RATE_LIMIT_MAX: Final[int] = 10  # tokens per window

# Key rotation settings
KEY_ROTATION_INTERVAL: Final[int] = 3600  # 1 hour
MINIMUM_KEY_AGE: Final[int] = 300  # 5 minutes
MAX_KEY_AGE: Final[int] = 7200  # 2 hours
KEY_LENGTH_BYTES: Final[int] = 64  # 512 bits
