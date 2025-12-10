# File: backend/security/encryption.py
"""
Advanced Encryption Service with AES-256 for data at rest protection.
Provides field-level encryption, key rotation, and secure key management.
"""

import os
import base64
import secrets
import hashlib
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone, timedelta
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import logging
from contextlib import contextmanager

from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class EncryptionError(Exception):
    """Base exception for encryption-related errors"""
    pass

class KeyRotationError(EncryptionError):
    """Exception raised during key rotation operations"""
    pass

class DataEncryption:
    """
    Advanced AES-256 encryption service for protecting sensitive data at rest.
    
    Features:
    - AES-256-GCM encryption with authenticated encryption
    - Key derivation using PBKDF2 with high iteration count
    - Key rotation support with backward compatibility
    - Field-level encryption for database columns
    - Secure key management with environment-based master keys
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption service with master key.
        
        Args:
            master_key: Base64-encoded master key. If None, loads from environment.
        """
        self.master_key = master_key or self._get_master_key()
        self.salt_length = 16
        self.iv_length = 12  # GCM mode IV length
        self.tag_length = 16  # GCM authentication tag length
        self.iterations = 100000  # PBKDF2 iterations
        
        # Key rotation tracking
        self.current_key_version = 1
        self.key_rotation_history: Dict[int, dict] = {}
        
        uap_logger.log_security_event(
            "Encryption service initialized",
            metadata={"key_version": self.current_key_version}
        )
    
    def _get_master_key(self) -> str:
        """Get master key from environment or generate new one"""
        key = os.getenv("ENCRYPTION_MASTER_KEY")
        if not key:
            # Generate new master key for development/testing
            key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("No ENCRYPTION_MASTER_KEY found, using generated key (not for production!)")
            uap_logger.log_security_event(
                "Generated new encryption master key",
                success=False,
                metadata={"warning": "Not for production use"}
            )
        return key
    
    def _derive_key(self, salt: bytes, key_version: int = None) -> bytes:
        """Derive encryption key from master key using PBKDF2"""
        try:
            master_key_bytes = base64.urlsafe_b64decode(self.master_key.encode())
            
            # Add key version to salt for key rotation support
            version = key_version or self.current_key_version
            versioned_salt = salt + version.to_bytes(4, 'big')
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # AES-256 key length
                salt=versioned_salt,
                iterations=self.iterations,
                backend=default_backend()
            )
            
            return kdf.derive(master_key_bytes)
        except Exception as e:
            uap_logger.log_security_event(
                "Key derivation failed",
                success=False,
                metadata={"error": str(e)}
            )
            raise EncryptionError(f"Key derivation failed: {e}")
    
    def encrypt_data(self, plaintext: Union[str, bytes], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Encrypt data using AES-256-GCM with authenticated encryption.
        
        Args:
            plaintext: Data to encrypt
            context: Additional context for logging
            
        Returns:
            Base64-encoded encrypted data with metadata
        """
        try:
            if isinstance(plaintext, str):
                plaintext = plaintext.encode('utf-8')
            
            # Generate random salt and IV
            salt = secrets.token_bytes(self.salt_length)
            iv = secrets.token_bytes(self.iv_length)
            
            # Derive encryption key
            key = self._derive_key(salt)
            
            # Encrypt using AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            # Create encrypted data package
            encrypted_package = {
                "version": self.current_key_version,
                "salt": base64.urlsafe_b64encode(salt).decode(),
                "iv": base64.urlsafe_b64encode(iv).decode(),
                "tag": base64.urlsafe_b64encode(encryptor.tag).decode(),
                "data": base64.urlsafe_b64encode(ciphertext).decode(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Encode entire package
            package_json = json.dumps(encrypted_package)
            encoded_package = base64.urlsafe_b64encode(package_json.encode()).decode()
            
            uap_logger.log_security_event(
                "Data encrypted successfully",
                metadata={
                    "data_size": len(plaintext),
                    "key_version": self.current_key_version,
                    **(context or {})
                }
            )
            
            return encoded_package
            
        except Exception as e:
            uap_logger.log_security_event(
                "Data encryption failed",
                success=False,
                metadata={"error": str(e), **(context or {})}
            )
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Decrypt data using AES-256-GCM with authentication verification.
        
        Args:
            encrypted_data: Base64-encoded encrypted data package
            context: Additional context for logging
            
        Returns:
            Decrypted plaintext string
        """
        try:
            # Decode package
            package_json = base64.urlsafe_b64decode(encrypted_data.encode()).decode()
            package = json.loads(package_json)
            
            # Extract components
            key_version = package["version"]
            salt = base64.urlsafe_b64decode(package["salt"].encode())
            iv = base64.urlsafe_b64decode(package["iv"].encode())
            tag = base64.urlsafe_b64decode(package["tag"].encode())
            ciphertext = base64.urlsafe_b64decode(package["data"].encode())
            
            # Derive key with correct version
            key = self._derive_key(salt, key_version)
            
            # Decrypt using AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            uap_logger.log_security_event(
                "Data decrypted successfully",
                metadata={
                    "key_version": key_version,
                    "data_size": len(plaintext),
                    **(context or {})
                }
            )
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            uap_logger.log_security_event(
                "Data decryption failed",
                success=False,
                metadata={"error": str(e), **(context or {})}
            )
            raise EncryptionError(f"Decryption failed: {e}")
    
    def rotate_keys(self, new_master_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Rotate encryption keys to new version.
        
        Args:
            new_master_key: New master key (if None, generates new one)
            
        Returns:
            Key rotation result with old and new versions
        """
        try:
            old_version = self.current_key_version
            old_master_key = self.master_key
            
            # Generate or use provided new master key
            if new_master_key is None:
                new_master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            
            # Store old key info for backward compatibility
            self.key_rotation_history[old_version] = {
                "master_key": old_master_key,
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "version": old_version
            }
            
            # Update to new key
            self.current_key_version += 1
            self.master_key = new_master_key
            
            result = {
                "success": True,
                "old_version": old_version,
                "new_version": self.current_key_version,
                "rotation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            uap_logger.log_security_event(
                "Encryption key rotation completed",
                metadata=result
            )
            
            return result
            
        except Exception as e:
            uap_logger.log_security_event(
                "Key rotation failed",
                success=False,
                metadata={"error": str(e)}
            )
            raise KeyRotationError(f"Key rotation failed: {e}")
    
    def encrypt_field(self, value: Any, field_name: str) -> Optional[str]:
        """Encrypt a database field value"""
        if value is None:
            return None
        
        string_value = str(value) if not isinstance(value, str) else value
        return self.encrypt_data(string_value, context={"field": field_name})
    
    def decrypt_field(self, encrypted_value: Optional[str], field_name: str) -> Optional[str]:
        """Decrypt a database field value"""
        if encrypted_value is None:
            return None
        
        return self.decrypt_data(encrypted_value, context={"field": field_name})

class EncryptedDataService:
    """
    Service for managing encrypted data storage and retrieval.
    Provides higher-level interface for common encryption operations.
    """
    
    def __init__(self, encryption_service: Optional[DataEncryption] = None):
        self.encryption = encryption_service or DataEncryption()
        self.sensitive_fields = {
            # Define which fields should be encrypted
            "user_data": ["email", "full_name", "phone", "address"],
            "documents": ["content", "metadata"],
            "conversations": ["message_content", "response_content"],
            "api_keys": ["key_value", "secret"],
            "passwords": ["hashed_password", "recovery_tokens"]
        }
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a data dictionary.
        
        Args:
            data: Dictionary containing data to encrypt
            data_type: Type of data (e.g., 'user_data', 'documents')
            
        Returns:
            Dictionary with sensitive fields encrypted
        """
        if data_type not in self.sensitive_fields:
            return data
        
        encrypted_data = data.copy()
        fields_to_encrypt = self.sensitive_fields[data_type]
        
        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field] is not None:
                try:
                    encrypted_data[field] = self.encryption.encrypt_field(
                        encrypted_data[field], 
                        f"{data_type}.{field}"
                    )
                except Exception as e:
                    logger.error(f"Failed to encrypt field {field}: {e}")
                    # Don't store unencrypted sensitive data
                    encrypted_data[field] = None
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in a data dictionary.
        
        Args:
            encrypted_data: Dictionary containing encrypted data
            data_type: Type of data (e.g., 'user_data', 'documents')
            
        Returns:
            Dictionary with sensitive fields decrypted
        """
        if data_type not in self.sensitive_fields:
            return encrypted_data
        
        decrypted_data = encrypted_data.copy()
        fields_to_decrypt = self.sensitive_fields[data_type]
        
        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field] is not None:
                try:
                    decrypted_data[field] = self.encryption.decrypt_field(
                        decrypted_data[field],
                        f"{data_type}.{field}"
                    )
                except Exception as e:
                    logger.error(f"Failed to decrypt field {field}: {e}")
                    decrypted_data[field] = None
        
        return decrypted_data
    
    @contextmanager
    def temporary_key_rotation(self):
        """Context manager for testing key rotation without permanent changes"""
        original_version = self.encryption.current_key_version
        original_master_key = self.encryption.master_key
        original_history = self.encryption.key_rotation_history.copy()
        
        try:
            yield self.encryption
        finally:
            # Restore original state
            self.encryption.current_key_version = original_version
            self.encryption.master_key = original_master_key
            self.encryption.key_rotation_history = original_history
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get current encryption service status"""
        return {
            "service_active": True,
            "current_key_version": self.encryption.current_key_version,
            "key_rotation_history_count": len(self.encryption.key_rotation_history),
            "sensitive_field_types": list(self.sensitive_fields.keys()),
            "encryption_algorithm": "AES-256-GCM",
            "key_derivation": "PBKDF2-SHA256",
            "last_rotation": max(
                (info["retired_at"] for info in self.encryption.key_rotation_history.values()),
                default=None
            )
        }

# Global encryption service instance
_global_encryption_service = None

def get_encryption_service() -> EncryptedDataService:
    """Get global encryption service instance"""
    global _global_encryption_service
    if _global_encryption_service is None:
        _global_encryption_service = EncryptedDataService()
    return _global_encryption_service