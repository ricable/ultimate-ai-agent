"""Data encryption module for SAFLA - provides encryption at rest functionality"""

import os
import base64
import json
import logging
import re
from typing import Any, Dict, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption operations fail"""
    pass


class DataEncryptor:
    """Handles encryption and decryption of data at rest"""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryptor with encryption key.
        
        Args:
            key: Base64 encoded encryption key or None to generate/load from env
        """
        if key:
            self._key = key.encode() if isinstance(key, str) else key
        else:
            self._key = self._get_or_generate_key()
        
        try:
            self._cipher = Fernet(self._key)
            logger.info("Data encryptor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryptor: {e}")
            raise EncryptionError(f"Invalid encryption key: {e}")
    
    def _get_or_generate_key(self) -> bytes:
        """Get encryption key from environment or generate new one"""
        # Try to get from environment
        env_key = os.getenv("SAFLA_ENCRYPTION_KEY")
        if env_key:
            try:
                # Validate key format
                Fernet(env_key.encode())
                return env_key.encode()
            except Exception:
                logger.warning("Invalid SAFLA_ENCRYPTION_KEY in environment")
        
        # Try to load from secure key file
        key_file = Path.home() / ".safla" / "encryption.key"
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    key = f.read()
                    # Validate key
                    Fernet(key)
                    return key
            except Exception as e:
                logger.warning(f"Failed to load key from file: {e}")
        
        # Generate new key
        logger.info("Generating new encryption key")
        key = Fernet.generate_key()
        
        # Save to secure location
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.touch(mode=0o600)  # Restrict permissions
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info(f"Encryption key saved to {key_file}")
        except Exception as e:
            logger.warning(f"Failed to save encryption key: {e}")
        
        return key
    
    def encrypt_string(self, data: str) -> str:
        """
        Encrypt a string.
        
        Args:
            data: String to encrypt
            
        Returns:
            Base64 encoded encrypted string
        """
        try:
            encrypted = self._cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt string: {e}")
    
    def decrypt_string(self, encrypted_data: str) -> str:
        """
        Decrypt a string.
        
        Args:
            encrypted_data: Base64 encoded encrypted string
            
        Returns:
            Decrypted string
        """
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self._cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt string: {e}")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """
        Encrypt a dictionary.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Base64 encoded encrypted JSON string
        """
        try:
            json_str = json.dumps(data, sort_keys=True)
            return self.encrypt_string(json_str)
        except Exception as e:
            logger.error(f"Dictionary encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt dictionary: {e}")
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt a dictionary.
        
        Args:
            encrypted_data: Base64 encoded encrypted JSON string
            
        Returns:
            Decrypted dictionary
        """
        try:
            json_str = self.decrypt_string(encrypted_data)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error after decryption: {e}")
            raise EncryptionError(f"Decrypted data is not valid JSON: {e}")
        except Exception as e:
            logger.error(f"Dictionary decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt dictionary: {e}")
    
    def encrypt_file(self, file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Encrypt a file.
        
        Args:
            file_path: Path to file to encrypt
            output_path: Output path for encrypted file (default: adds .enc extension)
            
        Returns:
            Path to encrypted file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise EncryptionError(f"File not found: {file_path}")
        
        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + '.enc')
        else:
            output_path = Path(output_path)
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Encrypt
            encrypted = self._cipher.encrypt(data)
            
            # Write encrypted content
            with open(output_path, 'wb') as f:
                f.write(encrypted)
            
            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt file: {e}")
    
    def decrypt_file(self, encrypted_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Decrypt a file.
        
        Args:
            encrypted_path: Path to encrypted file
            output_path: Output path for decrypted file (default: removes .enc extension)
            
        Returns:
            Path to decrypted file
        """
        encrypted_path = Path(encrypted_path)
        if not encrypted_path.exists():
            raise EncryptionError(f"Encrypted file not found: {encrypted_path}")
        
        if output_path is None:
            if encrypted_path.suffix == '.enc':
                output_path = encrypted_path.with_suffix('')
            else:
                output_path = encrypted_path.with_suffix('.decrypted')
        else:
            output_path = Path(output_path)
        
        try:
            # Read encrypted content
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted = self._cipher.decrypt(encrypted_data)
            
            # Write decrypted content
            with open(output_path, 'wb') as f:
                f.write(decrypted)
            
            logger.info(f"File decrypted: {encrypted_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt file: {e}")
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if not provided)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt


class SecureStorage:
    """Provides secure storage for sensitive data"""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """
        Initialize secure storage.
        
        Args:
            storage_path: Path to secure storage directory
        """
        if storage_path is None:
            storage_path = Path.home() / ".safla" / "secure_storage"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions on storage directory
        try:
            os.chmod(self.storage_path, 0o700)
        except Exception as e:
            logger.warning(f"Failed to set storage permissions: {e}")
        
        self.encryptor = DataEncryptor()
        logger.info(f"Secure storage initialized at {self.storage_path}")
    
    def store(self, key: str, data: Any, encrypt: bool = True) -> None:
        """
        Store data securely.
        
        Args:
            key: Storage key/identifier
            data: Data to store
            encrypt: Whether to encrypt the data
        """
        # Validate key
        if not re.match(r'^[a-zA-Z0-9_\-]+$', key):
            raise ValueError("Invalid storage key format")
        
        file_path = self.storage_path / f"{key}.dat"
        
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data)
            else:
                serialized = str(data)
            
            # Encrypt if requested
            if encrypt:
                content = self.encryptor.encrypt_string(serialized)
            else:
                content = serialized
            
            # Write to file with restricted permissions
            file_path.touch(mode=0o600)
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Data stored securely: {key}")
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise EncryptionError(f"Failed to store data: {e}")
    
    def retrieve(self, key: str, decrypt: bool = True) -> Any:
        """
        Retrieve data from secure storage.
        
        Args:
            key: Storage key/identifier
            decrypt: Whether to decrypt the data
            
        Returns:
            Retrieved data
        """
        file_path = self.storage_path / f"{key}.dat"
        
        if not file_path.exists():
            raise KeyError(f"No data found for key: {key}")
        
        try:
            # Read content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Decrypt if needed
            if decrypt:
                serialized = self.encryptor.decrypt_string(content)
            else:
                serialized = content
            
            # Deserialize
            try:
                return json.loads(serialized)
            except json.JSONDecodeError:
                return serialized
                
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            raise EncryptionError(f"Failed to retrieve data: {e}")
    
    def delete(self, key: str) -> None:
        """
        Delete data from secure storage.
        
        Args:
            key: Storage key/identifier
        """
        file_path = self.storage_path / f"{key}.dat"
        
        if file_path.exists():
            try:
                # Overwrite with random data before deletion
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_path.stat().st_size))
                
                # Delete file
                file_path.unlink()
                logger.info(f"Data deleted: {key}")
                
            except Exception as e:
                logger.error(f"Failed to delete data: {e}")
                raise EncryptionError(f"Failed to delete data: {e}")
    
    def list_keys(self) -> list[str]:
        """List all stored keys"""
        keys = []
        for file_path in self.storage_path.glob("*.dat"):
            keys.append(file_path.stem)
        return sorted(keys)