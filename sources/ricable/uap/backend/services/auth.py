# File: backend/services/auth.py
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Tuple
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr, validator
import uuid
import os
import secrets

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Models
class UserRole(BaseModel):
    """User role with permissions"""
    name: str
    permissions: List[str]
    description: Optional[str] = None


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    roles: List[str] = ["user"]  # Role names
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Dict = {}
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)


class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    roles: List[str] = ["user"]
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (with _ and - allowed)')
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class UserInDB(User):
    """User model with hashed password for database storage"""
    hashed_password: str


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiration


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime
    token_type: str  # "access" or "refresh"


# Default roles and permissions
DEFAULT_ROLES = {
    "admin": UserRole(
        name="admin",
        permissions=[
            "user:create", "user:read", "user:update", "user:delete",
            "agent:create", "agent:read", "agent:update", "agent:delete",
            "system:admin", "system:manage", "system:read", "websocket:connect"
        ],
        description="Full system administrator access"
    ),
    "manager": UserRole(
        name="manager",
        permissions=[
            "user:read", "user:update",
            "agent:create", "agent:read", "agent:update",
            "system:manage", "websocket:connect"
        ],
        description="Manager with user and agent management capabilities"
    ),
    "user": UserRole(
        name="user",
        permissions=[
            "agent:read", "agent:create",
            "websocket:connect"
        ],
        description="Standard user with basic agent access"
    ),
    "guest": UserRole(
        name="guest",
        permissions=[
            "agent:read"
        ],
        description="Read-only guest access"
    )
}


class AuthService:
    """Authentication service for JWT and RBAC"""
    
    def __init__(self):
        self.users_db: Dict[str, UserInDB] = {}  # In-memory storage for demo
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id mapping
        self.roles = DEFAULT_ROLES.copy()
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        admin_username = "admin"
        admin_email = "admin@example.com"
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123!")
        
        if not any(user.username == admin_username for user in self.users_db.values()):
            admin_user = UserCreate(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                full_name="System Administrator",
                roles=["admin"]
            )
            try:
                self.create_user(admin_user)
                print(f"Default admin user created: {admin_username}")
            except Exception as e:
                print(f"Failed to create default admin user: {e}")
    
    # Password utilities
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    # User management
    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user"""
        # Check if username or email already exists
        for existing_user in self.users_db.values():
            if existing_user.username == user_create.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
            if existing_user.email == user_create.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Validate roles
        for role in user_create.roles:
            if role not in self.roles:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid role: {role}"
                )
        
        # Create user
        user_id = str(uuid.uuid4())
        hashed_password = self.get_password_hash(user_create.password)
        
        user_in_db = UserInDB(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            roles=user_create.roles,
            hashed_password=hashed_password,
            created_at=datetime.now(timezone.utc)
        )
        
        self.users_db[user_id] = user_in_db
        
        # Return user without password
        return User(**user_in_db.dict(exclude={"hashed_password"}))
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username/password"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        for user in self.users_db.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        return self.users_db.get(user_id)
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        if user_id in self.users_db:
            self.users_db[user_id].last_login = datetime.now(timezone.utc)
    
    # JWT Token management
    def create_access_token(self, user: UserInDB) -> str:
        """Create access token"""
        permissions = self.get_user_permissions(user)
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "token_type": "access"
        }
        
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    def create_refresh_token(self, user: UserInDB) -> str:
        """Create refresh token"""
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "token_type": "refresh"
        }
        
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        self.refresh_tokens[token] = user.id
        return token
    
    def create_tokens(self, user: UserInDB) -> Token:
        """Create both access and refresh tokens"""
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def decode_token(self, token: str) -> TokenData:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check expiration
            exp = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            if datetime.now(timezone.utc) >= exp:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return TokenData(
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                exp=exp,
                iat=datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc),
                token_type=payload.get("token_type", "access")
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> Token:
        """Refresh access token using refresh token"""
        if refresh_token not in self.refresh_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        try:
            # Decode refresh token
            token_data = self.decode_token(refresh_token)
            
            if token_data.token_type != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Get user
            user = self.get_user_by_id(token_data.user_id)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Create new tokens
            return self.create_tokens(user)
            
        except HTTPException:
            # Remove invalid refresh token
            if refresh_token in self.refresh_tokens:
                del self.refresh_tokens[refresh_token]
            raise
    
    def revoke_refresh_token(self, refresh_token: str):
        """Revoke refresh token"""
        if refresh_token in self.refresh_tokens:
            del self.refresh_tokens[refresh_token]
    
    # RBAC (Role-Based Access Control)
    def get_user_permissions(self, user: UserInDB) -> List[str]:
        """Get all permissions for a user based on their roles"""
        permissions = set()
        for role_name in user.roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
        return list(permissions)
    
    def check_permission(self, user: UserInDB, permission: str) -> bool:
        """Check if user has a specific permission"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def require_permission(self, user: UserInDB, permission: str):
        """Require user to have specific permission, raise exception if not"""
        if not self.check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required"
            )
    
    def add_role(self, role: UserRole):
        """Add a new role"""
        self.roles[role.name] = role
    
    def get_role(self, role_name: str) -> Optional[UserRole]:
        """Get role by name"""
        return self.roles.get(role_name)
    
    def get_all_roles(self) -> Dict[str, UserRole]:
        """Get all available roles"""
        return self.roles.copy()
    
    # User authentication flow
    def login(self, login_data: UserLogin) -> Tuple[Token, User]:
        """Login user and return tokens and user info"""
        user = self.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        # Update last login
        self.update_last_login(user.id)
        
        # Create tokens
        tokens = self.create_tokens(user)
        user_info = User(**user.dict(exclude={"hashed_password"}))
        
        return tokens, user_info
    
    def logout(self, refresh_token: str):
        """Logout user by revoking refresh token"""
        self.revoke_refresh_token(refresh_token)
    
    # Current user utilities
    def get_current_user_from_token(self, token: str) -> UserInDB:
        """Get current user from access token"""
        token_data = self.decode_token(token)
        
        if token_data.token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user = self.get_user_by_id(token_data.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        return user


# Global auth service instance
auth_service = AuthService()