# ruv-swarm Test Application Backend

A comprehensive Node.js backend demonstrating all ruv-swarm features with Express, JWT authentication, SQLite database, and monitoring capabilities.

## Features

### Security & Authentication
- **JWT Authentication**: Secure token-based authentication with configurable expiration
- **Password Hashing**: bcrypt-based password hashing with salt rounds
- **Security Headers**: Helmet.js for comprehensive security headers
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Rate Limiting**: IP-based rate limiting to prevent abuse
- **Input Validation**: express-validator for request validation

### Database
- **SQLite Integration**: Lightweight, file-based database
- **User Management**: Complete user CRUD operations
- **Session Tracking**: JWT session management
- **API Logging**: Comprehensive request/response logging

### Monitoring & Observability
- **Prometheus Metrics**: 
  - HTTP request duration histogram
  - Total request counter
  - Default Node.js metrics
- **Winston Logging**:
  - Structured JSON logging
  - Multiple log levels
  - File and console transports
- **Health Checks**: Dedicated health endpoint with database connectivity check

### Error Handling
- **Centralized Error Handler**: Consistent error responses
- **Environment-based Responses**: Production-safe error messages
- **Stack Traces**: Development-only stack trace exposure

## Setup

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Database Setup**
   The database will be automatically initialized on first run.

4. **Start the Server**
   ```bash
   # Production
   npm start

   # Development (with auto-reload)
   npm run dev
   ```

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
  ```json
  {
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password"
  }
  ```

- `POST /api/auth/login` - User login
  ```json
  {
    "email": "john@example.com",
    "password": "secure_password"
  }
  ```

### User Management
- `GET /api/users` - List all users (requires authentication)
- `GET /api/users/:id` - Get user details (requires authentication)
- `PUT /api/users/:id` - Update user (requires authentication)
- `DELETE /api/users/:id` - Delete user (requires authentication)

### Monitoring
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Testing

```bash
# Run all tests
npm test

# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# Coverage report
npm test -- --coverage
```

## Project Structure

```
test-swarm/
├── src/
│   ├── server.js           # Main application entry
│   ├── middleware/         # Express middleware
│   │   ├── auth.js        # JWT authentication
│   │   ├── errorHandler.js # Error handling
│   │   └── metrics.js     # Prometheus metrics
│   ├── models/            # Database models
│   │   ├── User.js        # User model
│   │   └── database.js    # Database connection
│   ├── routes/            # API routes
│   │   ├── auth.js        # Authentication routes
│   │   ├── health.js      # Health check routes
│   │   └── users.js       # User management routes
│   └── utils/             # Utility functions
│       └── logger.js      # Winston logger setup
├── tests/                 # Test suites
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── logs/                 # Log files (gitignored)
└── database.sqlite       # SQLite database (gitignored)
```

## Security Considerations

1. **JWT Secret**: Always use a strong, unique secret in production
2. **HTTPS**: Deploy behind HTTPS in production
3. **Environment Variables**: Never commit sensitive data
4. **Rate Limiting**: Adjust limits based on your use case
5. **Input Validation**: Always validate and sanitize user input

## Monitoring

### Prometheus Metrics
Available at `/metrics` endpoint:
- `http_request_duration_seconds` - Request duration histogram
- `http_requests_total` - Total request counter
- Default Node.js metrics (memory, CPU, etc.)

### Logging
Logs are written to:
- `./logs/error.log` - Error-level logs only
- `./logs/combined.log` - All logs
- Console output in development mode

## ruv-swarm Integration

The backend is designed to work seamlessly with ruv-swarm features:
- Memory persistence for session management
- Neural agent coordination for request optimization
- Swarm orchestration for distributed processing
- Performance metrics for swarm monitoring

Enable ruv-swarm features by setting:
```env
RUV_SWARM_ENABLED=true
```

## Development

### Code Style
```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Adding New Features
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## Deployment

### Production Checklist
- [ ] Set `NODE_ENV=production`
- [ ] Generate strong JWT secret
- [ ] Configure proper database path
- [ ] Set up log rotation
- [ ] Configure reverse proxy (nginx/Apache)
- [ ] Enable HTTPS
- [ ] Set up monitoring alerts
- [ ] Configure backup strategy

## License

MIT