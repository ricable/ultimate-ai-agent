FROM node:22-slim

WORKDIR /app

# Install agentic-flow from npm
RUN npm install -g agentic-flow@1.1.10

# Test script
COPY docker/test-validation.sh /app/test-validation.sh
RUN chmod +x /app/test-validation.sh

CMD ["/app/test-validation.sh"]
