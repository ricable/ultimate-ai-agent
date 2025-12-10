// Example: Using SAFLA via MCP in JavaScript/Node.js

const axios = require('axios');

async function useSAFLAMCP() {
    const mcpRequest = {
        jsonrpc: "2.0",
        id: "embed_001",
        method: "generate_embeddings",
        params: {
            texts: ["Hello SAFLA!", "This is a test"],
            config: { batch_size: 256, cache_embeddings: true }
        }
    };
    
    try {
        const response = await axios.post(
            'https://safla.fly.dev/api/safla',
            mcpRequest
        );
        console.log('Embeddings generated:', response.data);
    } catch (error) {
        console.error('Error:', error);
    }
}

useSAFLAMCP();
