# QuDAG Native CLI & MCP Guide

Complete guide for using the QuDAG Native command-line interface and Model Context Protocol server.

## 🚀 Installation

```bash
npm install @daa/qudag-native

# Or use with npx (no installation needed)
npx @daa/qudag-native --help
```

## 📦 CLI Commands

### Module Information

```bash
qudag info
```

Shows module version, features, and capabilities.

### ML-KEM-768 (Quantum-Resistant Encryption)

**Generate Keypair:**
```bash
qudag mlkem768 keygen
qudag mlkem768 keygen -o ./keys -n mykey
```

**Encrypt (Encapsulate):**
```bash
qudag mlkem768 encrypt -p mlkem768.pub
qudag mlkem768 encrypt -p mlkem768.pub -o ./encrypted
```

**Decrypt (Decapsulate):**
```bash
qudag mlkem768 decrypt -c ciphertext.bin -s mlkem768.key
```

### ML-DSA-65 (Quantum-Resistant Signatures)

**Generate Keypair:**
```bash
qudag mldsa65 keygen
qudag mldsa65 keygen -o ./keys -n signer
```

**Sign Message:**
```bash
qudag mldsa65 sign -m message.txt -s mldsa65.key
qudag mldsa65 sign -m message.txt -s mldsa65.key -o signature.bin
```

**Verify Signature:**
```bash
qudag mldsa65 verify -m message.txt -g signature.bin -p mldsa65.pub
# Exit code 0 = valid, 1 = invalid
```

### BLAKE3 Hashing

**Hash File:**
```bash
qudag blake3 hash -f document.pdf
qudag blake3 hash -f document.pdf --hex
```

**Generate Fingerprint:**
```bash
qudag blake3 fingerprint -f myfile.bin
# Output: qf:abc123...
```

### Random Bytes

**Generate Random Data:**
```bash
qudag random -n 32
qudag random -n 64 -o random.bin
```

## 🔌 MCP Server

The QuDAG Native MCP server provides quantum cryptography operations via the Model Context Protocol.

### Start STDIO Server

```bash
qudag mcp
# Or explicitly:
qudag mcp -t stdio
```

### Start SSE Server

```bash
qudag mcp -t sse -p 3000
```

Server endpoints:
- `GET /sse` - SSE connection
- `POST /message` - Send MCP messages
- `GET /health` - Health check

## 🛠️ Available MCP Tools

### 1. `mlkem768_keygen`
Generate ML-KEM-768 keypair.

**Input:** None
**Output:**
```json
{
  "publicKey": "hex...",
  "secretKey": "hex...",
  "fingerprint": "qf:..."
}
```

### 2. `mlkem768_encapsulate`
Encapsulate shared secret.

**Input:**
```json
{
  "publicKey": "hex-encoded public key (1184 bytes)"
}
```

**Output:**
```json
{
  "ciphertext": "hex...",
  "sharedSecret": "hex..."
}
```

### 3. `mlkem768_decapsulate`
Decapsulate shared secret.

**Input:**
```json
{
  "ciphertext": "hex...",
  "secretKey": "hex..."
}
```

**Output:**
```json
{
  "sharedSecret": "hex..."
}
```

### 4. `mldsa65_keygen`
Generate ML-DSA-65 keypair.

**Input:** None
**Output:**
```json
{
  "publicKey": "hex...",
  "secretKey": "hex...",
  "fingerprint": "qf:..."
}
```

### 5. `mldsa65_sign`
Sign message with ML-DSA-65.

**Input:**
```json
{
  "message": "hex...",
  "secretKey": "hex..."
}
```

**Output:**
```json
{
  "signature": "hex..."
}
```

### 6. `mldsa65_verify`
Verify ML-DSA-65 signature.

**Input:**
```json
{
  "message": "hex...",
  "signature": "hex...",
  "publicKey": "hex..."
}
```

**Output:**
```json
{
  "valid": true
}
```

### 7. `blake3_hash`
Compute BLAKE3 hash.

**Input:**
```json
{
  "data": "hex..."
}
```

**Output:**
```json
{
  "hash": "hex..."
}
```

### 8. `quantum_fingerprint`
Generate quantum-resistant fingerprint.

**Input:**
```json
{
  "data": "hex..."
}
```

**Output:**
```json
{
  "fingerprint": "qf:..."
}
```

### 9. `random_bytes`
Generate random bytes.

**Input:**
```json
{
  "length": 32
}
```

**Output:**
```json
{
  "bytes": "hex..."
}
```

## 🔗 MCP Integration Example

### Using with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qudag-native": {
      "command": "npx",
      "args": ["@daa/qudag-native", "mcp"]
    }
  }
}
```

### Manual Testing (STDIO)

```bash
# Initialize
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | qudag mcp

# List tools
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | qudag mcp

# Generate keypair
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mlkem768_keygen","arguments":{}}}' | qudag mcp
```

### SSE Transport Example

```javascript
const evtSource = new EventSource('http://localhost:3000/sse');

evtSource.addEventListener('connected', (e) => {
  console.log('Connected:', JSON.parse(e.data));
});

// Send message
fetch('http://localhost:3000/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'tools/call',
    params: {
      name: 'mlkem768_keygen',
      arguments: {}
    },
    id: 1
  })
}).then(r => r.json()).then(console.log);
```

## 📊 Performance

- **ML-KEM-768 Keygen:** ~1-2ms
- **ML-KEM-768 Encapsulate:** ~1-2ms
- **ML-KEM-768 Decapsulate:** ~1-2ms
- **ML-DSA-65 Sign:** ~2-3ms
- **ML-DSA-65 Verify:** ~2-3ms
- **BLAKE3 Hash:** <1ms per MB

All operations use native Rust implementations for maximum performance.

## 🔒 Security Features

✅ **NIST FIPS 203 Compliant** - ML-KEM-768 (Kyber)
✅ **NIST FIPS 204 Compliant** - ML-DSA-65 (Dilithium3)
✅ **IND-CCA2 Secure** - Encryption
✅ **EUF-CMA Secure** - Signatures
✅ **Zero-Copy Operations** - Minimal memory overhead
✅ **Constant-Time Comparisons** - Side-channel resistant
✅ **Cryptographically Secure RNG** - OS-level entropy

## 📚 Additional Resources

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [ML-KEM Specification (FIPS 203)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf)
- [ML-DSA Specification (FIPS 204)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.204.pdf)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 📄 License

MIT OR Apache-2.0
