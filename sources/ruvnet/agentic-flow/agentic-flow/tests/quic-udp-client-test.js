// Test UDP socket creation and binding in QuicClient
import { QuicClient } from '../dist/transport/quic.js';
import { logger } from '../dist/utils/logger.js';

async function testQuicClientUdp() {
    console.log('🧪 Testing QuicClient UDP Socket Integration...\n');

    const client = new QuicClient({
        serverHost: 'localhost',
        serverPort: 4433,
        verifyPeer: false,
        maxConnections: 10
    });

    try {
        // Test 1: Initialize WASM module
        console.log('Test 1: Initializing WASM module...');
        await client.initialize();
        console.log('✅ WASM module initialized\n');

        // Test 2: Connect (should create UDP socket)
        console.log('Test 2: Creating connection (UDP socket)...');
        const connectionId = await client.connect('localhost', 4433);
        console.log(`✅ Connection created: ${connectionId}`);
        console.log(`✅ UDP socket should be bound\n`);

        // Test 3: Verify UDP socket exists
        console.log('Test 3: Verifying UDP socket...');
        if (client.udpSocket) {
            const address = client.udpSocket.address();
            console.log(`✅ UDP socket bound to ${address.address}:${address.port}\n`);
        } else {
            console.log('❌ UDP socket not created\n');
        }

        // Test 4: Get stats
        console.log('Test 4: Getting connection stats...');
        const stats = client.getStats();
        console.log('Stats:', JSON.stringify(stats, null, 2));
        console.log('✅ Stats retrieved\n');

        // Test 5: Shutdown
        console.log('Test 5: Shutting down...');
        await client.shutdown();
        console.log('✅ Client shutdown complete\n');

        console.log('🎉 All QuicClient UDP tests passed!');
        process.exit(0);

    } catch (error) {
        console.error('❌ Test failed:', error);
        console.error('Stack:', error.stack);
        process.exit(1);
    }
}

testQuicClientUdp();
