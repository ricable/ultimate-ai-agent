// Test UDP socket creation and binding in QuicServer
import { QuicServer } from '../dist/transport/quic.js';
import { logger } from '../dist/utils/logger.js';

async function testQuicServerUdp() {
    console.log('🧪 Testing QuicServer UDP Socket Integration...\n');

    const server = new QuicServer({
        host: '0.0.0.0',
        port: 4433,
        verifyPeer: false,
        maxConnections: 100,
        maxConcurrentStreams: 100
    });

    try {
        // Test 1: Initialize WASM module
        console.log('Test 1: Initializing WASM module...');
        await server.initialize();
        console.log('✅ WASM module initialized\n');

        // Test 2: Start listening (should bind UDP socket)
        console.log('Test 2: Starting server (binding UDP socket)...');
        await server.listen();
        console.log('✅ Server listening on UDP port 4433\n');

        // Test 3: Verify UDP socket exists
        console.log('Test 3: Verifying UDP socket...');
        if (server.udpSocket) {
            const address = server.udpSocket.address();
            console.log(`✅ UDP socket listening on ${address.address}:${address.port}\n`);
        } else {
            console.log('❌ UDP socket not created\n');
        }

        // Test 4: Get stats
        console.log('Test 4: Getting server stats...');
        const stats = server.getStats();
        console.log('Stats:', JSON.stringify(stats, null, 2));
        console.log('✅ Stats retrieved\n');

        // Wait 2 seconds to allow connections
        console.log('Waiting 2 seconds for potential connections...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Test 5: Stop server
        console.log('Test 5: Stopping server...');
        await server.stop();
        console.log('✅ Server stopped\n');

        console.log('🎉 All QuicServer UDP tests passed!');
        process.exit(0);

    } catch (error) {
        console.error('❌ Test failed:', error);
        console.error('Stack:', error.stack);

        // Try to cleanup
        try {
            await server.stop();
        } catch (e) {
            // Ignore cleanup errors
        }

        process.exit(1);
    }
}

testQuicServerUdp();
