# Docker WASM Test Report

**Date:** Wed Jul  2 23:56:29 UTC 2025
**Total Tests:** 
**Passed:** 0
0  
**Failed:** 4

## Test Details

Test Summary - Wed Jul  2 23:56:00 UTC 2025
========================
NPM Installation Test: FAILED
Global Installation Test: FAILED
Production Simulation: FAILED
All Tests Combined: FAILED

## WASM Verification

The tests specifically verify:
1. WASM files are present in the npm package
2. WASM binary has correct magic number (\0asm)
3. WASM module loads without falling back to placeholder
4. Memory usage indicates real WASM is running
5. All npx commands work correctly

## Recommendations

⚠️ Some tests failed. Check individual test outputs for details.
