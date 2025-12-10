/**
 * Zero-Copy Tensor Operations Example
 *
 * Demonstrates efficient zero-copy tensor operations using napi::Buffer.
 */

const {
  createTensorBuffer,
  tensorFromBuffer,
  concatenateTensors,
  splitTensor,
  TensorBuffer
} = require('..');

function printTensorInfo(name, tensor) {
  console.log(`\n${name}:`);
  console.log(`  Shape: [${tensor.shape.join(', ')}]`);
  console.log(`  Elements: ${tensor.numElements()}`);
  console.log(`  Bytes: ${tensor.byteSize()}`);
  console.log(`  Dtype: ${tensor.dtype}`);
}

function main() {
  console.log('⚡ Prime ML NAPI - Zero-Copy Tensor Operations\n');
  console.log('='.repeat(50));

  try {
    // 1. Create tensor from array (with copy)
    console.log('\n1️⃣  Creating tensor from array...');
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const tensor1 = createTensorBuffer(data, [3, 4]);
    printTensorInfo('Tensor 1', tensor1);
    console.log(`  Data (first 6): ${tensor1.toF32Array().slice(0, 6).join(', ')}`);

    // 2. Create tensor from existing buffer (zero-copy)
    console.log('\n2️⃣  Creating tensor from buffer (zero-copy)...');
    const float32Array = new Float32Array([10, 20, 30, 40]);
    const buffer = Buffer.from(float32Array.buffer);
    const tensor2 = tensorFromBuffer(buffer, [2, 2], 'f32');
    printTensorInfo('Tensor 2', tensor2);
    console.log(`  Data: ${tensor2.toF32Array().join(', ')}`);

    // 3. Reshape tensor (zero-copy)
    console.log('\n3️⃣  Reshaping tensor (zero-copy)...');
    const reshaped = tensor1.reshape([4, 3]);
    printTensorInfo('Reshaped Tensor', reshaped);
    console.log('  ✓ Same underlying buffer, different view');

    // 4. Concatenate tensors
    console.log('\n4️⃣  Concatenating tensors...');
    const t1 = createTensorBuffer([1, 2, 3], [3]);
    const t2 = createTensorBuffer([4, 5, 6], [3]);
    const t3 = createTensorBuffer([7, 8, 9], [3]);
    const concatenated = concatenateTensors([t1, t2, t3], 0);
    printTensorInfo('Concatenated Tensor', concatenated);
    console.log(`  Data: ${concatenated.toF32Array().join(', ')}`);

    // 5. Split tensor
    console.log('\n5️⃣  Splitting tensor...');
    const bigTensor = createTensorBuffer(
      Array.from({ length: 12 }, (_, i) => i + 1),
      [12]
    );
    const splits = splitTensor(bigTensor, 3);
    console.log(`  Split into ${splits.length} tensors:`);
    splits.forEach((split, i) => {
      console.log(`    Split ${i + 1}: [${split.toF32Array().join(', ')}]`);
    });

    // 6. Access raw buffer (zero-copy)
    console.log('\n6️⃣  Accessing raw buffer (zero-copy)...');
    const rawBuffer = tensor2.buffer;
    console.log(`  Buffer type: ${rawBuffer.constructor.name}`);
    console.log(`  Buffer length: ${rawBuffer.length} bytes`);
    console.log('  ✓ Direct memory access, no copy!');

    // 7. Large tensor operations (performance test)
    console.log('\n7️⃣  Performance test with large tensors...');
    const largeSize = 1000000; // 1 million elements
    const startTime = Date.now();

    const largeTensor = createTensorBuffer(
      Array.from({ length: largeSize }, () => Math.random()),
      [largeSize]
    );
    const createTime = Date.now() - startTime;

    const reshapeStart = Date.now();
    const reshaped1000x1000 = largeTensor.reshape([1000, 1000]);
    const reshapeTime = Date.now() - reshapeStart;

    console.log(`  Created tensor with ${largeSize.toLocaleString()} elements in ${createTime}ms`);
    console.log(`  Reshaped to [1000, 1000] in ${reshapeTime}ms (zero-copy!)`);
    printTensorInfo('Large Tensor', largeTensor);

    // 8. Working with different data types
    console.log('\n8️⃣  Different data types...');

    // f32 (single precision float)
    const f32Data = new Float32Array([1.5, 2.5, 3.5, 4.5]);
    const f32Tensor = tensorFromBuffer(
      Buffer.from(f32Data.buffer),
      [2, 2],
      'f32'
    );
    printTensorInfo('F32 Tensor', f32Tensor);

    // f64 (double precision float)
    const f64Data = new Float64Array([1.5, 2.5, 3.5, 4.5]);
    const f64Tensor = tensorFromBuffer(
      Buffer.from(f64Data.buffer),
      [2, 2],
      'f64'
    );
    printTensorInfo('F64 Tensor', f64Tensor);
    console.log(`  F64 Data: ${f64Tensor.toF64Array().join(', ')}`);

    // 9. Clone tensor (with copy)
    console.log('\n9️⃣  Cloning tensor...');
    const original = createTensorBuffer([1, 2, 3, 4], [2, 2]);
    const cloned = original.cloneTensor();
    printTensorInfo('Cloned Tensor', cloned);
    console.log('  ✓ Independent copy created');

    // 10. Memory efficiency comparison
    console.log('\n🔟 Memory Efficiency Summary:');
    console.log('  Zero-copy operations:');
    console.log('    • tensorFromBuffer()');
    console.log('    • reshape()');
    console.log('    • buffer getter');
    console.log('  Operations with copy:');
    console.log('    • createTensorBuffer()');
    console.log('    • toF32Array() / toF64Array()');
    console.log('    • concatenateTensors()');
    console.log('    • splitTensor()');
    console.log('    • cloneTensor()');

    console.log('\n✓ All tensor operations completed successfully!');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
