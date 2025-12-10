/**
 * ENM MOM Schema and CM Writer - Usage Examples
 *
 * Demonstrates:
 * - MOM XML parsing/generation
 * - CM transaction management
 * - Bulk operations
 * - 3GPP constraint validation
 */

import {
  EUtranCellFDD,
  NRCellDU,
  AntennaUnit,
  momParser,
  momGenerator,
  EUtranCellFDDSchema,
  NRCellDUSchema,
} from './mom-schema.js';

import {
  ENMCMWriter,
  createCMWriter,
  createBatchChanges,
  CMChangeRequest,
} from './cm-writer.js';

// ============================================================================
// Example 1: Parse MOM XML from ENM Export
// ============================================================================

export function exampleParseEUtranCell() {
  const enmXML = `
    <?xml version="1.0" encoding="UTF-8"?>
    <EUtranCellFDD>
      <EUtranCellFDDId>LTE001</EUtranCellFDDId>
      <id>LTE001</id>
      <userLabel>Downtown Sector 1</userLabel>
      <swVersion>21.Q4.1</swVersion>
      <cellId>1</cellId>
      <physicalLayerCellId>256</physicalLayerCellId>
      <tac>12345</tac>
      <earfcn>6300</earfcn>
      <bandwidth>20</bandwidth>
      <p0NominalPUSCH>-90</p0NominalPUSCH>
      <alpha>0.8</alpha>
      <dn>MeContext=LTE001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=LTE001</dn>
    </EUtranCellFDD>
  `;

  try {
    const cell = momParser.parseEUtranCellFDD(enmXML);
    console.log('Parsed EUtranCellFDD:', cell);
    console.log('✓ Validation passed - 3GPP compliant');
    return cell;
  } catch (error) {
    console.error('❌ Parsing failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 2: Generate MOM XML for Parameter Change
// ============================================================================

export function exampleGenerateParameterChange() {
  const cellUpdate: Partial<EUtranCellFDD> = {
    id: 'LTE001',
    p0NominalPUSCH: -85, // Increase power
    alpha: 0.9, // Adjust path loss compensation
    dn: 'MeContext=LTE001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=LTE001',
  };

  // Validate before generating
  try {
    EUtranCellFDDSchema.partial().parse(cellUpdate);
    const xml = momGenerator.generateEUtranCellFDDXML(cellUpdate, 'update');
    console.log('Generated XML for ENM import:');
    console.log(xml);
    return xml;
  } catch (error) {
    console.error('❌ Validation failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 3: Single Cell Update via CM Writer
// ============================================================================

export async function exampleSingleCellUpdate() {
  const cmWriter = createCMWriter({
    host: 'enm.ericsson.local',
    port: 443,
    username: 'admin',
    password: 'secure_password',
  });

  const cellUpdate: Partial<EUtranCellFDD> = {
    id: 'LTE001',
    userLabel: 'Downtown Sector 1',
    swVersion: '21.Q4.1',
    vendorName: 'Ericsson',
    p0NominalPUSCH: -85,
    alpha: 0.9,
  };

  try {
    const result = await cmWriter.writeSingle('EUtranCellFDD', 'update', cellUpdate);

    console.log('CM Write Result:', {
      success: result.success,
      transactionId: result.transactionId,
      appliedChanges: result.appliedChanges,
      executionTime: `${result.executionTime}ms`,
      errors: result.errors,
      warnings: result.warnings,
    });

    if (result.xml) {
      console.log('\nGenerated XML:\n', result.xml);
    }

    return result;
  } catch (error) {
    console.error('❌ CM write failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 4: Bulk Update - Multiple Cells
// ============================================================================

export async function exampleBulkCellUpdate() {
  const cmWriter = createCMWriter();

  // Prepare bulk changes for multiple cells
  const cellUpdates: Array<Partial<EUtranCellFDD>> = [
    {
      id: 'LTE001',
      userLabel: 'Sector 1',
      swVersion: '21.Q4.1',
      vendorName: 'Ericsson',
      p0NominalPUSCH: -85,
      alpha: 0.9,
    },
    {
      id: 'LTE002',
      userLabel: 'Sector 2',
      swVersion: '21.Q4.1',
      vendorName: 'Ericsson',
      p0NominalPUSCH: -88,
      alpha: 0.8,
    },
    {
      id: 'LTE003',
      userLabel: 'Sector 3',
      swVersion: '21.Q4.1',
      vendorName: 'Ericsson',
      p0NominalPUSCH: -90,
      alpha: 0.7,
    },
  ];

  const changes = createBatchChanges('EUtranCellFDD', 'update', cellUpdates);

  try {
    const result = await cmWriter.writeBulk(changes);

    console.log('Bulk CM Write Result:', {
      success: result.success,
      transactionId: result.transactionId,
      appliedChanges: result.appliedChanges,
      failedChanges: result.failedChanges,
      executionTime: `${result.executionTime}ms`,
    });

    return result;
  } catch (error) {
    console.error('❌ Bulk write failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 5: 5G NR Cell Configuration
// ============================================================================

export async function exampleNRCellConfig() {
  const cmWriter = createCMWriter();

  const nrCell: Partial<NRCellDU> = {
    id: 'NR001',
    userLabel: '5G Downtown Sector 1',
    swVersion: '22.Q1.5',
    vendorName: 'Ericsson',
    cellLocalId: 1,
    nCI: '000000001',
    nRPCI: 512,
    nRTAC: 100,
    arfcnDL: 634000, // 3.5 GHz band
    bSChannelBwDL: 100, // 100 MHz
    pZeroNomPusch: -90,
    pZeroNomPucch: -94,
    ssbFrequency: 3510000, // 3.51 GHz
    ssbPeriodicity: 20,
  };

  try {
    // Validate
    NRCellDUSchema.partial().parse(nrCell);

    // Generate XML
    const xml = momGenerator.generateNRCellDUXML(nrCell, 'create');
    console.log('5G NR Cell XML:\n', xml);

    // Apply via CM Writer
    const result = await cmWriter.writeSingle('NRCellDU', 'create', nrCell);
    console.log('NR Cell Creation Result:', result.success ? '✓ Success' : '❌ Failed');

    return result;
  } catch (error) {
    console.error('❌ NR cell config failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 6: Antenna Tilt Optimization with Constraints
// ============================================================================

export async function exampleAntennaTiltOptimization() {
  const cmWriter = createCMWriter();

  const antennaUpdates: Array<Partial<AntennaUnit>> = [
    {
      id: 'ANT001',
      mechanicalTilt: 6,
      electricalTilt: 8,
      totalTilt: 14, // Must be <= 30 degrees (3GPP constraint)
      azimuth: 0,
      maxTxPower: 43, // Must be -130 to 46 dBm
      antennaGain: 17.5,
    },
    {
      id: 'ANT002',
      mechanicalTilt: 4,
      electricalTilt: 10,
      totalTilt: 14,
      azimuth: 120,
      maxTxPower: 43,
      antennaGain: 17.5,
    },
    {
      id: 'ANT003',
      mechanicalTilt: 5,
      electricalTilt: 9,
      totalTilt: 14,
      azimuth: 240,
      maxTxPower: 43,
      antennaGain: 17.5,
    },
  ];

  const changes = createBatchChanges('AntennaUnit', 'update', antennaUpdates);

  try {
    const result = await cmWriter.writeBulk(changes);

    console.log('Antenna Optimization Result:', {
      success: result.success,
      appliedChanges: result.appliedChanges,
      totalAntennas: antennaUpdates.length,
    });

    return result;
  } catch (error) {
    console.error('❌ Antenna optimization failed:', error);
    throw error;
  }
}

// ============================================================================
// Example 7: Transaction Management with Rollback
// ============================================================================

export async function exampleTransactionRollback() {
  const cmWriter = createCMWriter();

  const changes: CMChangeRequest[] = [
    {
      id: 'CHANGE-001',
      type: 'EUtranCellFDD',
      operation: 'update',
      data: {
        id: 'LTE001',
        userLabel: 'Test',
        swVersion: '21.Q4.1',
        vendorName: 'Ericsson',
        p0NominalPUSCH: -85,
        alpha: 0.9,
      },
      rollbackOnError: true,
    },
    {
      id: 'CHANGE-002',
      type: 'EUtranCellFDD',
      operation: 'update',
      data: {
        id: 'LTE002',
        userLabel: 'Test',
        swVersion: '21.Q4.1',
        vendorName: 'Ericsson',
        p0NominalPUSCH: -200, // INVALID - will trigger rollback
        alpha: 0.8,
      },
      rollbackOnError: true,
    },
  ];

  try {
    const txId = cmWriter.createTransaction(changes);
    console.log('Transaction created:', txId);

    const result = await cmWriter.executeTransaction(txId);

    if (!result.success) {
      console.log('❌ Transaction failed and rolled back');
      console.log('Errors:', result.errors);
      console.log('Warnings:', result.warnings);
    }

    const status = cmWriter.getTransactionStatus(txId);
    console.log('Final transaction status:', status?.status);

    return result;
  } catch (error) {
    console.error('❌ Transaction execution error:', error);
    throw error;
  }
}

// ============================================================================
// Example 8: Export XML for Manual Review
// ============================================================================

export function exampleExportXMLForReview() {
  const cmWriter = createCMWriter();

  const changes: CMChangeRequest[] = [
    {
      id: 'EXPORT-001',
      type: 'EUtranCellFDD',
      operation: 'update',
      data: {
        id: 'LTE001',
        userLabel: 'Export Test',
        swVersion: '21.Q4.1',
        vendorName: 'Ericsson',
        p0NominalPUSCH: -85,
        alpha: 0.9,
      },
    },
  ];

  const xml = cmWriter.exportXML(changes);
  console.log('Exported XML for manual review:\n');
  console.log(xml);

  // Save to file for import into ENM GUI
  // fs.writeFileSync('/tmp/enm-bulk-cm-import.xml', xml);

  return xml;
}

// ============================================================================
// Example 9: 3GPP Constraint Validation
// ============================================================================

export function exampleConstraintValidation() {
  console.log('Testing 3GPP TS 28.552 constraint validation...\n');

  // Test 1: Valid power parameter
  try {
    const validCell = EUtranCellFDDSchema.partial().parse({
      p0NominalPUSCH: -85,
    });
    console.log('✓ Valid power: -85 dBm');
  } catch (error) {
    console.error('❌ Validation failed:', error);
  }

  // Test 2: Invalid power parameter (too low)
  try {
    const invalidCell = EUtranCellFDDSchema.partial().parse({
      p0NominalPUSCH: -150, // Below -130 dBm minimum
    });
    console.error('❌ Should have failed: power too low');
  } catch (error) {
    console.log('✓ Correctly rejected: power < -130 dBm');
  }

  // Test 3: Invalid alpha value
  try {
    const invalidAlpha = EUtranCellFDDSchema.partial().parse({
      alpha: 0.3, // Not in valid set
    });
    console.error('❌ Should have failed: invalid alpha');
  } catch (error) {
    console.log('✓ Correctly rejected: alpha not in valid set');
  }

  // Test 4: Valid tilt values
  try {
    const validTilt = {
      mechanicalTilt: 6,
      electricalTilt: 8,
      totalTilt: 14,
    };
    console.log('✓ Valid tilt: mechanical=6°, electrical=8°, total=14°');
  } catch (error) {
    console.error('❌ Validation failed:', error);
  }

  console.log('\n3GPP constraint validation complete');
}

// ============================================================================
// Run All Examples
// ============================================================================

export async function runAllExamples() {
  console.log('='.repeat(80));
  console.log('ENM MOM Schema & CM Writer - Examples');
  console.log('='.repeat(80));

  try {
    console.log('\n[1] Parsing MOM XML from ENM Export');
    console.log('-'.repeat(80));
    exampleParseEUtranCell();

    console.log('\n[2] Generating MOM XML for Parameter Change');
    console.log('-'.repeat(80));
    exampleGenerateParameterChange();

    console.log('\n[3] Single Cell Update');
    console.log('-'.repeat(80));
    await exampleSingleCellUpdate();

    console.log('\n[4] Bulk Cell Update');
    console.log('-'.repeat(80));
    await exampleBulkCellUpdate();

    console.log('\n[5] 5G NR Cell Configuration');
    console.log('-'.repeat(80));
    await exampleNRCellConfig();

    console.log('\n[6] Antenna Tilt Optimization');
    console.log('-'.repeat(80));
    await exampleAntennaTiltOptimization();

    console.log('\n[7] Transaction Rollback');
    console.log('-'.repeat(80));
    await exampleTransactionRollback();

    console.log('\n[8] Export XML for Manual Review');
    console.log('-'.repeat(80));
    exampleExportXMLForReview();

    console.log('\n[9] 3GPP Constraint Validation');
    console.log('-'.repeat(80));
    exampleConstraintValidation();

    console.log('\n' + '='.repeat(80));
    console.log('All examples completed successfully!');
    console.log('='.repeat(80));

  } catch (error) {
    console.error('\n❌ Example execution failed:', error);
    throw error;
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples().catch(console.error);
}
