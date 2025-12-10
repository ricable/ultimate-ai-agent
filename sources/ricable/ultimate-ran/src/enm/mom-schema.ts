/**
 * Ericsson ENM MOM (Managed Object Model) XML Schema Generator
 *
 * 3GPP TS 28.622/28.623 NRM Models
 * 3GPP TS 32.616 Bulk CM IRP
 * Ericsson ENM Scripting Interface
 *
 * @module mom-schema
 */

import { z } from 'zod';

// ============================================================================
// 3GPP TS 28.622/28.623 MOM Types
// ============================================================================

/**
 * Base Managed Element (3GPP TS 28.622 GenericNRM)
 */
export interface ManagedElement {
  id: string;
  userLabel: string;
  swVersion: string;
  vendorName: 'Ericsson';
  managedBy?: string;
  locationName?: string;
  dn?: string; // Distinguished Name (LDAP format)
}

/**
 * EUtranCellFDD - LTE FDD Cell (3GPP TS 28.623 EUtranNRM)
 */
export interface EUtranCellFDD extends ManagedElement {
  cellId: number; // 0-255
  physicalLayerCellId: number; // PCI: 0-503
  tac: number; // Tracking Area Code: 0-65535
  earfcn: number; // E-UTRA Absolute Radio Frequency Channel Number
  earfcnDl?: number; // Downlink EARFCN (if different)
  bandwidth: 5 | 10 | 15 | 20; // MHz

  // Power Control Parameters (3GPP TS 36.213)
  p0NominalPUSCH: number; // -130 to -70 dBm
  alpha: 0 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0; // Path loss compensation factor

  // Additional RAN parameters
  qRxLevMin?: number; // -70 to -22 dBm
  pMax?: number; // -30 to 33 dBm
  cellRange?: number; // meters
  administrativeState?: 'LOCKED' | 'UNLOCKED' | 'SHUTTING_DOWN';
  operationalState?: 'ENABLED' | 'DISABLED';
}

/**
 * NRCellDU - 5G NR Cell (3GPP TS 28.541 NR NRM)
 */
export interface NRCellDU extends ManagedElement {
  cellLocalId: number; // 0-16383
  nCI: string; // NR Cell Identity (36 bits)
  nRPCI: number; // Physical Cell ID: 0-1007
  nRTAC: number; // NR Tracking Area Code
  arfcnDL: number; // NR-ARFCN Downlink
  arfcnUL?: number; // NR-ARFCN Uplink
  bSChannelBwDL: number; // Channel bandwidth DL (MHz)
  bSChannelBwUL?: number; // Channel bandwidth UL (MHz)

  // 5G Power Control
  pZeroNomPusch: number; // -202 to 24 dBm
  pZeroNomPucch: number; // -202 to 24 dBm
  msg3DeltaPreamble?: number; // -1 to 6 dB

  // SSB Configuration
  ssbFrequency?: number; // SSB frequency in kHz
  ssbPeriodicity?: 5 | 10 | 20 | 40 | 80 | 160; // ms
  ssbDuration?: 1 | 2 | 3 | 4 | 5; // symbols
}

/**
 * Antenna Parameters (Ericsson specific)
 */
export interface AntennaUnit {
  id: string;
  mechanicalTilt: number; // 0-15 degrees
  electricalTilt: number; // 0-15 degrees
  azimuth: number; // 0-359 degrees
  totalTilt: number; // Calculated: mechanical + electrical
  maxTxPower: number; // -130 to 46 dBm (3GPP constraint)
  antennaGain: number; // dBi
  retSubunitRef?: string; // Reference to RET (Remote Electrical Tilt) unit
}

/**
 * SON (Self-Organizing Network) Parameters
 */
export interface SONParameters {
  mlbEnabled: boolean; // Mobility Load Balancing
  mroEnabled: boolean; // Mobility Robustness Optimization
  rachOptEnabled: boolean; // RACH Optimization
  icicEnabled: boolean; // Inter-Cell Interference Coordination

  // ANR (Automatic Neighbor Relations)
  anrEnabled: boolean;
  anrBlacklist?: string[];
  anrWhitelist?: string[];
}

// ============================================================================
// Zod Schema Validation (3GPP TS 32.616 Constraints)
// ============================================================================

/**
 * Base Managed Element Schema
 */
export const ManagedElementSchema = z.object({
  id: z.string().min(1).max(128),
  userLabel: z.string().min(1).max(256),
  swVersion: z.string().regex(/^\d+\.(Q\d+\.\d+|\d+\.\d+)/), // Ericsson version format (e.g., 21.Q4.1 or 1.2.3)
  vendorName: z.literal('Ericsson'),
  managedBy: z.string().optional(),
  locationName: z.string().max(256).optional(),
  dn: z.string().optional(), // LDAP DN format
});

/**
 * EUtranCellFDD Schema with 3GPP constraints
 */
export const EUtranCellFDDSchema = z.object({
  cellId: z.number().int().min(0).max(255),
  physicalLayerCellId: z.number().int().min(0).max(503),
  tac: z.number().int().min(0).max(65535),
  earfcn: z.number().int().min(0).max(262143), // 3GPP TS 36.101
  earfcnDl: z.number().int().min(0).max(262143).optional(),
  bandwidth: z.enum(['5', '10', '15', '20']).transform(Number),

  // Power Control (3GPP TS 28.552 constraints)
  p0NominalPUSCH: z.number().min(-130).max(-70),
  alpha: z.union([
    z.enum(['0', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']),
    z.number().refine(val => [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].includes(val))
  ]).transform(val => typeof val === 'string' ? Number(val) : val),

  // Optional parameters
  qRxLevMin: z.number().min(-70).max(-22).optional(),
  pMax: z.number().min(-30).max(33).optional(),
  cellRange: z.number().min(0).max(100000).optional(), // meters
  administrativeState: z.enum(['LOCKED', 'UNLOCKED', 'SHUTTING_DOWN']).optional(),
  operationalState: z.enum(['ENABLED', 'DISABLED']).optional(),
}).merge(ManagedElementSchema);

/**
 * NRCellDU Schema (5G NR)
 */
export const NRCellDUSchema = z.object({
  cellLocalId: z.number().int().min(0).max(16383),
  nCI: z.string().regex(/^[0-9A-F]{9}$/i), // 36-bit hex
  nRPCI: z.number().int().min(0).max(1007),
  nRTAC: z.number().int().min(0).max(16777215), // 24-bit
  arfcnDL: z.number().int().min(0).max(3279165), // 3GPP TS 38.104
  arfcnUL: z.number().int().min(0).max(3279165).optional(),
  bSChannelBwDL: z.number().int().positive(),
  bSChannelBwUL: z.number().int().positive().optional(),

  // 5G Power Control
  pZeroNomPusch: z.number().min(-202).max(24),
  pZeroNomPucch: z.number().min(-202).max(24),
  msg3DeltaPreamble: z.number().min(-1).max(6).optional(),

  // SSB Configuration
  ssbFrequency: z.number().int().positive().optional(),
  ssbPeriodicity: z.enum(['5', '10', '20', '40', '80', '160']).transform(Number).optional(),
  ssbDuration: z.enum(['1', '2', '3', '4', '5']).transform(Number).optional(),
}).merge(ManagedElementSchema);

/**
 * Antenna Unit Schema (Ericsson constraints)
 */
export const AntennaUnitSchema = z.object({
  id: z.string().min(1),
  mechanicalTilt: z.number().min(0).max(15), // Physical tilt constraint
  electricalTilt: z.number().min(0).max(15), // RET constraint
  azimuth: z.number().min(0).max(359),
  totalTilt: z.number().min(0).max(30), // Sum constraint
  maxTxPower: z.number().min(-130).max(46), // 3GPP TS 28.552
  antennaGain: z.number().min(-10).max(30), // dBi realistic range
  retSubunitRef: z.string().optional(),
});

/**
 * SON Parameters Schema
 */
export const SONParametersSchema = z.object({
  mlbEnabled: z.boolean(),
  mroEnabled: z.boolean(),
  rachOptEnabled: z.boolean(),
  icicEnabled: z.boolean(),
  anrEnabled: z.boolean(),
  anrBlacklist: z.array(z.string()).optional(),
  anrWhitelist: z.array(z.string()).optional(),
});

// ============================================================================
// MOM XML Parser (ENM CM Export Format)
// ============================================================================

/**
 * Parse MOM XML from ENM Scripting
 * Supports 3GPP TS 32.616 Bulk CM IRP XML format
 */
export class MOMXMLParser {
  private xmlHeader = '<?xml version="1.0" encoding="UTF-8"?>\n';

  /**
   * Parse EUtranCellFDD from XML
   */
  parseEUtranCellFDD(xml: string): EUtranCellFDD {
    // Simple XML parser (in production, use xml2js or fast-xml-parser)
    const extractValue = (tag: string): string => {
      const regex = new RegExp(`<${tag}>([^<]+)</${tag}>`, 'i');
      const match = xml.match(regex);
      return match ? match[1].trim() : '';
    };

    const extractNumber = (tag: string): number => {
      const value = extractValue(tag);
      return value ? parseFloat(value) : 0;
    };

    const cell = {
      id: extractValue('id') || extractValue('EUtranCellFDDId'),
      userLabel: extractValue('userLabel'),
      swVersion: extractValue('swVersion'),
      vendorName: 'Ericsson' as const,
      cellId: extractNumber('cellId'),
      physicalLayerCellId: extractNumber('physicalLayerCellId'),
      tac: extractNumber('tac'),
      earfcn: extractNumber('earfcn') || extractNumber('earfcnDl'),
      bandwidth: extractValue('dlChannelBandwidth') || extractValue('bandwidth'),
      p0NominalPUSCH: extractNumber('p0NominalPusch') || extractNumber('p0NominalPUSCH'),
      alpha: extractValue('alpha'),
      dn: extractValue('dn'),
    };

    // Validate against schema (this will coerce and validate types)
    return EUtranCellFDDSchema.parse(cell) as EUtranCellFDD;
  }

  /**
   * Parse NRCellDU from XML
   */
  parseNRCellDU(xml: string): NRCellDU {
    const extractValue = (tag: string): string => {
      const regex = new RegExp(`<${tag}>([^<]+)</${tag}>`, 'i');
      const match = xml.match(regex);
      return match ? match[1].trim() : '';
    };

    const extractNumber = (tag: string): number => {
      const value = extractValue(tag);
      return value ? parseFloat(value) : 0;
    };

    const cell = {
      id: extractValue('id') || extractValue('NRCellDUId'),
      userLabel: extractValue('userLabel'),
      swVersion: extractValue('swVersion'),
      vendorName: 'Ericsson' as const,
      cellLocalId: extractNumber('cellLocalId'),
      nCI: extractValue('nCI'),
      nRPCI: extractNumber('nRPCI'),
      nRTAC: extractNumber('nRTAC'),
      arfcnDL: extractNumber('arfcnDL'),
      bSChannelBwDL: extractNumber('bSChannelBwDL'),
      pZeroNomPusch: extractNumber('pZeroNomPusch'),
      pZeroNomPucch: extractNumber('pZeroNomPucch'),
      ssbPeriodicity: extractValue('ssbPeriodicity') || undefined,
    };

    return NRCellDUSchema.parse(cell) as NRCellDU;
  }

  /**
   * Parse Antenna Unit from XML
   */
  parseAntennaUnit(xml: string): AntennaUnit {
    const extractValue = (tag: string): string => {
      const regex = new RegExp(`<${tag}>([^<]+)</${tag}>`, 'i');
      const match = xml.match(regex);
      return match ? match[1].trim() : '';
    };

    const extractNumber = (tag: string): number => {
      const value = extractValue(tag);
      return value ? parseFloat(value) : 0;
    };

    const antenna = {
      id: extractValue('id') || extractValue('AntennaUnitId'),
      mechanicalTilt: extractNumber('mechanicalTilt'),
      electricalTilt: extractNumber('electricalTilt'),
      azimuth: extractNumber('azimuth'),
      totalTilt: extractNumber('totalTilt'),
      maxTxPower: extractNumber('maxTxPower'),
      antennaGain: extractNumber('antennaGain'),
      retSubunitRef: extractValue('retSubunitRef') || undefined,
    };

    return AntennaUnitSchema.parse(antenna) as AntennaUnit;
  }
}

// ============================================================================
// MOM XML Generator (ENM CM Import Format)
// ============================================================================

/**
 * Generate MOM XML for ENM CM operations
 * Conforms to 3GPP TS 32.616 Bulk CM IRP
 */
export class MOMXMLGenerator {
  private xmlHeader = '<?xml version="1.0" encoding="UTF-8"?>\n';

  /**
   * Generate XML for EUtranCellFDD parameter change
   */
  generateEUtranCellFDDXML(cell: Partial<EUtranCellFDD>, operation: 'create' | 'update' | 'delete' = 'update'): string {
    // Validate input
    const validated = EUtranCellFDDSchema.partial().parse(cell);

    const fdn = cell.dn || `MeContext=${cell.id},ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=${cell.id}`;

    let xml = this.xmlHeader;
    xml += '<bulkCmConfigDataFile xmlns="configData.xsd" xmlns:xn="genericNrm.xsd">\n';
    xml += '  <fileHeader fileFormatVersion="32.616 V8.0" vendorName="Ericsson"/>\n';
    xml += '  <configData>\n';
    xml += `    <managedObject class="EUtranCellFDD" operation="${operation}" version="1.0">\n`;
    xml += `      <dn>${fdn}</dn>\n`;

    // Add parameters
    Object.entries(validated).forEach(([key, value]) => {
      if (value !== undefined && key !== 'dn' && key !== 'id' && key !== 'vendorName') {
        xml += `      <p name="${key}">${value}</p>\n`;
      }
    });

    xml += '    </managedObject>\n';
    xml += '  </configData>\n';
    xml += '</bulkCmConfigDataFile>\n';

    return xml;
  }

  /**
   * Generate XML for NRCellDU parameter change
   */
  generateNRCellDUXML(cell: Partial<NRCellDU>, operation: 'create' | 'update' | 'delete' = 'update'): string {
    const validated = NRCellDUSchema.partial().parse(cell);

    const fdn = cell.dn || `MeContext=${cell.id},ManagedElement=1,GNBDUFunction=1,NRCellDU=${cell.id}`;

    let xml = this.xmlHeader;
    xml += '<bulkCmConfigDataFile xmlns="configData.xsd" xmlns:xn="nrNrm.xsd">\n';
    xml += '  <fileHeader fileFormatVersion="32.616 V16.0" vendorName="Ericsson"/>\n';
    xml += '  <configData>\n';
    xml += `    <managedObject class="NRCellDU" operation="${operation}" version="1.0">\n`;
    xml += `      <dn>${fdn}</dn>\n`;

    Object.entries(validated).forEach(([key, value]) => {
      if (value !== undefined && key !== 'dn' && key !== 'id' && key !== 'vendorName') {
        xml += `      <p name="${key}">${value}</p>\n`;
      }
    });

    xml += '    </managedObject>\n';
    xml += '  </configData>\n';
    xml += '</bulkCmConfigDataFile>\n';

    return xml;
  }

  /**
   * Generate XML for Antenna Unit parameter change
   */
  generateAntennaUnitXML(antenna: Partial<AntennaUnit>, operation: 'create' | 'update' | 'delete' = 'update'): string {
    const validated = AntennaUnitSchema.partial().parse(antenna);

    const fdn = `MeContext=${antenna.id},ManagedElement=1,Equipment=1,AntennaUnit=${antenna.id}`;

    let xml = this.xmlHeader;
    xml += '<bulkCmConfigDataFile xmlns="configData.xsd">\n';
    xml += '  <fileHeader fileFormatVersion="32.616 V8.0" vendorName="Ericsson"/>\n';
    xml += '  <configData>\n';
    xml += `    <managedObject class="AntennaUnit" operation="${operation}" version="1.0">\n`;
    xml += `      <dn>${fdn}</dn>\n`;

    Object.entries(validated).forEach(([key, value]) => {
      if (value !== undefined && key !== 'id') {
        xml += `      <p name="${key}">${value}</p>\n`;
      }
    });

    xml += '    </managedObject>\n';
    xml += '  </configData>\n';
    xml += '</bulkCmConfigDataFile>\n';

    return xml;
  }

  /**
   * Generate bulk XML for multiple managed objects
   */
  generateBulkXML(objects: Array<{
    type: 'EUtranCellFDD' | 'NRCellDU' | 'AntennaUnit';
    data: any;
    operation?: 'create' | 'update' | 'delete';
  }>): string {
    let xml = this.xmlHeader;
    xml += '<bulkCmConfigDataFile xmlns="configData.xsd">\n';
    xml += '  <fileHeader fileFormatVersion="32.616 V8.0" vendorName="Ericsson"/>\n';
    xml += '  <configData>\n';

    objects.forEach((obj) => {
      const operation = obj.operation || 'update';
      let fdn = '';
      let validated: any;

      switch (obj.type) {
        case 'EUtranCellFDD':
          validated = EUtranCellFDDSchema.partial().parse(obj.data);
          fdn = obj.data.dn || `MeContext=${obj.data.id},ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=${obj.data.id}`;
          break;
        case 'NRCellDU':
          validated = NRCellDUSchema.partial().parse(obj.data);
          fdn = obj.data.dn || `MeContext=${obj.data.id},ManagedElement=1,GNBDUFunction=1,NRCellDU=${obj.data.id}`;
          break;
        case 'AntennaUnit':
          validated = AntennaUnitSchema.partial().parse(obj.data);
          fdn = `MeContext=${obj.data.id},ManagedElement=1,Equipment=1,AntennaUnit=${obj.data.id}`;
          break;
      }

      xml += `    <managedObject class="${obj.type}" operation="${operation}" version="1.0">\n`;
      xml += `      <dn>${fdn}</dn>\n`;

      Object.entries(validated).forEach(([key, value]) => {
        if (value !== undefined && key !== 'dn' && key !== 'id' && key !== 'vendorName') {
          xml += `      <p name="${key}">${value}</p>\n`;
        }
      });

      xml += '    </managedObject>\n';
    });

    xml += '  </configData>\n';
    xml += '</bulkCmConfigDataFile>\n';

    return xml;
  }
}

// ============================================================================
// Export Instances
// ============================================================================

export const momParser = new MOMXMLParser();
export const momGenerator = new MOMXMLGenerator();

// ============================================================================
// Type Guards
// ============================================================================

export function isEUtranCellFDD(obj: any): obj is EUtranCellFDD {
  return EUtranCellFDDSchema.safeParse(obj).success;
}

export function isNRCellDU(obj: any): obj is NRCellDU {
  return NRCellDUSchema.safeParse(obj).success;
}

export function isAntennaUnit(obj: any): obj is AntennaUnit {
  return AntennaUnitSchema.safeParse(obj).success;
}
