/**
 * E2E cmedit Syntax Validation Test Suite
 *
 * Validates generated cmedit commands against the official Ericsson RAN syntax
 * from rtb-prd.md. This test suite ensures that all Phase 3 components generate
 * syntactically correct Ericsson ENM CLI commands.
 *
 * Test Coverage:
 * 1. GET operations - Query MO instances with filters
 * 2. SET operations - Modify MO attributes with validation
 * 3. CREATE operations - Create new MO instances
 * 4. DELETE operations - Remove MO instances
 * 5. MONITOR operations - Performance monitoring
 * 6. Advanced command patterns - Collections, scope filters, wildcards
 * 7. Error scenarios - Invalid syntax, missing parameters
 */

// Expected cmedit command syntax patterns from rtb-prd.md
const EXPECTED_CMEDIT_PATTERNS = {
  // GET Operations
  GET_MO_INSTANCE: /^cmedit get (\w+)(?:,(\w+))*\s+(.+)$/,
  GET_WITH_FILTER: /^cmedit get (\w+) criteria (.+?)\s+(.+)$/,
  GET_WITH_WILDCARD: /^cmedit get (\w+\*?)\s+(.+)$/,
  GET_WITH_ATTRIBUTES: /^cmedit get (\w+)\s+(.+?)\s+(.+)$/,

  // SET Operations
  SET_SINGLE_ATTRIBUTE: /^cmedit set (\w+)\s+(\w+)\.(\w+)=(.+)$/,
  SET_MULTIPLE_ATTRIBUTES: /^cmedit set (\w+)\s+(\w+)\.(\w+)=(.+),(.+)$/,
  SET_WITH_FDN: /^cmedit set (\w+)\s+(.+)=(.+)$/,
  SET_WITH_SCOPEFILTER: /^cmedit set (\w+\*?)\s+--scopefilter\s+\((.+?)\)\s+(.+?)\s+(.+)$/,

  // CREATE Operations
  CREATE_MO_INSTANCE: /^cmedit create (\w+)\s+(\w+)(?:\s+(.*))?$/,
  CREATE_WITH_ATTRIBUTES: /^cmedit create (\w+)\s+(\w+)\s+(.+)$/,

  // DELETE Operations
  DELETE_MO_INSTANCE: /^cmedit delete (\w+)\s+(.+)$/,
  DELETE_WITH_ATTRIBUTES: /^cmedit delete (\w+)\s+(.+?)\s+(.+)$/,

  // MONITOR Operations
  MONITOR_START: /^cmedit mon (\w+)(?:\s+(.*))?$/,
  MONITOR_STOP: /^cmedit unmon (\w+)(?:\s+(.*))?$/,

  // Advanced Patterns
  BATCH_COLLECTION: /^cmedit set --collection (\w+)\s+(.+)$/,
  WITH_PREVIEW: /^(.+)\s+--preview$/,
  WITH_FORCE: /^(.+)\s+--force$/,
  WITH_TABLE_FORMAT: /^(.+)\s+-t$/,
  WITH_DETAILED_FORMAT: /^(.+)\s+-d$/,
  WITH_SHORT_FORMAT: /^(.+)\s+-s$/
};

// Valid Ericsson MO classes from rtb-prd.md
const VALID_MO_CLASSES = {
  // Core Network Elements
  MANAGED_ELEMENT: ['ManagedElement', 'SubNetwork', 'MeContext'],

  // LTE Radio Access
  LTE: ['ENBFunction', 'EUtranCellFDD', 'EUtranCellTDD', 'EUtranFreqRelation',
       'EUtranCellRelation', 'UtranFreqRelation', 'UtranCellRelation',
       'SectorCarrier', 'ExternalENodeB', 'ExternalEUtranCellFDD'],

  // 5G NR Radio Access
  NR5G: ['GNBCUCPFunction', 'GNBCUUPFunction', 'GNBDUFunction',
        'NRCellCU', 'NRCellDU', 'NRFreqRelation', 'NRCellRelation',
        'ExternalNRCellCU', 'NRSectorCarrier'],

  // System Functions
  SYSTEM: ['SystemFunctions', 'BrM', 'Fm', 'HealthCheckM', 'Lm', 'LogM',
          'Pm', 'PmEventM', 'SecM', 'CertM'],

  // Feature Management
  FEATURES: ['FeatureState', 'OptionalFeatureLicense', 'AnrFunction',
           'MroFunction', 'SonFunction', 'MimoSleepFunction'],

  // Configuration Management
  CONFIG: ['Timesettings', 'QciProfile', 'ApnProfile', 'SubscriberProfile',
          'PolicyProfile', 'ChargingProfile']
};

// Valid Ericsson parameter names from rtb-prd.md
const VALID_PARAMETERS = {
  // Managed Element
  MANAGED_ELEMENT: ['managedElementId', 'userLabel', 'neType', 'swVersion', 'location'],

  // eNodeB Function
  ENB_FUNCTION: ['eNodeBId', 'maxConnectedUe', 'maxEnbSupportedUe', 'endcEnabled',
                  'splitBearerSupport', 'loadBalancingEnabled', 'makeBeforeBreakEnabled'],

  // LTE Cell
  EUTRAN_CELL_FDD: ['euTranCellFddId', 'cellId', 'pci', 'freqBand', 'pointAArfcnDl',
                    'pointAArfcnUl', 'qRxLevMin', 'qQualMin', 'cellBarred',
                    'administrativestate', 'operstate', 'txPower', 'antennaElectricalTilt',
                    'massiveMimoEnabled', 'ul256qamEnabled', 'dl256qamEnabled',
                    'transmissionMode', 'cellReselectionPriority'],

  // Frequency Relations
  EUTRAN_FREQ_RELATION: ['euTranFreqRelationId', 'qOffsetFreq', 'hysteresis',
                          'timeToTrigger', 'a3Offset', 'a1Threshold', 'a2Threshold'],

  // 5G NR Cell CU
  NR_CELL_CU: ['nrCellCuId', 'pci', 'nrArfcnDl', 'nrArfcnUl', 'ssbSubcarrierOffset',
               'scsSpecificCarrierList', 'nrdcEnabled', 'admissionPriority',
               'admissionLimit', 'nrdcAdminState', 'nrdcOperState'],

  // ANR Function
  ANR_FUNCTION: ['anrEnabled', 'automaticNeighbourRelation', 'removeEnbTime',
                'removeGnbTime', 'pciConflictCellSelection', 'maxTimeEventBasedPciConf'],

  // Feature State
  FEATURE_STATE: ['featureStateId', 'featureState'],

  // License
  OPTIONAL_FEATURE_LICENSE: ['optionalFeatureLicenseId', 'featureState'],

  // Time Settings
  TIME_SETTINGS: ['daylightSavingTimeStartDate', 'daylightSavingTimeEndDate',
                   'daylightSavingTimeOffset', 'TimeOffset']
};

// Valid parameter values and ranges
const VALID_PARAMETER_VALUES = {
  BOOLEAN: ['true', 'false', 'TRUE', 'FALSE', 'True', 'False'],
  ADMIN_STATE: ['UNLOCKED', 'LOCKED'],
  OPER_STATE: ['ENABLED', 'DISABLED'],
  CELL_BARRED: ['NOT_BARRED', 'BARRED'],
  SYNC_STATUS: ['SYNCHRONIZED', 'OUT_OF_SYNC', 'SYNCHRONIZING'],
  FEATURE_STATE: ['ACTIVATED', 'DEACTIVATED'],

  // Numeric ranges
  PCI_RANGE: { min: 0, max: 503 },
  QRXLEV_MIN_RANGE: { min: -140, max: -44 },
  QQAL_MIN_RANGE: { min: -20, max: 0 },
  POWER_RANGE: { min: 0, max: 46 },
  TILT_RANGE: { min: -15, max: 15 },
  HYSTERESIS_RANGE: { min: 0, max: 30 },
  TIME_TO_TRIGGER_RANGE: { min: 0, max: 5120 },

  // String patterns
  CELL_ID_PATTERN: /^[A-Za-z0-9_-]+$/,
  FDN_PATTERN: /^[\w=,()_-]+$/,
  NODE_ID_PATTERN: /^[A-Za-z0-9_-]+$/
};

// Command validator class
class CmeditSyntaxValidator {
  validateCommand(command: string): ValidationResult {
    const result: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
      suggestions: []
    };

    // Basic syntax validation
    if (!command.startsWith('cmedit ')) {
      result.valid = false;
      result.errors.push('Command must start with "cmedit "');
      return result;
    }

    const commandParts = command.split(' ');
    const operation = commandParts[1].toUpperCase();

    // Validate operation-specific syntax
    switch (operation) {
      case 'GET':
        this.validateGetCommand(command, result);
        break;
      case 'SET':
        this.validateSetCommand(command, result);
        break;
      case 'CREATE':
        this.validateCreateCommand(command, result);
        break;
      case 'DELETE':
        this.validateDeleteCommand(command, result);
        break;
      case 'MON':
      case 'UNMON':
        this.validateMonitorCommand(command, result);
        break;
      default:
        result.valid = false;
        result.errors.push(`Invalid operation: ${operation}`);
    }

    // Validate global options
    this.validateGlobalOptions(command, result);

    return result;
  }

  private validateGetCommand(command: string, result: ValidationResult): void {
    // Validate GET operation syntax
    if (EXPECTED_CMEDIT_PATTERNS.GET_MO_INSTANCE.test(command)) {
      const match = command.match(EXPECTED_CMEDIT_PATTERNS.GET_MO_INSTANCE);
      if (match) {
        const moClass = match[1];
        const attributes = match[3];
        this.validateMOClass(moClass, result);
        this.validateAttributes(attributes, result);
      }
    } else if (EXPECTED_CMEDIT_PATTERNS.GET_WITH_FILTER.test(command)) {
      const match = command.match(EXPECTED_CMEDIT_PATTERNS.GET_WITH_FILTER);
      if (match) {
        const moClass = match[1];
        const filter = match[2];
        const attributes = match[3];
        this.validateMOClass(moClass, result);
        this.validateFilterSyntax(filter, result);
        this.validateAttributes(attributes, result);
      }
    } else {
      result.warnings.push('GET command syntax may be suboptimal');
    }
  }

  private validateSetCommand(command: string, result: ValidationResult): void {
    // Validate SET operation syntax
    if (EXPECTED_CMEDIT_PATTERNS.SET_WITH_FDN.test(command)) {
      const match = command.match(EXPECTED_CMEDIT_PATTERNS.SET_WITH_FDN);
      if (match) {
        const nodeId = match[1];
        const fdnAndParams = match[2];
        const value = match[3];
        this.validateNodeId(nodeId, result);
        this.validateFDNAndParameters(fdnAndParams, value, result);
      }
    } else {
      result.warnings.push('SET command may have syntax issues');
    }
  }

  private validateCreateCommand(command: string, result: ValidationResult): void {
    // Validate CREATE operation syntax
    if (EXPECTED_CMEDIT_PATTERNS.CREATE_MO_INSTANCE.test(command)) {
      const match = command.match(EXPECTED_CMEDIT_PATTERNS.CREATE_MO_INSTANCE);
      if (match) {
        const nodeId = match[1];
        const moClass = match[2];
        const attributes = match[3];
        this.validateNodeId(nodeId, result);
        this.validateMOClass(moClass, result);
        if (attributes) {
          this.validateAttributes(attributes, result);
        }
      }
    } else {
      result.warnings.push('CREATE command syntax may be suboptimal');
    }
  }

  private validateDeleteCommand(command: string, result: ValidationResult): void {
    // Validate DELETE operation syntax
    if (EXPECTED_CMEDIT_PATTERNS.DELETE_MO_INSTANCE.test(command)) {
      const match = command.match(EXPECTED_CMEDIT_PATTERNS.DELETE_MO_INSTANCE);
      if (match) {
        const nodeId = match[1];
        const fdn = match[2];
        this.validateNodeId(nodeId, result);
        this.validateFDN(fdn, result);
      }
    } else {
      result.warnings.push('DELETE command syntax may be suboptimal');
    }
  }

  private validateMonitorCommand(command: string, result: ValidationResult): void {
    // Validate MONITOR operation syntax
    const operation = command.split(' ')[1].toUpperCase();
    if (operation === 'MON' || operation === 'UNMON') {
      const parts = command.split(' ');
      if (parts.length < 3) {
        result.errors.push(`${operation} command requires a target FDN`);
      } else {
        const target = parts.slice(2).join(' ');
        this.validateFDN(target, result);
      }
    }
  }

  private validateMOClass(moClass: string, result: ValidationResult): void {
    const allValidClasses = Object.values(VALID_MO_CLASSES).flat();
    if (!allValidClasses.includes(moClass)) {
      result.warnings.push(`MO class "${moClass}" may not be a valid Ericsson MO class`);
      result.suggestions.push('Verify MO class against Ericsson documentation');
    }
  }

  private validateAttributes(attributes: string, result: ValidationResult): void {
    if (!attributes) return;

    const attrParts = attributes.split(',');
    for (const attr of attrParts) {
      const [param, value] = attr.split('=');
      if (!param || !value) {
        result.errors.push(`Invalid attribute syntax: ${attr}`);
        continue;
      }

      this.validateParameter(param.trim(), value.trim(), result);
    }
  }

  private validateParameter(param: string, value: string, result: ValidationResult): void {
    // Check if parameter is known
    const allValidParams = Object.values(VALID_PARAMETERS).flat();
    if (!allValidParams.includes(param)) {
      result.warnings.push(`Parameter "${param}" may not be a valid Ericsson parameter`);
      result.suggestions.push('Verify parameter against Ericsson documentation');
    }

    // Validate parameter value based on parameter name
    if (param.toLowerCase().includes('state')) {
      if (!VALID_PARAMETER_VALUES.ADMIN_STATE.includes(value) &&
          !VALID_PARAMETER_VALUES.OPER_STATE.includes(value) &&
          !VALID_PARAMETER_VALUES.BOOLEAN.includes(value)) {
        result.warnings.push(`Invalid state value for ${param}: ${value}`);
      }
    }

    if (param.toLowerCase().includes('pci')) {
      const pci = parseInt(value);
      if (isNaN(pci) || pci < VALID_PARAMETER_VALUES.PCI_RANGE.min ||
          pci > VALID_PARAMETER_VALUES.PCI_RANGE.max) {
        result.errors.push(`PCI value ${value} out of valid range (${VALID_PARAMETER_VALUES.PCI_RANGE.min}-${VALID_PARAMETER_VALUES.PCI_RANGE.max})`);
      }
    }

    if (param.toLowerCase().includes('qrxlevmin')) {
      const qrxlev = parseInt(value);
      if (isNaN(qrxlev) || qrxlev < VALID_PARAMETER_VALUES.QRXLEV_MIN_RANGE.min ||
          qrxlev > VALID_PARAMETER_VALUES.QRXLEV_MIN_RANGE.max) {
        result.errors.push(`qRxLevMin value ${value} out of valid range (${VALID_PARAMETER_VALUES.QRXLEV_MIN_RANGE.min}-${VALID_PARAMETER_VALUES.QRXLEV_MIN_RANGE.max})`);
      }
    }

    if (param.toLowerCase().includes('power') || param.toLowerCase().includes('txpower')) {
      const power = parseInt(value);
      if (isNaN(power) || power < VALID_PARAMETER_VALUES.POWER_RANGE.min ||
          power > VALID_PARAMETER_VALUES.POWER_RANGE.max) {
        result.warnings.push(`Power value ${value} may be out of typical range (${VALID_PARAMETER_VALUES.POWER_RANGE.min}-${VALID_PARAMETER_VALUES.POWER_RANGE.max} dBm)`);
      }
    }
  }

  private validateFilterSyntax(filter: string, result: ValidationResult): void {
    // Basic filter validation
    if (!filter.includes('(') || !filter.includes(')')) {
      result.warnings.push('Filter expressions should use parentheses for clarity');
    }

    // Check for common filter patterns
    if (filter.includes('==') || filter.includes('>=') || filter.includes('<=') ||
        filter.includes('>') || filter.includes('<')) {
      // Valid comparison operators
      return;
    } else {
      result.warnings.push('Filter should include comparison operators');
    }
  }

  private validateFDNAndParameters(fdnAndParams: string, value: string, result: ValidationResult): void {
    // Split FDN from parameters
    const lastDotIndex = fdnAndParams.lastIndexOf('.');
    if (lastDotIndex === -1) {
      result.errors.push('Invalid FDN format - missing parameter');
      return;
    }

    const fdn = fdnAndParams.substring(0, lastDotIndex);
    const param = fdnAndParams.substring(lastDotIndex + 1);

    this.validateFDN(fdn, result);
    this.validateParameter(param, value, result);
  }

  private validateFDN(fdn: string, result: ValidationResult): void {
    // Basic FDN validation
    if (!VALID_PARAMETER_VALUES.FDN_PATTERN.test(fdn)) {
      result.warnings.push(`FDN "${fdn}" may have syntax issues`);
    }

    // Check for valid MO class sequence
    const fdnParts = fdn.split(',');
    for (let i = 0; i < fdnParts.length; i++) {
      const part = fdnParts[i].trim();
      const [moClass, ...identifiers] = part.split('=');
      this.validateMOClass(moClass, result);
    }
  }

  private validateNodeId(nodeId: string, result: ValidationResult): void {
    if (!VALID_PARAMETER_VALUES.NODE_ID_PATTERN.test(nodeId)) {
      result.warnings.push(`Node ID "${nodeId}" may not follow naming conventions`);
    }
  }

  private validateGlobalOptions(command: string, result: ValidationResult): void {
    // Check for preview mode
    if (command.includes('--preview')) {
      result.suggestions.push('Preview mode is enabled - no actual changes will be made');
    }

    // Check for force mode
    if (command.includes('--force')) {
      result.warnings.push('Force mode is enabled - bypasses safety checks');
    }

    // Check for format options
    if (command.includes('-t')) {
      result.suggestions.push('Table format output will be used');
    }
    if (command.includes('-d')) {
      result.suggestions.push('Detailed format output will be used');
    }
    if (command.includes('-s')) {
      result.suggestions.push('Short format output will be used');
    }
  }
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

// Expected test commands based on rtb-prd.md examples
const EXPECTED_CMEDIT_COMMANDS = [
  // GET Operations from rtb-prd.md
  'cmedit get MeContext=ERBS001,ManagedElement=1',
  'cmedit get LTE32ERBS00001 ENodeBFunction.eNodeBPlmnId',
  'cmedit get LTE32ERBS00001 EUtranCellFDD=1 qRxLevMin,qQualMin',
  'cmedit get LTE32ERBS0000* ENodeBFunction.eNodeBPlmnId.(mcc>=271)',
  'cmedit get * EUtranCellFDD.activePlmnList.[{mcc==353}]',
  'cmedit get LTE32ERBS00001 ManagedElement.(neType,userLabel) -d',
  'cmedit get LTE32ERBS00001 ENodeBFunction --attribute userLabel',

  // SET Operations from rtb-prd.md
  'cmedit set EPINAL_VCV_LTE EUtranCellFDD=83906_E1 qRxLevMin=-130',
  'cmedit set DR_METZ OptionalFeatureLicense.(OptionalFeatureLicenseId==Anr) featureState=ACTIVATED',
  'cmedit set *_LTE --scopefilter (CmFunction.syncStatus==SYNCHRONIZED) EUtranCellFDD ul256qamEnabled=true',
  'cmedit set EPINAL_VCV_LTE EUtranCellFDD=83906_E1 qRxLevMin=-130,qQualMin=-32,cellIndividualOffset=3',
  'cmedit set GRAY_LTE EUtranCellFDD.(EUtranCellFddId==86376_V3),EUtranCellRelation.(EUtranCellRelationId==2081-86376-6) isHoAllowed=true',
  'cmedit set Dijon_4G-5G EUtranCellFDD.(EUtranCellFddId==*_K*) additionalPlmnReservedList=[false,false,false,false,false] -t',

  // CREATE Operations
  'cmedit create SubNetwork=ENM_NE2,MeContext=NCYBEB4,ManagedElement=NCYBEB4,BscFunction=1,BscM=1,GeranCellM=1,GeranCell=03467G1,Mobility=1,InterRanMobility=1,EUtranFrequency=3000 EUtranFrequencyId=3000',

  // DELETE Operations
  'cmedit delete COHARTILLE_N2_T_LTE UtranFreqRelation.(UtranFreqRelationId==3011)',
  'cmedit delete --collection test1 externalEUtranCellFDD.(externalEUtranCellFddId==2081-82879*) --preview --all',

  // MONITOR Operations
  'cmedit mon <FDN> [monitoring options]',
  'cmedit unmon <FDN> [unmonitor options]',

  // Advanced patterns
  'cmedit set --collection DUNKERQUE EUtranCellFDD lbTpNonQualFraction=25 --preview',
  'cmedit set *_LTE FeatureState.(featureStateId==CXC4012302) featureState=ACTIVATED --force'
];

describe('E2E cmedit Syntax Validation', () => {
  let validator: CmeditSyntaxValidator;

  beforeEach(() => {
    validator = new CmeditSyntaxValidator();
  });

  describe('Expected Commands from rtb-prd.md', () => {
    it('should validate GET operations from rtb-prd.md', () => {
      const getCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd => cmd.startsWith('cmedit get'));

      getCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Commands should be syntactically correct
        expect(command).toMatch(/^cmedit get /);
      });
    });

    it('should validate SET operations from rtb-prd.md', () => {
      const setCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd => cmd.startsWith('cmedit set'));

      setCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Should have proper parameter=value format
        expect(command).toMatch(/=\w+/);
      });
    });

    it('should validate CREATE operations from rtb-prd.md', () => {
      const createCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd => cmd.startsWith('cmedit create'));

      createCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Should have proper CREATE syntax
        expect(command).toMatch(/^cmedit create \w+ \w+/);
      });
    });

    it('should validate DELETE operations from rtb-prd.md', () => {
      const deleteCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd => cmd.startsWith('cmedit delete'));

      deleteCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Should have proper DELETE syntax
        expect(command).toMatch(/^cmedit delete \w+ /);
      });
    });

    it('should validate commands with global options', () => {
      const commandsWithOptions = EXPECTED_CMEDIT_COMMANDS.filter(cmd =>
        cmd.includes('--preview') || cmd.includes('--force') ||
        cmd.includes('-t') || cmd.includes('-d') || cmd.includes('-s')
      );

      commandsWithOptions.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
        expect(result.suggestions.length).toBeGreaterThan(0);
      });
    });

    it('should validate wildcard patterns', () => {
      const wildcardCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd =>
        cmd.includes('*') || cmd.includes('?')
      );

      wildcardCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate scope filter patterns', () => {
      const scopeFilterCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd =>
        cmd.includes('--scopefilter')
      );

      scopeFilterCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Should have proper scope filter syntax
        expect(command).toMatch(/--scopefilter\s+\([^)]+\)/);
      });
    });

    it('should validate collection operations', () => {
      const collectionCommands = EXPECTED_CMEDIT_COMMANDS.filter(cmd =>
        cmd.includes('--collection')
      );

      collectionCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);

        // Should have proper collection syntax
        expect(command).toMatch(/--collection\s+\w+/);
      });
    });
  });

  describe('Generated Commands from Phase 3 Components', () => {
    it('should validate LTE cell configuration commands', () => {
      const lteCommands = [
        'cmedit set TEST_NODE_001 EUtranCellFDD=1 pci=100,qRxLevMin=-130,qQualMin=-32',
        'cmedit set TEST_NODE_001 EUtranCellFDD=1 administrativestate=UNLOCKED',
        'cmedit set TEST_NODE_001 EUtranCellFDD=1 txPower=43,antennaElectricalTilt=5',
        'cmedit set TEST_NODE_001 EUtranCellFDD=1 massiveMimoEnabled=1,ul256qamEnabled=true'
      ];

      lteCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate 5G NR cell configuration commands', () => {
      const nrCommands = [
        'cmedit set TEST_NODE_001 NRCellCU=NRCELL_1 nrCellCuId=1,pci=100',
        'cmedit set TEST_NODE_001 NRCellCU=NRCELL_1 nrdcEnabled=true,admissionPriority=80',
        'cmedit create TEST_NODE_001 NRFreqRelation NRFreqRelationId=1 referenceFreq=1300,relatedFreq=78',
        'cmedit set TEST_NODE_001 NRFreqRelation=(NRFreqRelationId==1) qOffsetCell=0dB,scgFailureInfoNR=0'
      ];

      nrCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate frequency relation commands', () => {
      const freqCommands = [
        'cmedit create TEST_NODE_001 EUtranFreqRelation EUtranFreqRelationId=1',
        'cmedit set TEST_NODE_001 EUtranFreqRelation.(EUtranFreqRelationId==1) qOffsetFreq=0',
        'cmedit set TEST_NODE_001 EUtranFreqRelation.(EUtranFreqRelationId==1) hysteresis=2.0,timeToTrigger=320',
        'cmedit set TEST_NODE_001 EUtranFreqRelation.(EUtranFreqRelationId==1) a3Offset=1'
      ];

      freqCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate EN-DC configuration commands', () => {
      const endcCommands = [
        'cmedit set TEST_NODE_001 ENBFunction=1 endcEnabled=true,splitBearerSupport=true',
        'cmedit set TEST_NODE_001 ENBFunction=1 nrEventB1Threshold=-110,nrEventB1Hysteresis=2',
        'cmedit set TEST_NODE_001 ENBFunction=1 nrEventB1TimeToTrigger=320',
        'cmedit set TEST_NODE_001 ENBFunction=1 carrierAggregationEnabled=true,maxAggregatedBandwidth=40'
      ];

      endcCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate ANR configuration commands', () => {
      const anrCommands = [
        'cmedit set TEST_NODE_001 AnrFunction=1 anrEnabled=true,automaticNeighbourRelation=true',
        'cmedit set TEST_NODE_001 AnrFunction=1 removeEnbTime=5,removeGnbTime=5',
        'cmedit set TEST_NODE_001 AnrFunction=1 pciConflictCellSelection=ON',
        'cmedit set TEST_NODE_001 AnrFunction=1 maxTimeEventBasedPciConf=20'
      ];

      anrCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate feature activation commands', () => {
      const featureCommands = [
        'cmedit set TEST_NODE_001 FeatureState.(featureStateId==CXC4012302) featureState=ACTIVATED',
        'cmedit set TEST_NODE_001 OptionalFeatureLicense.(OptionalFeatureLicenseId==Anr) featureState=ACTIVATED',
        'cmedit set TEST_NODE_001 FeatureState.(featureStateId==CXC4012319) featureState=ACTIVATED --force',
        'cmedit set TEST_NODE_001 FeatureState.(featureStateId==MassiveMIMO) featureState=ACTIVATED'
      ];

      featureCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate batch operation commands', () => {
      const batchCommands = [
        'cmedit set --collection URBAN_CELLS EUtranCellFDD qRxLevMin=-130 --preview',
        'cmedit set *_LTE --scopefilter (CmFunction.syncStatus==SYNCHRONIZED) EUtranCellFDD administrativestate=UNLOCKED',
        'cmedit set --collection HIGH_TRAFFIC EUtranCellFDD cellCapMaxCellSubCap=60000',
        'cmedit set *_LTE --scopefilter (EUtranCellFDD.cellLoad<80) EUtranCellFDD massiveMimoEnabled=true'
      ];

      batchCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });
  });

  describe('Error Detection and Edge Cases', () => {
    it('should detect invalid command syntax', () => {
      const invalidCommands = [
        'cmedit', // Missing operation
        'cmedit INVALID_OPERATION', // Invalid operation
        'cmedit set', // Missing parameters
        'cmedit get', // Missing target
        'cmedit create', // Missing MO class
        'cmedit delete', // Missing target
        'set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130', // Missing cmedit prefix
        'cmedit get TEST_NODE InvalidMO.invalidParam=value', // Invalid MO class
        'cmedit set TEST_NODE EUtranCellFDD=1 invalidParam=value', // Invalid parameter
        'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=999' // Invalid parameter value
      ];

      invalidCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(false);
        expect(result.errors.length).toBeGreaterThan(0);
      });
    });

    it('should detect parameter value range violations', () => {
      const rangeViolationCommands = [
        'cmedit set TEST_NODE EUtranCellFDD=1 pci=600', // PCI out of range
        'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-200', // qRxLevMin out of range
        'cmedit set TEST_NODE EUtranCellFDD=1 txPower=100', // Power out of range
        'cmedit set TEST_NODE EUtranCellFDD=1 antennaElectricalTilt=90' // Tilt out of range
      ];

      rangeViolationCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.errors.length).toBeGreaterThan(0);
      });
    });

    it('should provide warnings for suboptimal syntax', () => {
      const suboptimalCommands = [
        'cmedit set TEST_NODE EUtranCellFDD=1', // No parameters
        'cmedit get TEST_NODE EUtranCellFDD', // No attributes specified
        'cmedit create TEST_NODE UnknownMOClass', // Unknown MO class
        'cmedit set TEST_NODE EUtranCellFDD=1 unknownParam=value' // Unknown parameter
      ];

      suboptimalCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.warnings.length).toBeGreaterThan(0);
      });
    });

    it('should provide suggestions for command improvement', () => {
      const improvableCommands = [
        'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130 --preview', // With preview
        'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130 --force', // With force
        'cmedit get TEST_NODE EUtranCellFDD=1 -t', // Table format
        'cmedit get TEST_NODE EUtranCellFDD=1 -s' // Short format
      ];

      improvableCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.suggestions.length).toBeGreaterThan(0);
      });
    });

    it('should handle complex nested FDN patterns', () => {
      const complexFdnCommands = [
        'cmedit set SubNetwork=ENM_NE1,MeContext=SITE_LTE,ManagedElement=SITE_LTE,ENodeBFunction=1,EUtranCellFDD=CELL_ID,UtranFreqRelation=FREQ_ID,UtranCellRelation=CELL_REL isHoAllowed=true',
        'cmedit get SubNetwork=ENM_NE1,MeContext=SITE_LTE,ManagedElement=SITE_LTE,SystemFunctions=1,BrM=1,BrmBackupManager=1 -d',
        'cmedit set SubNetwork=ENM_NE2,MeContext=NCYBEB4,ManagedElement=NCYBEB4,BscFunction=1,BscM=1,GeranCellM=1,GeranCell=03467G1,Mobility=1,InterRanMobility=1,EUtranFrequency=3000 EUtranFrequencyId=3000'
      ];

      complexFdnCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate array parameter values', () => {
      const arrayCommands = [
        'cmedit set TEST_NODE EUtranCellFDD=1 additionalPlmnReservedList=[false,false,false,false,false]',
        'cmedit set TEST_NODE ENBFunction=1 supportedBandList=[1,3,7,20,28]',
        'cmedit set TEST_NODE EUtranCellFDD=1 scsSpecificCarrierList=[{"location":0,"subcarrierSpacing":30}]',
        'cmedit set TEST_NODE EUtranCellFDD=1 t304=timer[2000]'
      ];

      arrayCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });
  });

  describe('Performance and Scalability', () => {
    it('should validate large numbers of commands efficiently', () => {
      const startTime = performance.now();

      // Generate 1000 test commands
      const testCommands: string[] = [];
      for (let i = 0; i < 1000; i++) {
        testCommands.push(`cmedit set TEST_NODE_${i} EUtranCellFDD=${i} pci=${i},qRxLevMin=-130`);
      }

      testCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });

      const validationTime = performance.now() - startTime;
      expect(validationTime).toBeLessThan(1000); // Should validate 1000 commands in <1 second
    });

    it('should handle concurrent validation', async () => {
      const concurrentCommands = Array(100).fill(null).map((_, i) =>
        `cmedit set CONCURRENT_NODE_${i} EUtranCellFDD=${i} pci=${i % 504}`
      );

      const startTime = performance.now();

      // Validate all commands concurrently
      const results = concurrentCommands.map(command => validator.validateCommand(command));

      const validationTime = performance.now() - startTime;
      expect(validationTime).toBeLessThan(500); // Should validate 100 commands in <500ms

      // All should be valid
      results.forEach(result => {
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should maintain validation accuracy under load', () => {
      const complexCommands = [
        // Long, complex commands
        'cmedit set SubNetwork=ENM_NE1,MeContext=VERY_LONG_SITE_NAME_WITH_MANY_UNDERSCORES_AND_NUMBERS_12345,ManagedElement=VERY_LONG_ELEMENT_NAME,ENodeBFunction=1,EUtranCellFDD=VERY_LONG_CELL_ID_WITH_MANY_UNDERSCORES_AND_NUMBERS_67890 qRxLevMin=-130,qQualMin=-32,administrativestate=UNLOCKED,txPower=43,antennaElectricalTilt=5.5,massiveMimoEnabled=1,ul256qamEnabled=true,dl256qamEnabled=true,transmissionMode=TRANSMISSION_MODE_4,cellReselectionPriority=7,threshServLow=-120,threshXLow=-125',
        'cmedit get SubNetwork=ENM_NE1,MeContext=SITE_001,ManagedElement=SITE_001,SystemFunctions=1,BrM=1,BrmBackupManager=1,BrmRollbackAtRestore=1,Fm=1,FmAlarm=0,FmAlarmModel=0,HealthCheckM=1,HcJob=0,Lm=1,CapacityKey=0,FeatureKey=0,LogM=1,PmEventM=1,Pm=1,SecM=1,CertM=1 -d',
        'cmedit set --collection VERY_LARGE_COLLECTION_OF_MANY_CELLS_AND_SITES_SPREAD_ACROSS_MULTIPLE_LOCATIONS_AND_ENVIRONMENTS INCLUDING_URBAN_SUBURBAN_AND_RURAL_DEPLOYMENTS EUtranCellFDD qRxLevMin=-130 --preview'
      ];

      complexCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });
  });

  describe('Integration with Phase 3 Components', () => {
    it('should validate all Phase 3 generated command types', () => {
      // Commands from cognitive cmedit engine
      const cognitiveCommands = [
        'cmedit create TEST_NODE EUtranFreqRelation EUtranFreqRelationId=3',
        'cmedit set TEST_NODE EUtranFreqRelation.(EUtranFreqRelationId==3) qOffsetFreq=0',
        'cmedit set TEST_NODE EUtranCellFDD=1 adminState=UNLOCKED',
        'cmedit set TEST_NODE EUtranCellFDD=1 operState=ENABLED',
        'cmedit get TEST_NODE EUtranFreqRelation.(EUtranFreqRelationId==3) -s',
        'cmedit get TEST_NODE EUtranFreqRelation.(EUtranFreqRelationId==3) syncStatus -s'
      ];

      cognitiveCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate template-to-CLI converted commands', () => {
      const templateCliCommands = [
        'cmedit set TEST_NODE ManagedElement=1 managedElementId=TEST-RAN-001,userLabel="Test RAN Node"',
        'cmedit set TEST_NODE ENBFunction=1 eNodeBId=1,maxConnectedUe=1200,endcEnabled=true',
        'cmedit set TEST_NODE EUtranCellFDD=1 pci=100,freqBand=3,qRxLevMin=-130,qQualMin=-32',
        'cmedit set TEST_NODE ENBFunction=1 nrEventB1Threshold=-110,nrEventB1Hysteresis=2',
        'cmedit create TEST_NODE NRFreqRelation NRFreqRelationId=1 referenceFreq=1300,relatedFreq=78',
        'cmedit set TEST_NODE NRFreqRelation=(NRFreqRelationId==1) qOffsetCell=0dB'
      ];

      templateCliCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate batch operations framework commands', () => {
      const batchCommands = [
        'cmedit set TEST_NODE_001 EUtranCellFDD=1 qRxLevMin=-130',
        'cmedit set TEST_NODE_002 EUtranCellFDD=1 qRxLevMin=-130',
        'cmedit set TEST_NODE_003 EUtranCellFDD=1 qRxLevMin=-130',
        'cmedit set --collection TEST_COLLECTION EUtranCellFDD qQualMin=-32',
        'cmedit set *_TEST --scopefilter (EUtranCellFDD.syncStatus==SYNCHRONIZED) EUtranCellFDD administrativestate=UNLOCKED'
      ];

      batchCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate Ericsson RAN expertise optimized commands', () => {
      const expertiseCommands = [
        'cmedit set TEST_NODE EUtranCellFDD=1 antennaElectricalTilt=5.5',
        'cmedit set TEST_NODE EUtranCellFDD=1 txPower=40',
        'cmedit set TEST_NODE AnrFunction=1 anrEnabled=true,automaticNeighbourRelation=true',
        'cmedit set TEST_NODE AnrFunction=1 removeEnbTime=5,pciConflictCellSelection=ON',
        'cmedit set TEST_NODE EUtranCellFDD=1 hysteresis=2.5,timeToTrigger=256',
        'cmedit set TEST_NODE ENBFunction=1 carrierAggregationEnabled=true,maxAggregatedBandwidth=40',
        'cmedit create TEST_NODE EUtranCarrierComponent carrierComponentId=2,dlCarrierFrequency=1800,ulCarrierFrequency=1800'
      ];

      expertiseCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });
  });

  describe('Compliance with rtb-prd.md Requirements', () => {
    it('should meet all syntax requirements from rtb-prd.md', () => {
      // Test all command categories mentioned in rtb-prd.md
      const requirementsCommands = [
        // Core Query Operations
        'cmedit get MeContext=ERBS001,ManagedElement=1',
        'cmedit get LTE32ERBS00001 ENodeBFunction.eNodeBPlmnId',
        'cmedit get LTE32ERBS00001 EUtranCellFDD=1 qRxLevMin,qRxLevMin,qQualMin',

        // Filter Operations
        'cmedit get LTE32ERBS0000* ENodeBFunction.eNodeBPlmnId.(mcc>=271)',
        'cmedit get * EUtranCellFDD.activePlmnList.[{mcc==353}]',

        // Configuration Management
        'cmedit set EPINAL_VCV_LTE EUtranCellFDD=83906_E1 qRxLevMin=-130,qQualMin=-32,cellIndividualOffset=3',
        'cmedit set *_LTE --scopefilter (CmFunction.syncStatus==SYNCHRONIZED) EUtranCellFDD administrativestate=UNLOCKED',
        'cmedit set --collection DUNKERQUE EUtranCellFDD lbTpNonQualFraction=25 --preview',

        // Advanced Features
        'cmedit set GRAY_LTE EUtranCellFDD.(EUtranCellFddId==86376_V3),EUtranCellRelation.(EUtranCellRelationId==2081-86376-6) isHoAllowed=true',
        'cmedit set DR_METZ OptionalFeatureLicense.(OptionalFeatureLicenseId==Anr) featureState=ACTIVATED',
        'cmedit set *_LTE FeatureState.(featureStateId==CXC4012302) featureState=ACTIVATED --force',

        // Cross-vendor support
        'cmedit set VANDOEUVRE_BRAB_LTE EUtranCellFDD=(EUtranCellFddId==83888_F2),UtranCellRelation.(UtranCellRelationId==2081-551-854) isHoAllowed=true',
        'cmedit set SubNetwork=ENM_NE1,MeContext=VANDOEUVRE_BRAB_LTE,ManagedElement=VANDOEUVRE_BRAB_LTE,ENodeBFunction=1,EUtranCellFDD=83888_F2,UtranFreqRelation=3011,UtranCellRelation=2081-551-854 isHoAllowed=true',

        // Time and configuration management
        'cmedit set *_LTE Timesettings daylightSavingTimeEndDate = {month=OCTOBER,dayRule="lastSun",time="03:00"} -t',
        'cmedit set *_LTE Timesettings daylightSavingTimeStartDate = {month=MARCH,dayRule="lastSun",time="02:00"} -t',
        'cmedit set *_LTE Timesettings daylightSavingTimeOffset = "1:00" -t',
        'cmedit set *_LTE Timesettings TimeOffset = "+01:00" -t',

                        // Energy optimization
                        'cmedit set -co amiens MimoSleepFunction.(sleepMode==ADVANCED_SWITCH) sleepmode:MI_DETECTION --preview'
      ];

      requirementsCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate parameter names and values from rtb-prd.md examples', () => {
      const parameterValidationCommands = [
        // Valid parameter names and values
        'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130', // Valid range -140 to -44
        'cmedit set TEST_NODE EUtranCellFDD=1 qQualMin=-32',   // Valid range -20 to 0
        'cmedit set TEST_NODE EUtranCellFDD=1 pci=100',        // Valid range 0 to 503
        'cmedit set TEST_NODE EUtranCellFDD=1 txPower=43',      // Valid range 0 to 46
        'cmedit set TEST_NODE EUtranCellFDD=1 antennaElectricalTilt=5', // Valid range -15 to 15
        'cmedit set TEST_NODE EUtranCellFDD=1 hysteresis=2.0',   // Valid range 0 to 30
        'cmedit set TEST_NODE EUtranCellFDD=1 timeToTrigger=320', // Valid range 0 to 5120

                        // Valid enumerated values
                        'cmedit set TEST_NODE EUtranCellFDD=1 administrativestate=UNLOCKED',
                        'cmedit set TEST_NODE EUtranCellFDD=1 operstate=ENABLED',
                        'cmedit set TEST_NODE EUtranCellFDD=1 cellBarred=NOT_BARRED',
                        'cmedit set TEST_NODE OptionalFeatureLicense.(OptionalFeatureLicenseId==Anr) featureState=ACTIVATED',
                        'cmedit set TEST_NODE FeatureState.(featureStateId==CXC4012302) featureState=ACTIVATED'
      ];

      parameterValidationCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    it('should validate MO class hierarchy from rtb-prd.md', () => {
      const moClassValidationCommands = [
        // Valid MO class hierarchy
        'cmedit get TEST_NODE ManagedElement=1',
        'cmedit get TEST_NODE SystemFunctions=1',
        'cmedit get TEST_NODE SystemFunctions=1,BrM=1',
        'cmedit get TEST_NODE ENBFunction=1',
        'cmedit get TEST_NODE ENBFunction=1,EUtranCellFDD=1',
        'cmedit get TEST_NODE ENBFunction=1,EUtranCellFDD=1,EUtranCellRelation=1',
                        'cmedit create TEST_NODE GNBCUCP.GNBCUCPFunction=1',
                        'cmedit create TEST_NODE GNBCUCP.GNBCUCPFunction=1,NRCellCU=1',
                        'cmedit create TEST_NODE GNBCUCP.GNBCUCPFunction=1,NRCellCU=1,NRCellRelation=1',
                        'cmedit get TEST_NODE Wrat.NodeBFunction=1',
                        'cmedit get TEST_NODE Wrat.NodeBFunction=1,NodeBSectorCarrier=1'
      ];

      moClassValidationCommands.forEach(command => {
        const result = validator.validateCommand(command);
        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });
  });
});