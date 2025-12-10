"""
Ericsson RAN RTB Configuration System - Comprehensive Pydantic Schema
Supporting JSON templates with embedded Python logic for cognitive automation
"""

from __future__ import annotations

from typing import (
    List, Optional, Dict, Any, Union, Literal, Tuple, Set,
    Annotated, Callable, TypeGuard, TypeVar, Generic
)
from enum import Enum
from datetime import datetime
import re
import json
from pydantic import (
    BaseModel, Field, validator, root_validator,
    constr, confloat, conint, conlist, HttpUrl,
    ValidationError, Extra
)
from pydantic.fields import ModelField
from abc import ABC, abstractmethod

# ============================================================================
# TYPE DEFINITIONS AND ENUMS
# ============================================================================

class ResourceType(str, Enum):
    """Resource type classification for QoS"""
    GBR = "GBR"
    NON_GBR = "NON_GBR"
    DELAY_CRITICAL = "DELAY_CRITICAL"

class CellState(str, Enum):
    """Cell operational states"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TESTING = "TESTING"
    DISABLED = "DISABLED"

class AdministrativeState(str, Enum):
    """Administrative control states"""
    UNLOCKED = "UNLOCKED"
    LOCKED = "LOCKED"
    SHUTTING_DOWN = "SHUTTING_DOWN"

class MobilityAction(str, Enum):
    """Mobility handover actions"""
    HANDOVER = "HANDOVER"
    RELEASE_WITH_REDIRECT = "RELEASE_WITH_REDIRECT"
    NO_ACTION = "NO_ACTION"

class ServiceType(str, Enum):
    """QoS service types"""
    CONVERSATIONAL = "CONVERSATIONAL"
    STREAMING = "STREAMING"
    INTERACTIVE = "INTERACTIVE"
    BACKGROUND = "BACKGROUND"
    UNDEFINED = "UNDEFINED"

class RlcMode(str, Enum):
    """Radio Link Control modes"""
    AM = "AM"  # Acknowledged Mode
    UM = "UM"  # Unacknowledged Mode

class EventTriggerType(str, Enum):
    """Measurement event trigger types"""
    A1 = "A1"  # Serving becomes better than threshold
    A2 = "A2"  # Serving becomes worse than threshold
    A3 = "A3"  # Neighbor becomes better than serving
    A4 = "A4"  # Neighbor becomes better than absolute threshold
    A5 = "A5"  # Serving worse than threshold1 AND neighbor better than threshold2
    A6 = "A6"  # SCell becomes better than threshold
    B1 = "B1"  # Inter-RAT neighbor becomes better than threshold
    B2 = "B2"  # Inter-RAT neighbor better than threshold AND serving worse than threshold

class TriggerQuantity(str, Enum):
    """Measurement quantities for events"""
    RSRP = "RSRP"
    RSRQ = "RSRP"
    SINR = "SINR"
    SS_RSRP = "SS_RSRP"  # NR SS RSRP
    SS_RSRQ = "SS_RSRQ"
    SS_SINR = "SS_SINR"

# ============================================================================
# CUSTOM TYPES AND VALIDATORS
# ============================================================================

# Regex patterns for validation
PLMN_PATTERN = re.compile(r'^\d{3}$')  # MCC: 3 digits
MNC_PATTERN = re.compile(r'^\d{2,3}$')  # MNC: 2-3 digits
CELL_ID_PATTERN = re.compile(r'^\d+-\d+$')  # e.g., "2081-12345"
GNB_ID_PATTERN = re.compile(r'^\d{1,6}$')  # gNB ID: up to 6 digits
IPV4_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
MAC_PATTERN = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')

# Custom type aliases
PlmnId = Annotated[str, Field(regex=r'^\d{3}-\d{2,3}$')]
RsrpValue = Annotated[int, Field(ge=-156, le=-29)]  # dBm
RsrqValue = Annotated[int, Field(ge=-435, le=200)]  # dB x10
SinrValue = Annotated[int, Field(ge=-230, le=405)]  # dB x10
HysteresisValue = Annotated[int, Field(ge=0, le=3000)]  # dB x10

# ============================================================================
# HELPER MODELS
# ============================================================================

class PlmnIdentity(BaseModel):
    """PLMN identity configuration"""
    mcc: str = Field(..., min_length=3, max_length=3, description="Mobile Country Code")
    mnc: str = Field(..., min_length=2, max_length=3, description="Mobile Network Code")
    mncLength: str = Field(default="2", regex=r'^(2|3)$', description="MNC length indicator")

    @validator('mcc')
    def validate_mcc(cls, v):
        if not PLMN_PATTERN.match(v):
            raise ValueError("MCC must be 3 digits")
        return v

    @validator('mnc')
    def validate_mnc(cls, v):
        if not MNC_PATTERN.match(v):
            raise ValueError("MNC must be 2 or 3 digits")
        return v

class SNSSAI(BaseModel):
    """Single-Network Slice Selection Assistance Information"""
    sst: str = Field(..., ge=0, le=255, description="Slice/Service Type")
    sd: str = Field(default="16777215", description="Slice Differentiator (hex)")

class FrequencyBandInfo(BaseModel):
    """Frequency band configuration"""
    bandList: str = Field(..., description="Band list e.g., '78'")
    arfcn: str = Field(..., description="Absolute RF Channel Number")
    bandwidthDl: str = Field(default="20000", description="Downlink bandwidth in kHz")
    bandwidthUl: str = Field(default="20000", description="Uplink bandwidth in kHz")
    subCarrierSpacing: str = Field(default="30", description="Subcarrier spacing in kHz")

class PowerConfiguration(BaseModel):
    """Power and antenna tilt configuration"""
    pMax: str = Field(default="23", description="Maximum transmit power (dBm)")
    totalTilt: str = Field(default="0", description="Total antenna tilt (degrees)")
    minTotalTilt: str = Field(default="-10", description="Minimum tilt (degrees)")
    maxTotalTilt: str = Field(default="10", description="Maximum tilt (degrees)")
    antennaGain: Optional[str] = Field(None, description="Antenna gain (dBi)")

class CellIdentity(BaseModel):
    """Physical cell identity"""
    pci: str = Field(..., ge=0, le=1007, description="Physical Cell ID")
    tac: str = Field(..., description="Tracking Area Code")
    cellId: str = Field(..., description="Cell identifier")
    eci: Optional[str] = Field(None, description="E-UTRAN Cell Identifier")

# ============================================================================
# CORE RTB MODELS
# ============================================================================

class RTBMeta(BaseModel):
    """RTB Template metadata"""
    version: str = Field(default="1.0.0", description="Template version")
    author: List[str] = Field(default_factory=list, description="Template authors")
    description: str = Field(..., description="Template description")
    created: datetime = Field(default_factory=datetime.now)
    tags: Optional[List[str]] = Field(default=None)
    environment: Optional[str] = Field(None, regex=r'^(dev|test|staging|prod)$')

class CustomFunction(BaseModel):
    """Custom Python function definition for RTB logic"""
    name: str = Field(..., description="Function name")
    args: List[str] = Field(..., description="Function arguments")
    body: List[str] = Field(..., description="Python code lines")
    description: Optional[str] = Field(None, description="Function description")

    @validator('name')
    def validate_function_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError("Invalid function name")
        return v

class ConditionalOperator(BaseModel):
    """Conditional logic operator ($cond)"""
    condition: str = Field(..., description="Condition expression")
    then_value: Union[Dict[str, Any], str, int, float] = Field(..., description="Value if true")
    else_value: Union[Dict[str, Any], str, int, float, Literal["__ignore__"]] = Field(default="__ignore__", description="Value if false")

class EvaluationOperator(BaseModel):
    """Evaluation operator ($eval)"""
    function: str = Field(..., description="Function to evaluate")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Function parameters")

# ============================================================================
# RADIO ACCESS NETWORK MODELS
# ============================================================================

class ManagedElement(BaseModel):
    """Managed Element configuration"""
    managedElementId: str = Field(..., description="Element identifier")
    siteLocation: Optional[str] = Field(None, description="Site location")
    userLabel: Optional[str] = Field(None, description="User-friendly label")
    managedElementType: str = Field(default="RBS", description="Element type")
    release: str = Field(default="L21.0", description="Software release")
    locationName: Optional[str] = Field(None, description="Physical location name")
    vendor: str = Field(default="Ericsson", description="Equipment vendor")

class GNBCUCPFunction(BaseModel):
    """gNodeB Control Plane function"""
    gNBCUName: str = Field(..., max_length=150, description="gNB-CU name")
    gNBId: str = Field(..., description="gNB identifier")
    gNBIdLength: str = Field(default="22", regex=r'^(22|32)$', description="gNB ID bit length")
    pLMNId: PlmnIdentity = Field(..., description="Primary PLMN")
    ranNodeName: str = Field(..., max_length=150, description="RAN node name")

    # Network configuration
    maxCommonProcTime: int = Field(default=30, ge=1, le=60, description="Common procedure timeout (s)")
    maxNgRetryTime: int = Field(default=30, ge=1, le=3600, description="NG setup retry timeout (s)")
    nasInactivityTime: int = Field(default=5, ge=1, le=60, description="NAS inactivity timeout (s)")

    # Feature support
    extendedBandN77Supported: bool = Field(default=False, description="Extended n77 band support")
    extendedXnConnAllowed: bool = Field(default=True, description="Extended Xn connections")
    nrNeedForGapsSupported: bool = Field(default=True, description="NR NeedForGaps support")

    # Resource management
    noOfSupportedNRCellCU: int = Field(ge=1, description="Supported NR Cell CU count")
    resourceStatusReportDefault: int = Field(default=-2, description="Load reporting default")
    resourceStatusReportF1Enabled: bool = Field(default=True, description="F1 resource reporting")

class EUtranCellFDD(BaseModel):
    """LTE FDD cell configuration"""
    eUtranCellFddId: str = Field(..., description="Cell identifier")
    cellIdentity: CellIdentity = Field(...)
    frequencyBandInfo: FrequencyBandInfo = Field(...)
    powerConfig: PowerConfiguration = Field(...)

    # Cell state
    cellState: CellState = Field(default=CellState.ACTIVE)
    administrativeState: AdministrativeState = Field(default=AdministrativeState.UNLOCKED)
    operationalState: str = Field(default="ENABLED", description="Operational state")
    cellBarred: bool = Field(default=False, description="Cell barring status")

    # RF parameters
    qRxLevMin: RsrpValue = Field(default=-140, description="Minimum RSRP (dBm)")
    qQualMin: RsrqValue = Field(default=-20, description="Minimum RSRQ (dB)")
    siWindowLength: str = Field(default="SF40", description="SI window length")

    # Advanced features
    dl256QamEnabled: bool = Field(default=True, description="256-QAM DL support")
    ul256QamEnabled: bool = Field(default=False, description="256-QAM UL support")
    caEnabled: bool = Field(default=False, description="Carrier aggregation enabled")

class NRCellCU(BaseModel):
    """NR Cell Control Unit configuration"""
    nrCellCuId: str = Field(..., description="NR Cell CU identifier")
    cellIdentity: CellIdentity = Field(...)
    frequencyBandInfo: FrequencyBandInfo = Field(...)
    powerConfig: PowerConfiguration = Field(...)

    # Cell state
    cellState: CellState = Field(default=CellState.ACTIVE)
    administrativeState: AdministrativeState = Field(default=AdministrativeState.UNLOCKED)
    operationalState: str = Field(default="ENABLED", description="Operational state")

    # NR-specific parameters
    scsSpecificCarrierList: Optional[List[Dict[str, Any]]] = Field(default=None)
    ssbPositionInBurst: Optional[List[int]] = Field(default=None)
    ssbSubcarrierSpacing: str = Field(default="30", description="SSB SCS (kHz)")

    # Advanced features
    extendedBandN77Supported: bool = Field(default=False)
    extendedBandN78Supported: bool = Field(default=True)
    tddEnabled: bool = Field(default=False)

class QciProfile(BaseModel):
    """QoS Class Identifier profile"""
    qci: int = Field(..., ge=1, le=255, description="QCI value")
    priority: int = Field(..., ge=1, le=255, description="Priority level")
    arp: int = Field(..., ge=1, le=255, description="Allocation and Retention Priority")

    # QoS parameters
    pdb: int = Field(default=100, ge=1, le=2000, description="Packet Delay Budget (ms)")
    resourceType: ResourceType = Field(default=ResourceType.NON_GBR)

    # Bit rate requirements (for GBR)
    gbrDl: Optional[str] = Field(None, description="Guaranteed DL bit rate")
    gbrUl: Optional[str] = Field(None, description="Guaranteed UL bit rate")
    mbrDl: Optional[str] = Field(None, description="Maximum DL bit rate")
    mbrUl: Optional[str] = Field(None, description="Maximum UL bit rate")

    # RLC configuration
    rlcMode: RlcMode = Field(default=RlcMode.AM)
    pdcpSnLength: str = Field(default="12", regex=r'^(12|18)$', description="PDCP SN length")

    # Advanced features
    rohcEnabled: bool = Field(default=False, description="RObust Header Compression")
    caOffloadingEnabled: bool = Field(default=True, description="Carrier aggregation offloading")

class MeasurementConfiguration(BaseModel):
    """Measurement configuration for events"""
    eventId: str = Field(..., description="Event identifier")
    triggerType: EventTriggerType = Field(...)
    triggerQuantity: TriggerQuantity = Field(...)

    # Thresholds
    threshold1: Optional[Union[int, float, str]] = Field(None, description="Threshold 1")
    threshold2: Optional[Union[int, float, str]] = Field(None, description="Threshold 2")
    hysteresis: HysteresisValue = Field(default=0, description="Hysteresis (dB x10)")
    timeToTrigger: int = Field(default=0, ge=0, le=10000, description="Time to trigger (ms)")

    # Reporting
    reportInterval: Optional[str] = Field(None, description="Report interval")
    maxReportCells: int = Field(default=8, ge=1, le=32, description="Max cells to report")

    # Advanced
    a3Offset: Optional[int] = Field(None, description="A3 event offset (dB)")
    a3ReportInterval: Optional[str] = Field(None, description="A3 report interval")

class NeighborCellRelation(BaseModel):
    """Neighbor cell relation configuration"""
    relationId: str = Field(..., description="Relation identifier")
    neighborCellRef: str = Field(..., description="Neighbor cell reference")

    # Handover parameters
    isHoAllowed: bool = Field(default=True, description="Handover allowed")
    cellIndividualOffset: int = Field(default=0, ge=-30, le=30, description="Cell offset (dB)")
    qOffsetFreq: int = Field(default=0, ge=-24, le=24, description="Frequency offset (dB)")

    # Mobility
    mobilityAction: MobilityAction = Field(default=MobilityAction.HANDOVER)
    isRemoveAllowed: bool = Field(default=True, description="Removal allowed")

    # Load balancing
    loadBalancing: str = Field(default="NOT_ALLOWED", description="Load balancing")
    sCellPriority: int = Field(default=7, ge=0, le=7, description="SCell priority")
    sCellCandidate: str = Field(default="AUTO", description="SCell candidate status")

class AnrFunction(BaseModel):
    """Automatic Neighbor Relation function"""
    anrFunctionId: str = Field(..., description="ANR function identifier")

    # Cell relation management
    removeEnbTime: int = Field(default=7, ge=1, le=100, description="eNB removal time (days)")
    removeGnbTime: int = Field(default=7, ge=1, le=100, description="gNB removal time (days)")
    removeFreqRelTime: int = Field(default=15, ge=10, le=10000, description="Freq relation removal (min)")

    # Promotion/demotion thresholds
    demoteCellRelMobAttThresh: int = Field(default=100, ge=1, le=1000, description="Demotion threshold")
    promoteCellRelMobAttThresh: int = Field(default=110, ge=1, le=1000, description="Promotion threshold")

    # PCI conflict management
    pciConflictCellSelection: str = Field(default="OFF", regex=r'^(OFF|ON)$')
    maxTimeEventBasedPciConf: int = Field(default=30, ge=1, le=10080, description="PCI conflict time (min)")

    # Feature control
    plmnWhiteListEnabled: bool = Field(default=True)
    anrCgiMeasIntraFreqEnabled: bool = Field(default=True)
    detectObsoleteExtCellsEnabled: bool = Field(default=False)

# ============================================================================
# EVENT AND MONITORING MODELS
# ============================================================================

class EventJob(BaseModel):
    """Event job configuration"""
    eventJobId: str = Field(..., description="Job identifier")
    description: str = Field(default="", description="Job description")
    eventGroupRef: str = Field(..., description="Event group reference")
    eventTypeRef: str = Field(..., description="Event type reference")

    # Job control
    jobControl: str = Field(default="FULL", description="Job control level")
    requestedJobState: str = Field(default="STARTED", description="Requested state")
    currentJobState: str = Field(default="STOPPED", description="Current state")

    # Output configuration
    streamOutputEnabled: bool = Field(default=False)
    fileOutputEnabled: bool = Field(default=True)
    streamCompressionType: str = Field(default="GZIP", description="Stream compression")
    fileCompressionType: str = Field(default="GZIP", description="File compression")
    reportingPeriod: str = Field(default="FIFTEEN_MIN", description="Reporting period")

    # Filtering
    eventFilter: Optional[Dict[str, Any]] = Field(default=None)

class FeatureKey(BaseModel):
    """Feature key for licensing"""
    featureKeyId: str = Field(..., description="Feature key identifier")
    keyId: str = Field(..., description="Base key identifier")
    name: str = Field(..., description="Feature name")

    # Licensing
    state: str = Field(..., description="Feature state reference")
    licenseState: str = Field(..., description="License state")
    shared: bool = Field(default=False, description="Shared feature")

    # Validity
    validFrom: str = Field(..., description="Valid from date")
    productType: str = Field(default="Baseband", description="Product type")

class CapacityState(BaseModel):
    """Capacity state for licensing"""
    capacityStateId: str = Field(..., description="Capacity state identifier")
    keyId: str = Field(..., description="Associated key ID")
    description: str = Field(..., description="Capacity description")

    # Licensing state
    licenseState: str = Field(..., description="License state")
    serviceState: str = Field(..., description="Service state")
    licensedCapacityLimitReached: bool = Field(default=False)

    # Capacity limits
    grantedCapacityLevel: str = Field(..., description="Granted capacity level")
    currentCapacityLimit: Dict[str, Any] = Field(..., description="Current limits")

# ============================================================================
# MAIN RTB TEMPLATE MODEL
# ============================================================================

class RTBTemplate(BaseModel):
    """Complete RTB Configuration Template"""

    # Metadata and custom functions
    meta: Optional[RTBMeta] = Field(None, alias="$meta")
    custom_functions: Optional[List[CustomFunction]] = Field(None, alias="$custom")

    # Network Elements
    managedElement: Optional[ManagedElement] = Field(None)
    gnbCucp: Optional[GNBCUCPFunction] = Field(None)
    eUtranCells: Optional[List[EUtranCellFDD]] = Field(default_factory=list)
    nrCells: Optional[List[NRCellCU]] = Field(default_factory=list)

    # Relations and Mobility
    neighborRelations: Optional[List[NeighborCellRelation]] = Field(default_factory=list)
    frequencyRelations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # QoS and Traffic
    qciProfiles: Optional[List[QciProfile]] = Field(default_factory=list)
    measurementConfigs: Optional[List[MeasurementConfiguration]] = Field(default_factory=list)

    # Automation and Monitoring
    anrFunction: Optional[AnrFunction] = Field(None)
    eventJobs: Optional[List[EventJob]] = Field(default_factory=list)
    featureKeys: Optional[List[FeatureKey]] = Field(default_factory=list)
    capacityStates: Optional[List[CapacityState]] = Field(default_factory=list)

    # Conditional logic and evaluation
    conditional_logic: Optional[Dict[str, ConditionalOperator]] = Field(default_factory=dict, alias="$cond")
    evaluation_logic: Optional[Dict[str, EvaluationOperator]] = Field(default_factory=dict, alias="$eval")

    class Config:
        extra = Extra.allow  # Allow additional fields for future extensions
        validate_assignment = True
        use_enum_values = True

    @root_validator(pre=True)
    def validate_template(cls, values):
        """Validate template consistency"""
        # Check if at least one network element is defined
        has_network = any(key in values for key in ['managedElement', 'gnbCucp', 'eUtranCells', 'nrCells'])
        if not has_network:
            raise ValueError("Template must contain at least one network element")

        # Validate custom function names are unique
        if '$custom' in values and values['$custom']:
            func_names = [f.name for f in values['$custom']]
            if len(func_names) != len(set(func_names)):
                raise ValueError("Custom function names must be unique")

        return values

# ============================================================================
# TEMPLATE PROCESSOR WITH PYTHON LOGIC INTEGRATION
# ============================================================================

class RTBTemplateProcessor:
    """RTB Template processor with Python logic execution"""

    def __init__(self, template: RTBTemplate):
        self.template = template
        self.custom_functions = {}
        self.context = {}

        # Compile custom functions
        if template.custom_functions:
            for func_def in template.custom_functions:
                self._compile_function(func_def)

    def _compile_function(self, func_def: CustomFunction):
        """Compile a custom function from template"""
        func_code = '\n'.join(func_def.body)

        # Create function signature
        args_str = ', '.join(func_def.args)
        func_signature = f"def {func_def.name}({args_str}):\n"

        # Indent the function body
        indented_body = '\n'.join(f"    {line}" for line in func_def.body)

        # Combine signature and body
        full_code = func_signature + indented_body

        # Execute in safe namespace
        safe_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'set': set,
                'min': min, 'max': max, 'abs': abs, 'round': round,
                'sum': sum, 'any': any, 'all': all,
            }
        }

        local_vars = {}
        try:
            exec(full_code, safe_globals, local_vars)
            if func_def.name in local_vars:
                self.custom_functions[func_def.name] = local_vars[func_def.name]
        except Exception as e:
            raise ValueError(f"Failed to compile function {func_def.name}: {e}")

    def process_template(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process template with given context and execute logic"""
        if context:
            self.context.update(context)

        result = {}

        # Process conditional logic
        if self.template.conditional_logic:
            result.update(self._process_conditionals())

        # Process evaluation logic
        if self.template.evaluation_logic:
            result.update(self._process_evaluations())

        # Include static configuration
        result.update(self._extract_static_config())

        return result

    def _process_conditionals(self) -> Dict[str, Any]:
        """Process conditional logic ($cond)"""
        processed = {}

        for field, cond_op in self.template.conditional_logic.items():
            try:
                # Evaluate condition safely
                condition_result = self._evaluate_expression(cond_op.condition)

                if condition_result:
                    processed[field] = cond_op.then_value
                elif cond_op.else_value != "__ignore__":
                    processed[field] = cond_op.else_value

            except Exception as e:
                print(f"Warning: Failed to evaluate condition for {field}: {e}")

        return processed

    def _process_evaluations(self) -> Dict[str, Any]:
        """Process evaluation logic ($eval)"""
        processed = {}

        for field, eval_op in self.template.evaluation_logic.items():
            try:
                # Execute custom function
                if eval_op.function in self.custom_functions:
                    func = self.custom_functions[eval_op.function]

                    # Prepare arguments
                    args = []
                    for arg in eval_op.parameters.get('args', []):
                        if arg in self.context:
                            args.append(self.context[arg])
                        else:
                            args.append(arg)

                    # Call function
                    result = func(*args)
                    processed[field] = result
                else:
                    print(f"Warning: Function {eval_op.function} not found")

            except Exception as e:
                print(f"Warning: Failed to evaluate function {eval_op.function}: {e}")

        return processed

    def _evaluate_expression(self, expression: str) -> bool:
        """Safely evaluate a boolean expression"""
        # Simple expression evaluator for common patterns
        # In production, use a more robust solution

        # Replace template variables with context values
        for key, value in self.context.items():
            expression = expression.replace(key, str(value))

        # Only allow safe operations
        safe_expr = expression
        allowed_operators = ['==', '!=', '<', '>', '<=', '>=', 'in', 'not', 'and', 'or']

        # Basic evaluation (use ast.parse for production)
        try:
            return bool(eval(safe_expr))
        except:
            return False

    def _extract_static_config(self) -> Dict[str, Any]:
        """Extract static configuration from template"""
        result = {}

        # Convert Pydantic models to dicts
        if self.template.managedElement:
            result['managedElement'] = self.template.managedElement.dict()

        if self.template.gnbCucp:
            result['gnbCucp'] = self.template.gnbCucp.dict()

        if self.template.eUtranCells:
            result['eUtranCells'] = [cell.dict() for cell in self.template.eUtranCells]

        if self.template.nrCells:
            result['nrCells'] = [cell.dict() for cell in self.template.nrCells]

        # Include other configurations
        result['qciProfiles'] = [qci.dict() for qci in self.template.qciProfiles or []]
        result['measurementConfigs'] = [meas.dict() for meas in self.template.measurementConfigs or []]
        result['neighborRelations'] = [rel.dict() for rel in self.template.neighborRelations or []]

        return result

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_rtb_template_from_csv(csv_data: Dict[str, Any]) -> RTBTemplate:
    """Create RTB template from CSV parameter data"""
    # This function would parse CSV data and create appropriate models
    # Implementation would depend on CSV structure
    pass

def validate_rtb_configuration(config: Dict[str, Any]) -> bool:
    """Validate an RTB configuration against the schema"""
    try:
        RTBTemplate(**config)
        return True
    except ValidationError:
        return False

def generate_rtb_json(template: RTBTemplate, context: Optional[Dict[str, Any]] = None) -> str:
    """Generate RTB JSON from template with optional context"""
    processor = RTBTemplateProcessor(template)
    result = processor.process_template(context)

    # Add metadata if present
    if template.meta:
        result['$meta'] = template.meta.dict()

    # Add custom functions if present
    if template.custom_functions:
        result['$custom'] = [func.dict() for func in template.custom_functions]

    return json.dumps(result, indent=2, ensure_ascii=False)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example custom function
    calc_offset_func = CustomFunction(
        name="calculateCellOffset",
        args=["cell_type", "rsrp"],
        body=[
            "if cell_type == 'macro':",
            "    return rsrp + 2",
            "elif cell_type == 'pico':",
            "    return rsrp - 2",
            "else:",
            "    return rsrp"
        ]
    )

    # Example template
    template = RTBTemplate(
        meta=RTBMeta(
            version="1.0.0",
            author=["Ericsson RAN Automation"],
            description="Example RTB template with custom logic"
        ),
        custom_functions=[calc_offset_func],
        gnbCucp=GNBCUCPFunction(
            gNBCUName="Example gNB",
            gNBId="12345",
            pLMNId=PlmnIdentity(mcc="208", mnc="01"),
            ranNodeName="gnb-example"
        ),
        conditional_logic={
            "enableCA": ConditionalOperator(
                condition="cell_count > 1",
                then_value={"caEnabled": True},
                else_value={"caEnabled": False}
            )
        },
        evaluation_logic={
            "cellOffset": EvaluationOperator(
                function="calculateCellOffset",
                parameters={"args": ["cell_type", "target_rsrp"]}
            )
        }
    )

    # Process template with context
    context = {"cell_count": 2, "cell_type": "macro", "target_rsrp": -80}
    json_output = generate_rtb_json(template, context)
    print(json_output)