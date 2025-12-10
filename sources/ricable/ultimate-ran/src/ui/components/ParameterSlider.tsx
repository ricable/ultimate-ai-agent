/**
 * Parameter Slider Component
 * Interactive controls for P0 (transmit power) and Alpha (tilt) adjustments
 *
 * @module ui/components/ParameterSlider
 * @version 7.0.0-alpha.1
 */

import React, { useState, useEffect } from 'react';
import type { CellStatus } from '../types.js';

// ============================================================================
// Types
// ============================================================================

export interface ParameterDefinition {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  description?: string;
}

interface ParameterSliderProps {
  cell: CellStatus;
  parameter: ParameterDefinition;
  onChange: (cellId: string, parameter: string, value: number) => void;
  onCommit?: (cellId: string, parameter: string, value: number) => void;
  disabled?: boolean;
  showBounds?: boolean;
  showRecommendation?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

function calculateRecommendation(
  parameter: string,
  cell: CellStatus
): { value: number; reason: string } | null {
  // Simplified recommendation logic based on KPI
  if (parameter === 'power_dbm') {
    if (cell.kpi.rsrp_avg < -110) {
      return {
        value: Math.min(cell.power_dbm + 3, 46),
        reason: 'Weak RSRP detected, recommend increasing power'
      };
    } else if (cell.kpi.sinr_avg < 5 && cell.kpi.rsrp_avg > -90) {
      return {
        value: Math.max(cell.power_dbm - 2, -130),
        reason: 'High interference with strong signal, recommend reducing power'
      };
    }
  } else if (parameter === 'tilt_deg') {
    if (cell.kpi.prb_utilization > 80 && cell.kpi.sinr_avg < 10) {
      return {
        value: Math.min(cell.tilt_deg + 1, 15),
        reason: 'High load with interference, recommend increasing tilt'
      };
    }
  }

  return null;
}

function getValueColor(value: number, min: number, max: number): string {
  const normalized = (value - min) / (max - min);

  if (normalized < 0.3) return '#2ECC71'; // Green (low range)
  if (normalized < 0.7) return '#FFA500'; // Orange (mid range)
  return '#E74C3C'; // Red (high range)
}

function formatValue(value: number, step: number): string {
  const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return value.toFixed(decimals);
}

// ============================================================================
// Component
// ============================================================================

export const ParameterSlider: React.FC<ParameterSliderProps> = ({
  cell,
  parameter,
  onChange,
  onCommit,
  disabled = false,
  showBounds = true,
  showRecommendation = true
}) => {
  const currentValue = (cell as any)[parameter.name] ?? parameter.min;
  const [localValue, setLocalValue] = useState(currentValue);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    setLocalValue(currentValue);
  }, [currentValue]);

  const recommendation = showRecommendation ? calculateRecommendation(parameter.name, cell) : null;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    setLocalValue(newValue);
    onChange(cell.cell_id, parameter.name, newValue);
  };

  const handleMouseDown = () => {
    setIsDragging(true);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    if (onCommit && localValue !== currentValue) {
      onCommit(cell.cell_id, parameter.name, localValue);
    }
  };

  const handleReset = () => {
    setLocalValue(currentValue);
    onChange(cell.cell_id, parameter.name, currentValue);
  };

  const handleApplyRecommendation = () => {
    if (recommendation) {
      setLocalValue(recommendation.value);
      onChange(cell.cell_id, parameter.name, recommendation.value);
      if (onCommit) {
        onCommit(cell.cell_id, parameter.name, recommendation.value);
      }
    }
  };

  const valueColor = getValueColor(localValue, parameter.min, parameter.max);
  const hasChanged = localValue !== currentValue;
  const delta = localValue - currentValue;
  const percentChange = ((delta / currentValue) * 100) || 0;

  return (
    <div
      style={{
        padding: '16px',
        border: '1px solid #e0e0e0',
        borderRadius: '8px',
        background: '#fff',
        marginBottom: '12px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
            {parameter.label}
          </div>
          {parameter.description && (
            <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              {parameter.description}
            </div>
          )}
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: valueColor, fontFamily: 'monospace' }}>
            {formatValue(localValue, parameter.step)} {parameter.unit}
          </div>
          {hasChanged && (
            <div
              style={{
                fontSize: '12px',
                color: delta > 0 ? '#2ECC71' : '#E74C3C',
                fontWeight: 'bold'
              }}
            >
              {delta > 0 ? '+' : ''}
              {formatValue(delta, parameter.step)} ({delta > 0 ? '+' : ''}
              {percentChange.toFixed(1)}%)
            </div>
          )}
        </div>
      </div>

      {/* Slider */}
      <div style={{ marginBottom: '12px' }}>
        <input
          type="range"
          min={parameter.min}
          max={parameter.max}
          step={parameter.step}
          value={localValue}
          onChange={handleChange}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onTouchStart={handleMouseDown}
          onTouchEnd={handleMouseUp}
          disabled={disabled}
          style={{
            width: '100%',
            height: '8px',
            borderRadius: '4px',
            outline: 'none',
            background: `linear-gradient(to right, #2ECC71 0%, #FFA500 50%, #E74C3C 100%)`,
            opacity: disabled ? 0.5 : 1,
            cursor: disabled ? 'not-allowed' : 'pointer'
          }}
        />

        {/* Bounds Display */}
        {showBounds && (
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: '4px',
              fontSize: '11px',
              color: '#999',
              fontFamily: 'monospace'
            }}
          >
            <span>
              Min: {parameter.min} {parameter.unit}
            </span>
            <span>
              Max: {parameter.max} {parameter.unit}
            </span>
          </div>
        )}
      </div>

      {/* Quick Presets */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <button
          onClick={() => {
            const min = parameter.min;
            setLocalValue(min);
            onChange(cell.cell_id, parameter.name, min);
          }}
          disabled={disabled}
          style={{
            padding: '6px 12px',
            fontSize: '12px',
            background: '#f5f5f5',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: disabled ? 'not-allowed' : 'pointer',
            opacity: disabled ? 0.5 : 1
          }}
        >
          Min
        </button>
        <button
          onClick={() => {
            const mid = (parameter.min + parameter.max) / 2;
            setLocalValue(mid);
            onChange(cell.cell_id, parameter.name, mid);
          }}
          disabled={disabled}
          style={{
            padding: '6px 12px',
            fontSize: '12px',
            background: '#f5f5f5',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: disabled ? 'not-allowed' : 'pointer',
            opacity: disabled ? 0.5 : 1
          }}
        >
          Mid
        </button>
        <button
          onClick={() => {
            const max = parameter.max;
            setLocalValue(max);
            onChange(cell.cell_id, parameter.name, max);
          }}
          disabled={disabled}
          style={{
            padding: '6px 12px',
            fontSize: '12px',
            background: '#f5f5f5',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: disabled ? 'not-allowed' : 'pointer',
            opacity: disabled ? 0.5 : 1
          }}
        >
          Max
        </button>
        {hasChanged && (
          <button
            onClick={handleReset}
            disabled={disabled}
            style={{
              padding: '6px 12px',
              fontSize: '12px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              color: '#E65100',
              fontWeight: 'bold',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1
            }}
          >
            Reset
          </button>
        )}
      </div>

      {/* AI Recommendation */}
      {recommendation && (
        <div
          style={{
            padding: '12px',
            background: '#e3f2fd',
            border: '1px solid #2196F3',
            borderRadius: '4px',
            marginBottom: '8px'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span style={{ fontSize: '16px' }}>💡</span>
            <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#1565C0' }}>
              AI Recommendation
            </div>
          </div>
          <div style={{ fontSize: '12px', color: '#333', marginBottom: '8px' }}>
            {recommendation.reason}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ fontSize: '14px', fontWeight: 'bold', fontFamily: 'monospace' }}>
              Suggested: {formatValue(recommendation.value, parameter.step)} {parameter.unit}
            </div>
            <button
              onClick={handleApplyRecommendation}
              disabled={disabled}
              style={{
                padding: '6px 12px',
                fontSize: '12px',
                background: '#2196F3',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                fontWeight: 'bold',
                cursor: disabled ? 'not-allowed' : 'pointer',
                opacity: disabled ? 0.5 : 1
              }}
            >
              Apply
            </button>
          </div>
        </div>
      )}

      {/* Cell Info */}
      <div
        style={{
          padding: '8px',
          background: '#f9f9f9',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#666'
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
          <div>
            <strong>Cell:</strong> {cell.cell_id}
          </div>
          <div>
            <strong>Status:</strong> {cell.status}
          </div>
          <div>
            <strong>PCI:</strong> {cell.pci}
          </div>
          <div>
            <strong>RSRP:</strong> {cell.kpi.rsrp_avg.toFixed(1)} dBm
          </div>
          <div>
            <strong>SINR:</strong> {cell.kpi.sinr_avg.toFixed(1)} dB
          </div>
          <div>
            <strong>PRB:</strong> {cell.kpi.prb_utilization.toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Change Indicator */}
      {isDragging && (
        <div
          style={{
            marginTop: '8px',
            padding: '8px',
            background: '#fff3e0',
            border: '1px solid #FF9800',
            borderRadius: '4px',
            textAlign: 'center',
            fontSize: '12px',
            fontWeight: 'bold',
            color: '#E65100'
          }}
        >
          ⚠️ Release to commit change
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Multi-Parameter Control Panel
// ============================================================================

interface ParameterControlPanelProps {
  cells: CellStatus[];
  parameters: ParameterDefinition[];
  onChange: (cellId: string, parameter: string, value: number) => void;
  onCommit?: (cellId: string, parameter: string, value: number) => void;
  disabled?: boolean;
}

export const ParameterControlPanel: React.FC<ParameterControlPanelProps> = ({
  cells,
  parameters,
  onChange,
  onCommit,
  disabled = false
}) => {
  const [selectedCellId, setSelectedCellId] = useState<string>(cells[0]?.cell_id || '');

  const selectedCell = cells.find(c => c.cell_id === selectedCellId);

  if (!selectedCell) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#999' }}>
        No cells available for parameter control
      </div>
    );
  }

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <h2 style={{ marginBottom: '16px' }}>Parameter Control Panel</h2>

      {/* Cell Selector */}
      {cells.length > 1 && (
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>
            Select Cell:
          </label>
          <select
            value={selectedCellId}
            onChange={e => setSelectedCellId(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              fontSize: '14px',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          >
            {cells.map(cell => (
              <option key={cell.cell_id} value={cell.cell_id}>
                {cell.cell_id} (PCI: {cell.pci}, Status: {cell.status})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Parameter Sliders */}
      <div>
        {parameters.map(param => (
          <ParameterSlider
            key={param.name}
            cell={selectedCell}
            parameter={param}
            onChange={onChange}
            onCommit={onCommit}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default ParameterSlider;
