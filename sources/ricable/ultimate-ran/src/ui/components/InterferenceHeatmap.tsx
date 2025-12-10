/**
 * Interference Heatmap Component
 * D3.js/visx-based heatmap visualization for cell interference
 *
 * @module ui/components/InterferenceHeatmap
 * @version 7.0.0-alpha.1
 */

import React, { useEffect, useRef, useState } from 'react';
import type { CellStatus, InterferenceMatrix } from '../types.js';

// ============================================================================
// Types
// ============================================================================

interface InterferenceHeatmapProps {
  cells: CellStatus[];
  interferenceMatrix: InterferenceMatrix;
  threshold?: number;
  width?: number;
  height?: number;
  onCellClick?: (cellId: string) => void;
  onCellHover?: (cellId: string | null) => void;
}

interface HeatmapCell {
  row: number;
  col: number;
  value: number;
  cellIdRow: string;
  cellIdCol: string;
  color: string;
}

// ============================================================================
// Color Scale
// ============================================================================

function getInterferenceColor(value: number, threshold: number): string {
  // Green (low) -> Yellow (medium) -> Red (high) -> Dark Red (critical)
  if (value > threshold + 20) return '#8B0000'; // Dark red - critical interference
  if (value > threshold + 10) return '#FF0000'; // Red - high interference
  if (value > threshold) return '#FFA500'; // Orange - medium interference
  if (value > threshold - 10) return '#FFFF00'; // Yellow - low interference
  return '#00FF00'; // Green - no interference
}

// ============================================================================
// Component
// ============================================================================

export const InterferenceHeatmap: React.FC<InterferenceHeatmapProps> = ({
  cells,
  interferenceMatrix,
  threshold = -90,
  width = 800,
  height = 800,
  onCellClick,
  onCellHover
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<HeatmapCell | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    const cellCount = cells.length;
    if (cellCount === 0) return;

    const cellSize = Math.min(width, height) / cellCount;
    const margin = 50;

    // Draw heatmap cells
    for (let row = 0; row < cellCount; row++) {
      for (let col = 0; col < cellCount; col++) {
        const value = interferenceMatrix.matrix[row]?.[col] ?? 0;
        const color = getInterferenceColor(value, threshold);

        const x = margin + col * cellSize;
        const y = margin + row * cellSize;

        ctx.fillStyle = color;
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // Add border for clarity
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
      }
    }

    // Draw axis labels
    ctx.fillStyle = '#000';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';

    for (let i = 0; i < cellCount; i++) {
      const cellId = cells[i]?.cell_id || `Cell ${i}`;
      const shortId = cellId.slice(-6); // Last 6 chars

      // Y-axis (row labels)
      ctx.fillText(shortId, margin - 5, margin + i * cellSize + cellSize / 2);

      // X-axis (column labels)
      ctx.save();
      ctx.translate(margin + i * cellSize + cellSize / 2, margin - 5);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'right';
      ctx.fillText(shortId, 0, 0);
      ctx.restore();
    }

    // Draw title
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Cell-to-Cell Interference Matrix (dBm)', width / 2, 20);

    // Draw legend
    drawLegend(ctx, width - 150, height - 150, threshold);
  }, [cells, interferenceMatrix, threshold, width, height]);

  const drawLegend = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    threshold: number
  ) => {
    const legendHeight = 100;
    const legendWidth = 20;

    ctx.font = '10px Arial';
    ctx.textAlign = 'left';

    const levels = [
      { label: 'Critical', value: threshold + 20, color: '#8B0000' },
      { label: 'High', value: threshold + 10, color: '#FF0000' },
      { label: 'Medium', value: threshold, color: '#FFA500' },
      { label: 'Low', value: threshold - 10, color: '#FFFF00' },
      { label: 'None', value: threshold - 20, color: '#00FF00' }
    ];

    levels.forEach((level, i) => {
      const yPos = y + (i * legendHeight) / levels.length;

      ctx.fillStyle = level.color;
      ctx.fillRect(x, yPos, legendWidth, legendHeight / levels.length);

      ctx.fillStyle = '#000';
      ctx.fillText(`${level.label} (${level.value}dBm)`, x + legendWidth + 5, yPos + 12);
    });
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const margin = 50;
    const cellCount = cells.length;
    const cellSize = Math.min(width, height) / cellCount;

    const col = Math.floor((x - margin) / cellSize);
    const row = Math.floor((y - margin) / cellSize);

    if (row >= 0 && row < cellCount && col >= 0 && col < cellCount) {
      const value = interferenceMatrix.matrix[row]?.[col] ?? 0;
      const cellIdRow = cells[row]?.cell_id || `Cell ${row}`;
      const cellIdCol = cells[col]?.cell_id || `Cell ${col}`;

      const heatmapCell: HeatmapCell = {
        row,
        col,
        value,
        cellIdRow,
        cellIdCol,
        color: getInterferenceColor(value, threshold)
      };

      setHoveredCell(heatmapCell);
      setTooltip({
        x: event.clientX,
        y: event.clientY,
        content: `${cellIdRow} → ${cellIdCol}: ${value.toFixed(1)} dBm`
      });

      if (onCellHover) {
        onCellHover(cellIdRow);
      }
    } else {
      setHoveredCell(null);
      setTooltip(null);
      if (onCellHover) {
        onCellHover(null);
      }
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
    setTooltip(null);
    if (onCellHover) {
      onCellHover(null);
    }
  };

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredCell && onCellClick) {
      onCellClick(hoveredCell.cellIdRow);
    }
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{
          border: '1px solid #ccc',
          cursor: hoveredCell ? 'pointer' : 'default',
          background: '#fff'
        }}
      />

      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x + 10,
            top: tooltip.y + 10,
            background: 'rgba(0, 0, 0, 0.8)',
            color: '#fff',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            whiteSpace: 'nowrap'
          }}
        >
          {tooltip.content}
        </div>
      )}

      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        <strong>Interference Threshold:</strong> {threshold} dBm
        <br />
        <strong>Total Cells:</strong> {cells.length}
        <br />
        <strong>Matrix Dimensions:</strong> {interferenceMatrix.matrix.length} x{' '}
        {interferenceMatrix.matrix[0]?.length || 0}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default InterferenceHeatmap;
