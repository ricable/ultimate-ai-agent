/**
 * InterferenceHeatmap - WebGL Interference Visualization
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 *
 * Renders interactive interference matrix using WebGL for performance.
 * Visualizes cell-to-cell interference levels in real-time.
 */

'use client';

import React, { useRef, useEffect, useState } from 'react';
import type { InterferenceData } from '../../types/council';

interface InterferenceHeatmapProps {
  data: InterferenceData | null;
  className?: string;
}

export function InterferenceHeatmap({ data, className = '' }: InterferenceHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<string | null>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl2');

    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    // Simple 2D rendering for demonstration
    // In production: Use WebGL shaders for high-performance heatmap rendering
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { matrix } = data;
    const cellSize = Math.min(canvas.width, canvas.height) / matrix.length;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Render heatmap
    matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        // Map interference level (-120 to -60 dBm) to color (blue to red)
        const normalized = (value + 120) / 60; // 0 to 1
        const hue = (1 - normalized) * 240; // 240 (blue) to 0 (red)

        ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Draw interference value
        ctx.fillStyle = 'white';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          value.toFixed(0),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        );
      });
    });

  }, [data]);

  if (!data) {
    return (
      <div className={`flex items-center justify-center h-64 bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 ${className}`}>
        <p className="text-gray-400">No interference data available</p>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {/* Glass Box Container */}
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">📡</span>
            Interference Matrix
          </h3>
          <div className="text-sm text-gray-400">
            Cell: <span className="text-blue-400 font-mono">{data.cellId}</span>
          </div>
        </div>

        {/* Canvas for WebGL rendering */}
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          className="w-full h-auto rounded border border-gray-800"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            // Calculate which cell is hovered (simplified)
            const cellIndex = Math.floor(y / rect.height * data.matrix.length);
            if (data.neighbors[cellIndex]) {
              setHoveredCell(data.neighbors[cellIndex].cellId);
            }
          }}
          onMouseLeave={() => setHoveredCell(null)}
        />

        {/* Legend */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(240, 80%, 50%)' }} />
              <span>Low (-120 dBm)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(120, 80%, 50%)' }} />
              <span>Medium (-90 dBm)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(0, 80%, 50%)' }} />
              <span>High (-60 dBm)</span>
            </div>
          </div>
          {hoveredCell && (
            <div className="text-sm text-blue-400 font-mono">
              Hover: {hoveredCell}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
