import React from 'react';
import { Box, Layers, Cpu, Maximize2, Gauge } from 'lucide-react';

interface MetricsCardProps {
  voxelCount: number;
  meanIntensity: number;
  volumeShape?: number[];
  cropBbox?: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    z_min: number;
    z_max: number;
  } | null;
  processingTimeSeconds?: number;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({
  voxelCount,
  meanIntensity,
  volumeShape,
  cropBbox,
  processingTimeSeconds,
}) => {
  // Estimated volume in cubic millimeters assuming standard dental CBCT isometric voxel spacing (0.35mm ~ 0.4mm)
  const approxVoxelSpacingMm = 0.4;
  const approxVolumeMm3 = (voxelCount * Math.pow(approxVoxelSpacingMm, 3)).toFixed(2);

  return (
    <div className="p-6 rounded-2xl bg-slate-900/70 backdrop-blur-xl border border-slate-700/60 shadow-xl space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Gauge className="w-5 h-5 text-cyan-400" />
          <h3 className="font-display font-semibold text-slate-100 text-base">
            Volumetric & Segmentation Metrics
          </h3>
        </div>
        <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono">
          Stage 1 + 2 UNet
        </span>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-2 gap-3">
        {/* Implant Voxels */}
        <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Box className="w-3.5 h-3.5 text-cyan-400" />
            <span>Implant Voxel Count</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-xl font-bold text-slate-100">
              {voxelCount.toLocaleString()}
            </span>
            <span className="text-[11px] text-slate-400">voxels</span>
          </div>
        </div>

        {/* Estimated Volume */}
        <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>Estimated Volume</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-xl font-bold text-slate-100">
              {approxVolumeMm3}
            </span>
            <span className="text-[11px] text-slate-400">mm³</span>
          </div>
        </div>

        {/* Local Bone Bed Intensity */}
        <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Cpu className="w-3.5 h-3.5 text-amber-400" />
            <span>Mean ROI Density</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-xl font-bold text-amber-300">
              {meanIntensity.toFixed(4)}
            </span>
            <span className="text-[11px] text-slate-400">norm.</span>
          </div>
        </div>

        {/* Inference Latency */}
        <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Dual-Stage Latency</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-xl font-bold text-emerald-300">
              {processingTimeSeconds ? `${processingTimeSeconds.toFixed(1)}s` : '~8.5s'}
            </span>
            <span className="text-[11px] text-slate-400">end-to-end</span>
          </div>
        </div>
      </div>

      {/* Scan Dimensions & Crop Bounding Box */}
      <div className="p-3 rounded-xl bg-slate-800/30 border border-slate-700/40 space-y-2 text-xs">
        {volumeShape && volumeShape.length >= 3 && (
          <div className="flex items-center justify-between text-slate-300">
            <span className="text-slate-400">Full Scan Matrix:</span>
            <span className="font-mono font-medium text-slate-200">
              {volumeShape[0]} × {volumeShape[1]} × {volumeShape[2]} (D × H × W)
            </span>
          </div>
        )}

        {cropBbox && (
          <div className="flex items-center justify-between text-slate-300 pt-1 border-t border-slate-800">
            <span className="text-slate-400">Cropped Jaw BBox:</span>
            <span className="font-mono text-[11px] text-slate-300">
              [{cropBbox.x_min}:{cropBbox.x_max}, {cropBbox.y_min}:{cropBbox.y_max}, {cropBbox.z_min}:{cropBbox.z_max}]
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
