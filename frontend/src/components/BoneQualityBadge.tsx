import React from 'react';
import { ShieldAlert, ShieldCheck, Activity, Info, Award } from 'lucide-react';

interface BoneQualityProps {
  classification: string; // 'D1', 'D2', 'D3', 'D4'
  meanIntensity?: number;
}

interface MischInfo {
  title: string;
  subtitle: string;
  anatomicalDescription: string;
  typicalLocation: string;
  hounsfieldUnits: string;
  surgicalGuidance: string;
  stabilityIndex: string;
  colorClass: string;
  badgeClass: string;
  borderClass: string;
  icon: React.ReactNode;
}

const MISCH_SPECS: Record<string, MischInfo> = {
  D1: {
    title: 'D1 Bone Quality',
    subtitle: 'Dense Cortical Bone',
    anatomicalDescription: 'Homogeneous, solid compact cortical bone with minimal vascular channels.',
    typicalLocation: 'Anterior Mandible (chin symphysis)',
    hounsfieldUnits: '> 1250 HU',
    surgicalGuidance: 'High initial primary stability (>45 Ncm). Tap drill required; copious irrigation to prevent thermal osteonecrosis.',
    stabilityIndex: 'Very High (Immediate loading feasible)',
    colorClass: 'text-emerald-400',
    badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    borderClass: 'border-emerald-500/40 shadow-[0_0_20px_rgba(16,185,129,0.15)]',
    icon: <ShieldCheck className="w-6 h-6 text-emerald-400" />,
  },
  D2: {
    title: 'D2 Bone Quality',
    subtitle: 'Thick Porous Cortical & Coarse Trabecular',
    anatomicalDescription: 'Dense porous outer cortex enclosing strong, coarse trabecular bone core.',
    typicalLocation: 'Posterior Mandible & Anterior Maxilla',
    hounsfieldUnits: '850 – 1250 HU',
    surgicalGuidance: 'Optimal bed for osseointegration. Standard osteotomy drilling protocol; high primary stability (35–45 Ncm).',
    stabilityIndex: 'Excellent (Ideal implant prognosis)',
    colorClass: 'text-sky-400',
    badgeClass: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
    borderClass: 'border-sky-500/40 shadow-[0_0_20px_rgba(14,165,233,0.15)]',
    icon: <Award className="w-6 h-6 text-sky-400" />,
  },
  D3: {
    title: 'D3 Bone Quality',
    subtitle: 'Thin Porous Cortical & Fine Trabecular',
    anatomicalDescription: 'Thin, porous cortical shell surrounding permeable, fine trabecular cancellous bone.',
    typicalLocation: 'Posterior Maxilla & Anterior/Posterior Mandible',
    hounsfieldUnits: '350 – 850 HU',
    surgicalGuidance: 'Moderate resistance. Consider slight undersized osteotomy to improve crestal engagement; torque 25–35 Ncm.',
    stabilityIndex: 'Moderate (Progressive loading recommended)',
    colorClass: 'text-amber-400',
    badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    borderClass: 'border-amber-500/40 shadow-[0_0_20px_rgba(245,158,11,0.15)]',
    icon: <Activity className="w-6 h-6 text-amber-400" />,
  },
  D4: {
    title: 'D4 Bone Quality',
    subtitle: 'Fine Trabecular Bone (Low Density)',
    anatomicalDescription: 'Very thin or absent cortical plate enclosing low-density, fine sponge-like trabeculae.',
    typicalLocation: 'Posterior Maxilla (Tuberosity region)',
    hounsfieldUnits: '150 – 350 HU',
    surgicalGuidance: 'Low resistance. Use osseodensification / bone condensation burs without final drills; wider implant diameter suggested.',
    stabilityIndex: 'Low (Delayed loading / extended healing required)',
    colorClass: 'text-rose-400',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    borderClass: 'border-rose-500/40 shadow-[0_0_20px_rgba(244,63,94,0.15)]',
    icon: <ShieldAlert className="w-6 h-6 text-rose-400" />,
  },
};

export const BoneQualityBadge: React.FC<BoneQualityProps> = ({
  classification,
  meanIntensity,
}) => {
  const normKey = (classification || 'D3').toUpperCase().trim();
  const info = MISCH_SPECS[normKey] || MISCH_SPECS['D3'];

  return (
    <div className={`p-6 rounded-2xl bg-slate-900/70 backdrop-blur-xl border ${info.borderClass} transition-all duration-300`}>
      {/* Header with Classification Pill */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-800/80 border border-slate-700">
            {info.icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`text-2xl font-bold font-display ${info.colorClass}`}>
                {normKey}
              </span>
              <span className="text-xs px-2.5 py-0.5 rounded-full font-medium tracking-wide uppercase border border-slate-700 bg-slate-800/60 text-slate-300">
                Misch Scale
              </span>
            </div>
            <h4 className="text-sm font-semibold text-slate-200 mt-0.5">{info.subtitle}</h4>
          </div>
        </div>

        {/* Intensity Meter */}
        {meanIntensity !== undefined && (
          <div className="text-right">
            <span className="text-[11px] uppercase tracking-wider text-slate-400 font-medium block">
              Normalized Intensity
            </span>
            <span className="font-mono text-base font-bold text-slate-100">
              {meanIntensity.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Anatomical Details */}
      <p className="text-xs text-slate-300 leading-relaxed mb-4 bg-slate-800/40 p-3 rounded-xl border border-slate-700/50">
        {info.anatomicalDescription}
      </p>

      {/* Grid of Key Clinical Indicators */}
      <div className="grid grid-cols-2 gap-3 text-xs mb-4">
        <div className="p-3 rounded-xl bg-slate-800/50 border border-slate-700/60">
          <span className="text-slate-400 text-[11px] block font-medium">Predominant Site</span>
          <span className="text-slate-200 font-semibold">{info.typicalLocation}</span>
        </div>
        <div className="p-3 rounded-xl bg-slate-800/50 border border-slate-700/60">
          <span className="text-slate-400 text-[11px] block font-medium">Estimated Range</span>
          <span className="text-slate-200 font-mono font-semibold">{info.hounsfieldUnits}</span>
        </div>
      </div>

      {/* Surgical Guidance Callout */}
      <div className="p-3.5 rounded-xl bg-slate-800/70 border border-slate-700 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
        <div className="text-xs">
          <span className="font-semibold text-sky-300 block mb-0.5">Surgical Guidance & Protocol:</span>
          <span className="text-slate-300 leading-normal">{info.surgicalGuidance}</span>
        </div>
      </div>
    </div>
  );
};
