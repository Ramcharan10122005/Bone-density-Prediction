import React, { useState, useEffect, useRef } from 'react';
import { UploadZone } from './components/UploadZone';
import { Viewer3D, MeshData } from './components/Viewer3D';
import { BoneQualityBadge } from './components/BoneQualityBadge';
import { MetricsCard } from './components/MetricsCard';
import {
  Activity,
  Layers,
  Sparkles,
  Server,
  Zap,
  Info,
  CheckCircle2,
  AlertTriangle,
  Play,
} from 'lucide-react';

interface AnalysisResult {
  bone_quality: string;
  implant_volume_voxels: number;
  mean_intensity: number;
  roi_mesh: MeshData;
  implant_mesh: MeshData;
  volume_shape?: number[];
  crop_bbox?: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    z_min: number;
    z_max: number;
  };
}

export const App: React.FC = () => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [backendHealth, setBackendHealth] = useState<'checking' | 'healthy' | 'offline'>('checking');
  const [scanStartTime, setScanStartTime] = useState<number | null>(null);
  const [inferenceDuration, setInferenceDuration] = useState<number | undefined>(undefined);

  const pollIntervalRef = useRef<number | null>(null);

  // Check backend & ML service health on load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'ok') {
            setBackendHealth('healthy');
            return;
          }
        }
        setBackendHealth('offline');
      } catch (err) {
        setBackendHealth('offline');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Poll job status
  useEffect(() => {
    if (!jobId || jobStatus === 'completed' || jobStatus === 'failed') {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    const pollJob = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;

        const data = await res.json();
        setJobStatus(data.status);

        // Update progress smoothly based on BullMQ progress or state
        if (data.status === 'waiting') {
          setProgress(20);
        } else if (data.status === 'active') {
          setProgress((prev) => Math.min(Math.max(prev + 5, 45), 90));
        } else if (data.status === 'completed') {
          setProgress(100);
          setIsLoading(false);
          setResult(data.result);

          if (scanStartTime) {
            setInferenceDuration((Date.now() - scanStartTime) / 1000);
          }
        } else if (data.status === 'failed') {
          setIsLoading(false);
          setError(data.error || 'Inference execution failed on ML service.');
        }
      } catch (err: any) {
        console.error('Job polling error:', err);
      }
    };

    pollIntervalRef.current = window.setInterval(pollJob, 1000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [jobId, jobStatus, scanStartTime]);

  // Handle User File Upload
  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setJobStatus('uploading');
    setProgress(15);
    setScanStartTime(Date.now());

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `Upload failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      setJobId(data.jobId);
      setJobStatus('queued');
      setProgress(30);
    } catch (err: any) {
      setIsLoading(false);
      setError(err.message || 'Failed to upload scan file to server.');
    }
  };

  // Quick Load Demo Button (re-queries last test job or shows pre-loaded sample)
  const handleLoadDemo = async () => {
    setIsLoading(true);
    setError(null);
    setProgress(35);
    setScanStartTime(Date.now());

    try {
      // Check clean filtered demo job or fallback
      let res = await fetch('/api/jobs/job-d6747eef');
      if (!res.ok) {
        res = await fetch('/api/jobs/job-f7c26105');
      }
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'completed' && data.result) {
          setProgress(100);
          setIsLoading(false);
          setJobStatus('completed');
          setResult(data.result);
          setInferenceDuration(8.4);
          return;
        }
      }
      throw new Error('No cached demo job found. Please drop your 74171 AXIAL.nrrd file.');
    } catch (err: any) {
      setIsLoading(false);
      setError(err.message);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation Bar */}
      <header className="header-bar">
        <div className="brand-section">
          <div className="brand-icon-box">
            <Zap className="w-6 h-6" />
          </div>
          <div>
            <h1 className="brand-title">OSSEOVUE</h1>
            <p className="brand-subtitle">Autonomous 3D Implant Segmentation & Misch Bone Quality</p>
          </div>
        </div>

        {/* Status Indicators & Demo Button */}
        <div className="flex items-center gap-3">
          {/* Quick Demo Button */}
          <button
            id="btn-load-demo"
            onClick={handleLoadDemo}
            disabled={isLoading}
            className="btn-secondary"
            title="Load verified sample patient scan (74171 AXIAL)"
          >
            <Play className="w-3.5 h-3.5 text-sky-400 fill-sky-400/30" />
            <span>Load Demo Scan</span>
          </button>

          {/* Service Health Pill */}
          <div className="status-pill">
            <div className={`status-dot ${backendHealth === 'healthy' ? '' : 'bg-rose-500 shadow-rose-500'}`} />
            <span>{backendHealth === 'healthy' ? 'Cluster Active' : 'Connecting...'}</span>
          </div>
        </div>
      </header>

      {/* Main Two-Column Layout */}
      <main className="dashboard-grid">
        {/* Left Column: Upload & Clinical Diagnostics */}
        <div className="left-column">
          {/* File Upload Zone */}
          <section className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-slate-100 font-display flex items-center gap-2">
                <Layers className="w-4 h-4 text-sky-400" />
                <span>Input CBCT Scan</span>
              </h2>
              <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                .nrrd format
              </span>
            </div>

            <UploadZone
              onFileUpload={handleFileUpload}
              isLoading={isLoading}
              jobStatus={jobStatus}
              progress={progress}
            />

            {error && (
              <div className="mt-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
                <div>
                  <span className="font-semibold block mb-0.5">Execution Error:</span>
                  <span>{error}</span>
                </div>
              </div>
            )}
          </section>

          {/* Bone Quality Diagnostic Badge */}
          {result && (
            <section>
              <BoneQualityBadge
                classification={result.bone_quality}
                meanIntensity={result.mean_intensity}
              />
            </section>
          )}

          {/* Volumetric Metrics Card */}
          {result && (
            <section>
              <MetricsCard
                voxelCount={result.implant_volume_voxels}
                meanIntensity={result.mean_intensity}
                volumeShape={result.volume_shape}
                cropBbox={result.crop_bbox}
                processingTimeSeconds={inferenceDuration}
              />
            </section>
          )}

          {/* Pipeline Architectural Card */}
          <section className="glass-card p-5 text-xs space-y-3">
            <h3 className="font-semibold text-slate-200 flex items-center gap-2">
              <Server className="w-3.5 h-3.5 text-cyan-400" />
              <span>Two-Stage Neural Architecture</span>
            </h3>
            <div className="space-y-2 text-slate-400 leading-relaxed">
              <div className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-sky-400 mt-1.5 shrink-0" />
                <span>
                  <strong className="text-slate-200">Stage 1:</strong> Sliding window UNet (192×144×128) segments jaw region of interest (label 6).
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 mt-1.5 shrink-0" />
                <span>
                  <strong className="text-slate-200">Stage 2:</strong> Cropped 96³ cubic subvolume localizes implant fixture (label 7) & extracts bone density.
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 shrink-0" />
                <span>
                  <strong className="text-slate-200">Marching Cubes:</strong> Vectorized isosurface extraction with affine translation alignment.
                </span>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Interactive 3D Medical Scene */}
        <div className="right-column">
          <section className="glass-card p-4 flex flex-col gap-4">
            <div className="flex items-center justify-between px-2 pt-1">
              <div>
                <h2 className="text-lg font-bold text-slate-100 font-display flex items-center gap-2">
                  <span>3D Anatomical Reconstruction</span>
                  {result && (
                    <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium">
                      Aligned Coordinates
                    </span>
                  )}
                </h2>
                <p className="text-xs text-slate-400">
                  Volumetric isosurface rendering of jaw boundary and planned implant fixture
                </p>
              </div>

              {result && (
                <div className="text-right hidden sm:block">
                  <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                    Classification
                  </span>
                  <span className="font-display font-bold text-base text-cyan-400">
                    Misch {result.bone_quality}
                  </span>
                </div>
              )}
            </div>

            {/* 3D Scene Container */}
            <Viewer3D
              implantMesh={result?.implant_mesh || null}
              roiMesh={result?.roi_mesh || null}
              boneQualityClass={result?.bone_quality}
            />

            {/* Instructions Guide */}
            <div className="flex items-center justify-between text-[11px] text-slate-400 px-2 py-1">
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-slate-500" />
                Left-click to Rotate • Right-click to Pan • Scroll to Zoom
              </span>
              <span className="font-mono text-slate-500">
                GPU Accelerated WebGL 2.0
              </span>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
