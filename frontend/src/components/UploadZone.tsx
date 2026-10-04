import React, { useState, useRef } from 'react';
import { UploadCloud, FileUp, AlertCircle, CheckCircle2, Loader2, Sparkles } from 'lucide-react';

interface UploadZoneProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  jobStatus: string | null;
  progress: number;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileUpload,
  isLoading,
  jobStatus,
  progress,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    setErrorMsg(null);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndProcess(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMsg(null);
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcess(e.target.files[0]);
    }
  };

  const validateAndProcess = (file: File) => {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.nrrd')) {
      setErrorMsg('Please upload a 3D medical volume with .nrrd format (e.g. 74171 AXIAL.nrrd).');
      return;
    }
    setSelectedFileName(file.name);
    onFileUpload(file);
  };

  // Helper status description
  const getStatusText = () => {
    if (progress < 25) return 'Uploading 3D scan volume to pipeline...';
    if (progress < 45) return 'Queueing job in BullMQ & initializing neural networks...';
    if (progress < 70) return 'Executing Stage 1 Sliding Window (Jaw ROI Segmentation)...';
    if (progress < 90) return 'Running Stage 2 UNet & Marching Cubes 3D surface reconstruction...';
    if (progress < 100) return 'Classifying Misch bone density and assembling 3D meshes...';
    return 'Analysis completed!';
  };

  return (
    <div className="w-full space-y-4">
      <div
        id="dropzone-container"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className={`relative group rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-300 cursor-pointer overflow-hidden ${
          isDragOver
            ? 'border-sky-400 bg-sky-500/10 scale-[1.01]'
            : 'border-slate-700/80 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/60'
        } ${isLoading ? 'pointer-events-none opacity-90' : ''}`}
      >
        {/* Subtle background glow effect */}
        <div className="absolute -top-24 -left-24 w-48 h-48 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

        <input
          ref={fileInputRef}
          type="file"
          id="file-upload-input"
          accept=".nrrd"
          className="hidden"
          onChange={handleFileInputChange}
          disabled={isLoading}
        />

        <div className="relative z-10 flex flex-col items-center">
          <div className="w-16 h-16 mb-4 rounded-2xl bg-gradient-to-tr from-sky-500/20 to-teal-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 group-hover:scale-110 transition-transform">
            {isLoading ? (
              <Loader2 className="w-8 h-8 animate-spin text-sky-400" />
            ) : (
              <UploadCloud className="w-8 h-8 text-sky-400" />
            )}
          </div>

          <h3 className="font-display font-semibold text-lg text-slate-100 mb-1">
            {isLoading ? 'Processing 3D Medical Scan' : 'Upload Dental CBCT Volume'}
          </h3>

          <p className="text-xs text-slate-400 max-w-sm mb-4 leading-relaxed">
            Drag & drop your <span className="font-mono text-sky-300 font-medium">.nrrd</span> volumetric CT scan here,
            or click to browse from your filesystem.
          </p>

          {/* Selected File Badge */}
          {selectedFileName && !errorMsg && (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs text-slate-200 font-mono mb-2">
              <FileUp className="w-3.5 h-3.5 text-sky-400" />
              <span>{selectedFileName}</span>
            </div>
          )}

          {/* Loading Progress Bar */}
          {isLoading && (
            <div className="w-full max-w-md mt-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
                  {getStatusText()}
                </span>
                <span className="font-mono font-bold text-sky-400">{progress}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                <div
                  className="h-full bg-gradient-to-r from-sky-500 to-teal-400 transition-all duration-500 rounded-full"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Error Callout */}
      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
