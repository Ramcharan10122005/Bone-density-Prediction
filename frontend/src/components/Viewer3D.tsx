import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCw, Eye, EyeOff, Camera, Box, Layers, RefreshCw, Crosshair } from 'lucide-react';

export interface MeshData {
  vertices: number[][];
  faces: number[][];
}

interface Viewer3DProps {
  implantMesh: MeshData | null;
  roiMesh: MeshData | null;
  boneQualityClass?: string;
}

export const Viewer3D: React.FC<Viewer3DProps> = ({
  implantMesh,
  roiMesh,
  boneQualityClass,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);

  // Mesh refs
  const implantMeshRef = useRef<THREE.Mesh | null>(null);
  const roiMeshRef = useRef<THREE.Mesh | null>(null);

  // View state
  const [roiMode, setRoiMode] = useState<'solid' | 'wireframe' | 'hidden'>('solid');
  const [showImplant, setShowImplant] = useState<boolean>(true);
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [cameraPosition, setCameraPosition] = useState<string>('');

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Clear any previous canvas
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 560;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x070b14);

    // Subtle dark grid
    const gridHelper = new THREE.GridHelper(400, 40, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -60;
    scene.add(gridHelper);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 4000);
    camera.position.set(160, 140, 200);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;
    container.appendChild(renderer.domElement);

    // 4. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2000;
    controls.minDistance = 10;
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 2.2); // Cyan surgical key light
    dirLight1.position.set(200, 300, 200);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa855f7, 1.4); // Violet accent light
    dirLight2.position.set(-200, -100, -200);
    scene.add(dirLight2);

    const dirLight3 = new THREE.DirectionalLight(0xffffff, 1.6); // Fill light
    dirLight3.position.set(0, 250, -250);
    scene.add(dirLight3);

    // Point light that illuminates the center cavity
    const pointLight = new THREE.PointLight(0x38bdf8, 2, 300);
    pointLight.position.set(0, 50, 0);
    scene.add(pointLight);

    // 6. Build Geometries from meshes
    const group = new THREE.Group();
    groupRef.current = group;
    scene.add(group);

    const createGeometry = (meshData: MeshData) => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(meshData.vertices.flat());
      const indices = new Uint32Array(meshData.faces.flat());

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeVertexNormals();
      return geometry;
    };

    // Build ROI Mesh (Jaw Anatomy)
    if (roiMesh && roiMesh.vertices.length > 0) {
      const roiGeo = createGeometry(roiMesh);
      const roiMaterial = new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        transparent: true,
        opacity: 0.32,
        roughness: 0.5,
        metalness: 0.15,
        side: THREE.DoubleSide,
        depthWrite: false,
      });

      const roiMeshObj = new THREE.Mesh(roiGeo, roiMaterial);
      roiMeshObj.name = 'roi_mesh';
      roiMeshObj.renderOrder = 1;
      roiMeshRef.current = roiMeshObj;
      group.add(roiMeshObj);
    }

    // Build Implant Mesh (Segmented Titanium Fixture / Bed)
    if (implantMesh && implantMesh.vertices.length > 0) {
      const implantGeo = createGeometry(implantMesh);

      // High-visibility glowing surgical titanium cyan
      const implantMaterial = new THREE.MeshStandardMaterial({
        color: 0x00f0ff,
        emissive: 0x0284c7,
        emissiveIntensity: 0.6,
        metalness: 0.95,
        roughness: 0.15,
        side: THREE.DoubleSide,
      });

      const implantMeshObj = new THREE.Mesh(implantGeo, implantMaterial);
      implantMeshObj.name = 'implant_mesh';
      implantMeshObj.renderOrder = 2;
      implantMeshRef.current = implantMeshObj;
      group.add(implantMeshObj);
    }

    // Center the entire group
    const box = new THREE.Box3().setFromObject(group);
    const center = new THREE.Vector3();
    box.getCenter(center);
    group.position.set(-center.x, -center.y, -center.z);

    // Dynamic camera framing based on bounding size
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 60);
    const camDist = maxDim * 1.5;

    camera.position.set(camDist * 0.75, camDist * 0.55, camDist * 0.85);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();

    // 7. Animation Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (autoRotate && groupRef.current) {
        groupRef.current.rotation.y += 0.008;
      }

      controls.update();
      renderer.render(scene, camera);

      setCameraPosition(
        `X: ${camera.position.x.toFixed(0)} Y: ${camera.position.y.toFixed(0)} Z: ${camera.position.z.toFixed(0)}`
      );
    };

    animate();

    // 8. Resize Handler
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight || 560;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [implantMesh, roiMesh, autoRotate]);

  // Handle ROI Mode change (solid, wireframe, hidden)
  useEffect(() => {
    if (!roiMeshRef.current) return;
    const mesh = roiMeshRef.current;
    const mat = mesh.material as THREE.MeshStandardMaterial;

    if (roiMode === 'hidden') {
      mesh.visible = false;
    } else if (roiMode === 'wireframe') {
      mesh.visible = true;
      mat.wireframe = true;
      mat.opacity = 0.55;
      mat.needsUpdate = true;
    } else {
      mesh.visible = true;
      mat.wireframe = false;
      mat.opacity = 0.32;
      mat.needsUpdate = true;
    }
  }, [roiMode]);

  // Handle Implant visibility toggle
  useEffect(() => {
    if (!implantMeshRef.current) return;
    implantMeshRef.current.visible = showImplant;
  }, [showImplant]);

  // Reset Camera View
  const handleResetCamera = () => {
    if (!cameraRef.current || !controlsRef.current || !groupRef.current) return;
    const box = new THREE.Box3().setFromObject(groupRef.current);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 60);
    const camDist = maxDim * 1.5;

    cameraRef.current.position.set(camDist * 0.75, camDist * 0.55, camDist * 0.85);
    cameraRef.current.lookAt(0, 0, 0);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  // Focus on Implant
  const handleFocusImplant = () => {
    if (!implantMeshRef.current || !cameraRef.current || !controlsRef.current || !groupRef.current) return;
    const implantGeo = implantMeshRef.current.geometry;
    implantGeo.computeBoundingBox();
    if (!implantGeo.boundingBox) return;

    const center = new THREE.Vector3();
    implantGeo.boundingBox.getCenter(center);
    // Add group offset
    center.add(groupRef.current.position);

    controlsRef.current.target.copy(center);
    cameraRef.current.position.set(center.x + 40, center.y + 30, center.z + 45);
    cameraRef.current.lookAt(center);
    controlsRef.current.update();
  };

  // Capture Snapshot
  const handleSnapshot = () => {
    const canvas = mountRef.current?.querySelector('canvas');
    if (!canvas) return;
    const url = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `osseovue-plan-${Date.now()}.png`;
    link.href = url;
    link.click();
  };

  const totalVertices =
    (implantMesh?.vertices?.length || 0) + (roiMesh?.vertices?.length || 0);
  const totalFaces =
    (implantMesh?.faces?.length || 0) + (roiMesh?.faces?.length || 0);

  return (
    <div className="relative w-full h-[600px] rounded-2xl overflow-hidden border border-slate-700/60 shadow-2xl bg-[#070b14] flex flex-col">
      {/* 3D Canvas Mounting Point */}
      <div
        ref={mountRef}
        id="three-canvas-container"
        className="w-full flex-1 cursor-grab active:cursor-grabbing relative"
      />

      {/* Floating View Controls Top Bar */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
        {/* Left Badge: 3D Scene Status */}
        <div className="pointer-events-auto flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/90 backdrop-blur-md border border-slate-700 text-xs shadow-lg">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-semibold text-slate-200">Interactive 3D Engine</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400 font-mono text-[11px]">{cameraPosition || 'Active'}</span>
        </div>

        {/* Right Tools: Camera & Snapshot */}
        <div className="pointer-events-auto flex items-center gap-2">
          {implantMesh && implantMesh.vertices.length > 0 && (
            <button
              id="btn-focus-implant"
              onClick={handleFocusImplant}
              className="px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 bg-slate-900/90 text-cyan-300 border border-cyan-500/40 hover:bg-slate-800 backdrop-blur-md transition-all shadow-sm"
              title="Focus Camera Directly on Implant"
            >
              <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
              <span>Focus Implant</span>
            </button>
          )}

          <button
            id="btn-auto-rotate"
            onClick={() => setAutoRotate(!autoRotate)}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 backdrop-blur-md border transition-all ${
              autoRotate
                ? 'bg-sky-500/20 text-sky-300 border-sky-500/50'
                : 'bg-slate-900/90 text-slate-300 border-slate-700 hover:bg-slate-800'
            }`}
            title="Toggle Auto-Rotation"
          >
            <RotateCw className={`w-3.5 h-3.5 ${autoRotate ? 'animate-spin' : ''}`} />
            <span>Turntable</span>
          </button>

          <button
            id="btn-reset-view"
            onClick={handleResetCamera}
            className="px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 bg-slate-900/90 text-slate-300 border border-slate-700 hover:bg-slate-800 backdrop-blur-md transition-all"
            title="Reset Perspective View"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          <button
            id="btn-snapshot"
            onClick={handleSnapshot}
            className="px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 bg-slate-900/90 text-slate-300 border border-slate-700 hover:bg-slate-800 backdrop-blur-md transition-all"
            title="Export High-Res Snapshot"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Snapshot</span>
          </button>
        </div>
      </div>

      {/* Floating Bottom Control Bar */}
      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none">
        {/* Layer Controls: ROI & Implant Toggles */}
        <div className="pointer-events-auto flex items-center gap-2 bg-slate-900/95 backdrop-blur-md border border-slate-700 p-1.5 rounded-2xl shadow-2xl">
          {/* Implant Visibility Toggle */}
          <button
            id="toggle-implant-btn"
            onClick={() => setShowImplant(!showImplant)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all ${
              showImplant
                ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {showImplant ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Implant Fixture</span>
          </button>

          <div className="w-[1px] h-4 bg-slate-700" />

          {/* ROI Display Modes */}
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider px-2 font-medium">Jaw ROI:</span>
            <button
              id="roi-solid-btn"
              onClick={() => setRoiMode('solid')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                roiMode === 'solid'
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Ghost
            </button>
            <button
              id="roi-wire-btn"
              onClick={() => setRoiMode('wireframe')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                roiMode === 'wireframe'
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Wireframe
            </button>
            <button
              id="roi-hide-btn"
              onClick={() => setRoiMode('hidden')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                roiMode === 'hidden'
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Hide
            </button>
          </div>
        </div>

        {/* Geometry Stats Pill */}
        <div className="pointer-events-auto hidden md:flex items-center gap-3 px-3 py-1.5 rounded-xl bg-slate-900/90 backdrop-blur-md border border-slate-700 text-[11px] text-slate-300 font-mono shadow-lg">
          <div className="flex items-center gap-1">
            <Layers className="w-3 h-3 text-cyan-400" />
            <span>Verts: {totalVertices.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-1">
            <Box className="w-3 h-3 text-indigo-400" />
            <span>Faces: {totalFaces.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
