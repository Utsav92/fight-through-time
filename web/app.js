/**
 * FIGHT THROUGH TIME — INTERACTIVE 4D MOTION SCULPTURE
 * Three.js WebGL 4D Motion Visualization Engine
 */

(function () {
  "use strict";

  // --- Configuration & State ---
  const state = {
    currentFrameIndex: 60, // starts at frame index 60 (frame 420: peak combat)
    historyCount: 7,
    poseDelaySec: 0.40,
    poseDelayFrames: 10,
    timeDepth: 1.5,
    timeWindowSec: 2.4,
    visualMode: "HYBRID", // 'BODY', 'GHOST', 'POINT_CLOUD', 'TRAJECTORY', 'HYBRID'
    interactionMode: "CAMERA", // 'CAMERA', 'TIME', '4D_NAV'
    showBodies: true,
    showPoints: true,
    showTrajectories: true,
    showImpacts: true,
    isPlaying: false,
    isReverse: false,
    isFrozen: false,
    playbackSpeed: 1.0,
    colorFighterA: "#00e5ff",
    colorFighterB: "#ff9100",
    telemetry: null,
    totalFrames: 0,
    startFrame: 360,
    endFrame: 530,
    fps: 25.028
  };

  const OPACITY_LADDER = [1.0, 0.82, 0.68, 0.55, 0.43, 0.32, 0.22, 0.15, 0.10, 0.08];

  // --- Three.js Globals ---
  let scene, camera, renderer, controls;
  let groundGrid, timeAxis3DGroup;
  let bodyLayersGroup, pointCloudGroup, trajectoryGroup, impactGroup;
  let atlasTextures = {
    combined: [],
    fighterA: [],
    fighterB: []
  };

  // DOM Elements
  const container = document.getElementById("canvas-container");
  const valFrame = document.getElementById("val-frame");
  const valTime = document.getElementById("val-time");
  const valCursor = document.getElementById("val-cursor");
  const valAction = document.getElementById("val-action");

  const sliderHistory = document.getElementById("slider-history");
  const lblHistory = document.getElementById("lbl-history");
  const sliderDelay = document.getElementById("slider-delay");
  const lblDelay = document.getElementById("lbl-delay");
  const sliderDepth = document.getElementById("slider-depth");
  const lblDepth = document.getElementById("lbl-depth");
  const sliderWindow = document.getElementById("slider-window");
  const lblWindow = document.getElementById("lbl-window");

  const axisRail = document.getElementById("axis-rail");
  const axisProgress = document.getElementById("axis-progress");
  const axisCursor = document.getElementById("axis-cursor");
  const historyMarkersLayer = document.getElementById("history-markers-layer");
  const timelineAnnotations = document.getElementById("timeline-annotations");
  const actionJumpContainer = document.getElementById("action-jump-container");

  const btnPlay = document.getElementById("btn-play");
  const btnReverse = document.getElementById("btn-reverse");
  const btnFreeze = document.getElementById("btn-freeze");
  const btnPrevFrame = document.getElementById("btn-prev-frame");
  const btnNextFrame = document.getElementById("btn-next-frame");
  const btnPrev10 = document.getElementById("btn-prev-10");
  const btnNext10 = document.getElementById("btn-next-10");
  const btnFirst = document.getElementById("btn-first");
  const btnLast = document.getElementById("btn-last");

  const btnModeCamera = document.getElementById("btn-mode-camera");
  const btnModeTime = document.getElementById("btn-mode-time");
  const btnMode4D = document.getElementById("btn-mode-4d");
  const modalHelp = document.getElementById("modal-help");
  const btnHelp = document.getElementById("btn-help");
  const btnCloseHelp = document.getElementById("btn-close-help");

  // --- Initialization ---
  async function init() {
    initThree();
    await loadTelemetry();
    await loadAtlasTextures();
    initUI();
    initEvents();
    renderTemporalSculpture();
    animate();
  }

  // --- Three.js Scene Setup ---
  function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050608);
    scene.fog = new THREE.FogExp2(0x050608, 0.04);

    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    setCameraPreset(45); // default 45° motion sculpture reveal

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0.9, -4.5);
    controls.maxPolarAngle = Math.PI * 0.52; // prevent going beneath ground
    controls.minDistance = 1.0;
    controls.maxDistance = 30.0;

    // Subtle Ground Grid
    const gridHelper = new THREE.GridHelper(30, 60, 0x00e5ff, 0x1e293b);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Subtle ambient & directional lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // Groups for 4D elements
    bodyLayersGroup = new THREE.Group();
    pointCloudGroup = new THREE.Group();
    trajectoryGroup = new THREE.Group();
    impactGroup = new THREE.Group();
    timeAxis3DGroup = new THREE.Group();

    scene.add(bodyLayersGroup);
    scene.add(pointCloudGroup);
    scene.add(trajectoryGroup);
    scene.add(impactGroup);
    scene.add(timeAxis3DGroup);

    build3DTimeAxis();
    window.addEventListener("resize", onWindowResize);
  }

  // --- 3D Time Axis Visual in Scene (Section 12) ---
  function build3DTimeAxis() {
    timeAxis3DGroup.clear();
    // A subtle glowing arrow along the floor extending into depth (Z axis)
    const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.4 });
    const points = [
      new THREE.Vector3(2.5, 0.02, 0.5),
      new THREE.Vector3(2.5, 0.02, -15.0)
    ];
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const timeLine = new THREE.Line(geom, lineMat);
    timeAxis3DGroup.add(timeLine);

    // Ticks along the 3D axis
    for (let z = 0; z >= -15; z -= 1.5) {
      const tickPts = [
        new THREE.Vector3(2.3, 0.02, z),
        new THREE.Vector3(2.7, 0.02, z)
      ];
      const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPts);
      const tickLine = new THREE.Line(tickGeom, lineMat);
      timeAxis3DGroup.add(tickLine);
    }
  }

  // --- Load Telemetry Data ---
  async function loadTelemetry() {
    try {
      const res = await fetch("data/fight_telemetry_compact.json");
      state.telemetry = await res.json();
      state.totalFrames = state.telemetry.totalFrames || state.telemetry.frames.length;
      state.startFrame = state.telemetry.start_frame || 360;
      state.endFrame = state.telemetry.end_frame || 530;
      state.fps = state.telemetry.fps || 25.028;
      console.log(`Telemetry loaded: ${state.totalFrames} frames.`);
    } catch (err) {
      console.warn("Failed to load compact telemetry, trying root fallback:", err);
      const res = await fetch("../data/fight_telemetry_compact.json");
      state.telemetry = await res.json();
      state.totalFrames = state.telemetry.frames.length;
    }
  }

  // --- Load Sprite Atlases for 60 FPS GPU Rendering ---
  async function loadAtlasTextures() {
    const loader = new THREE.TextureLoader();
    const loadTex = (url) => new Promise((resolve) => {
      loader.load(url, (tex) => {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        resolve(tex);
      }, undefined, () => resolve(null));
    });

    const [c0, c1, a0, a1, b0, b1] = await Promise.all([
      loadTex("data/sprites/atlas_combined_0.png"),
      loadTex("data/sprites/atlas_combined_1.png"),
      loadTex("data/sprites/atlas_fighter_a_0.png"),
      loadTex("data/sprites/atlas_fighter_a_1.png"),
      loadTex("data/sprites/atlas_fighter_b_0.png"),
      loadTex("data/sprites/atlas_fighter_b_1.png")
    ]);

    atlasTextures.combined = [c0, c1].filter(Boolean);
    atlasTextures.fighterA = [a0, a1].filter(Boolean);
    atlasTextures.fighterB = [b0, b1].filter(Boolean);
  }

  // --- Camera Presets (Section 8) ---
  function setCameraPreset(angleDeg) {
    const rad = (angleDeg * Math.PI) / 180;
    const distance = 8.5;
    const targetY = 0.9;
    const targetZ = -(state.historyCount * state.timeDepth * 0.45);

    // Compute camera position rotated around Y
    const x = Math.sin(rad) * distance;
    const z = Math.cos(rad) * distance + targetZ;
    const y = 1.5 + (angleDeg / 90) * 1.5;

    camera.position.set(x, y, z);
    if (controls) {
      controls.target.set(0, targetY, targetZ);
      controls.update();
    }

    document.querySelectorAll(".cam-btn").forEach((b) => {
      b.classList.toggle("active", parseFloat(b.dataset.angle) === angleDeg);
    });
  }

  // --- UI Setup & Action Markers ---
  function initUI() {
    // Populate Key Action Jump Buttons & Timeline Markers
    if (state.telemetry && state.telemetry.key_actions) {
      actionJumpContainer.innerHTML = "";
      timelineAnnotations.innerHTML = "";

      state.telemetry.key_actions.forEach((act) => {
        // Right side jump button
        const btn = document.createElement("button");
        btn.className = "action-jump-btn";
        btn.innerHTML = `
          <span>F${act.frame} ${act.label}</span>
          <span class="action-tag tag-${act.type.toLowerCase()}">${act.type}</span>
        `;
        btn.onclick = () => jumpToFrame(act.frame);
        actionJumpContainer.appendChild(btn);

        // Marker on the timeline bar
        const frameIdx = act.frame - state.startFrame;
        const norm = Math.max(0, Math.min(1, frameIdx / (state.totalFrames - 1)));
        const marker = document.createElement("div");
        marker.className = `anno-marker tag-${act.type.toLowerCase()}`;
        marker.style.left = `${norm * 100}%`;
        marker.textContent = act.type;
        marker.title = `${act.label} (F${act.frame})`;
        marker.onclick = () => jumpToFrame(act.frame);
        timelineAnnotations.appendChild(marker);
      });
    }

    updateTelemetryHUD();
    updateTimelineUI();
  }

  // --- Event Listeners ---
  function initEvents() {
    // Playback Controls
    btnPlay.onclick = togglePlay;
    btnReverse.onclick = toggleReverse;
    btnFreeze.onclick = toggleFreeze;

    btnPrevFrame.onclick = () => stepFrame(-1);
    btnNextFrame.onclick = () => stepFrame(1);
    btnPrev10.onclick = () => stepFrame(-10);
    btnNext10.onclick = () => stepFrame(10);
    btnFirst.onclick = () => jumpToFrameIndex(0);
    btnLast.onclick = () => jumpToFrameIndex(state.totalFrames - 1);

    // Speed Selector
    document.querySelectorAll(".spd-btn").forEach((b) => {
      b.onclick = () => {
        document.querySelectorAll(".spd-btn").forEach((el) => el.classList.remove("active"));
        b.classList.add("active");
        state.playbackSpeed = parseFloat(b.dataset.speed);
      };
    });

    // Sliders
    sliderHistory.oninput = (e) => {
      state.historyCount = parseInt(e.target.value);
      lblHistory.textContent = state.historyCount;
      state.timeWindowSec = (state.historyCount - 1) * state.poseDelaySec;
      sliderWindow.value = state.timeWindowSec.toFixed(1);
      lblWindow.textContent = `${state.timeWindowSec.toFixed(2)}s`;
      renderTemporalSculpture();
    };

    sliderDelay.oninput = (e) => {
      state.poseDelaySec = parseFloat(e.target.value);
      state.poseDelayFrames = Math.max(1, Math.round(state.poseDelaySec * state.fps));
      lblDelay.textContent = `${state.poseDelaySec.toFixed(2)}s (${state.poseDelayFrames}f)`;
      state.timeWindowSec = (state.historyCount - 1) * state.poseDelaySec;
      sliderWindow.value = state.timeWindowSec.toFixed(1);
      lblWindow.textContent = `${state.timeWindowSec.toFixed(2)}s`;
      renderTemporalSculpture();
    };

    sliderDepth.oninput = (e) => {
      state.timeDepth = parseFloat(e.target.value);
      lblDepth.textContent = state.timeDepth.toFixed(2);
      renderTemporalSculpture();
      build3DTimeAxis();
    };

    sliderWindow.oninput = (e) => {
      state.timeWindowSec = parseFloat(e.target.value);
      lblWindow.textContent = `${state.timeWindowSec.toFixed(2)}s`;
      state.poseDelaySec = state.timeWindowSec / Math.max(1, state.historyCount - 1);
      state.poseDelayFrames = Math.max(1, Math.round(state.poseDelaySec * state.fps));
      sliderDelay.value = state.poseDelaySec.toFixed(2);
      lblDelay.textContent = `${state.poseDelaySec.toFixed(2)}s (${state.poseDelayFrames}f)`;
      renderTemporalSculpture();
    };

    // Visual Modes (Section 33)
    document.querySelectorAll(".vis-btn").forEach((b) => {
      b.onclick = () => {
        document.querySelectorAll(".vis-btn").forEach((el) => el.classList.remove("active"));
        b.classList.add("active");
        state.visualMode = b.dataset.mode;
        renderTemporalSculpture();
      };
    });

    // Layer Toggles
    document.getElementById("chk-bodies").onchange = (e) => {
      state.showBodies = e.target.checked;
      bodyLayersGroup.visible = state.showBodies;
    };
    document.getElementById("chk-points").onchange = (e) => {
      state.showPoints = e.target.checked;
      pointCloudGroup.visible = state.showPoints;
    };
    document.getElementById("chk-trajectories").onchange = (e) => {
      state.showTrajectories = e.target.checked;
      trajectoryGroup.visible = state.showTrajectories;
    };
    document.getElementById("chk-impacts").onchange = (e) => {
      state.showImpacts = e.target.checked;
      impactGroup.visible = state.showImpacts;
    };

    // Color Pickers
    document.getElementById("col-fighter-a").oninput = (e) => {
      state.colorFighterA = e.target.value;
      renderTemporalSculpture();
    };
    document.getElementById("col-fighter-b").oninput = (e) => {
      state.colorFighterB = e.target.value;
      renderTemporalSculpture();
    };

    // Camera Preset Buttons
    document.querySelectorAll(".cam-btn").forEach((b) => {
      b.onclick = () => setCameraPreset(parseFloat(b.dataset.angle));
    });
    document.getElementById("btn-cam-reset").onclick = () => setCameraPreset(45);

    // Navigation Mode Buttons
    btnModeCamera.onclick = () => setInteractionMode("CAMERA");
    btnModeTime.onclick = () => setInteractionMode("TIME");
    btnMode4D.onclick = () => setInteractionMode("4D_NAV");

    // Help Modal
    btnHelp.onclick = () => modalHelp.classList.remove("hidden");
    btnCloseHelp.onclick = () => modalHelp.classList.add("hidden");
    modalHelp.onclick = (e) => {
      if (e.target === modalHelp) modalHelp.classList.add("hidden");
    };

    // --- Time Axis Pointer Drag (Sections 9, 10, 25) ---
    setupTimeAxisDragging();

    // --- Keyboard Control (Section 11, 26) ---
    window.addEventListener("keydown", onKeyDown);

    // Canvas Mouse Interaction for Time Mode / 4D Mode
    setupCanvasInteraction();
  }

  // --- Interactive 4D Time Axis Dragging ---
  function setupTimeAxisDragging() {
    let isDragging = false;

    function updateFromPointer(clientX) {
      const rect = axisRail.getBoundingClientRect();
      let norm = (clientX - rect.left) / rect.width;
      norm = Math.max(0.0, Math.min(1.0, norm));
      const targetIndex = Math.round(norm * (state.totalFrames - 1));
      jumpToFrameIndex(targetIndex);
    }

    axisRail.addEventListener("pointerdown", (e) => {
      isDragging = true;
      axisRail.setPointerCapture(e.pointerId);
      updateFromPointer(e.clientX);
    });

    window.addEventListener("pointermove", (e) => {
      if (isDragging) {
        updateFromPointer(e.clientX);
      }
    });

    window.addEventListener("pointerup", (e) => {
      if (isDragging) {
        isDragging = false;
        try { axisRail.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    });
  }

  // --- Canvas Mouse Control for Time Mode and 4D Nav ---
  function setupCanvasInteraction() {
    let isCanvasDragging = false;
    let startX = 0;
    let startY = 0;
    let startFrameIdx = 0;

    renderer.domElement.addEventListener("mousedown", (e) => {
      if (state.interactionMode === "TIME" || state.interactionMode === "4D_NAV") {
        isCanvasDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startFrameIdx = state.currentFrameIndex;
        controls.enabled = false;
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (isCanvasDragging) {
        const dx = e.clientX - startX;
        // Map horizontal drag to frame scrub
        const frameOffset = Math.round(dx * 0.35);
        const target = Math.max(0, Math.min(state.totalFrames - 1, startFrameIdx + frameOffset));
        jumpToFrameIndex(target);

        if (state.interactionMode === "4D_NAV") {
          const dy = e.clientY - startY;
          camera.position.y = Math.max(0.3, 1.8 - dy * 0.01);
          controls.update();
        }
      }
    });

    window.addEventListener("mouseup", () => {
      if (isCanvasDragging) {
        isCanvasDragging = false;
        if (state.interactionMode === "CAMERA") {
          controls.enabled = true;
        }
      }
    });
  }

  function setInteractionMode(mode) {
    state.interactionMode = mode;
    btnModeCamera.classList.toggle("active", mode === "CAMERA");
    btnModeTime.classList.toggle("active", mode === "TIME");
    btnMode4D.classList.toggle("active", mode === "4D_NAV");
    controls.enabled = (mode === "CAMERA");
  }

  // --- Keyboard Shortcuts (Section 26) ---
  function onKeyDown(e) {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        stepFrame(e.shiftKey ? -10 : -1);
        break;
      case "ArrowRight":
        e.preventDefault();
        stepFrame(e.shiftKey ? 10 : 1);
        break;
      case " ":
        e.preventDefault();
        togglePlay();
        break;
      case "r":
      case "R":
        toggleReverse();
        break;
      case "f":
      case "F":
        toggleFreeze();
        break;
      case "t":
      case "T":
        setInteractionMode("TIME");
        break;
      case "c":
      case "C":
        setInteractionMode("CAMERA");
        break;
      case "Home":
        e.preventDefault();
        jumpToFrameIndex(0);
        break;
      case "End":
        e.preventDefault();
        jumpToFrameIndex(state.totalFrames - 1);
        break;
    }
  }

  // --- Frame Navigation Core ---
  function jumpToFrameIndex(idx) {
    state.currentFrameIndex = Math.max(0, Math.min(state.totalFrames - 1, idx));
    updateTelemetryHUD();
    updateTimelineUI();
    renderTemporalSculpture();
  }

  function jumpToFrame(frameNumber) {
    const idx = frameNumber - state.startFrame;
    jumpToFrameIndex(idx);
  }

  function stepFrame(delta) {
    jumpToFrameIndex(state.currentFrameIndex + delta);
  }

  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    state.isFrozen = false;
    btnPlay.textContent = state.isPlaying ? "PAUSE" : "PLAY";
    btnFreeze.classList.remove("active");
  }

  function toggleReverse() {
    state.isReverse = !state.isReverse;
    btnReverse.classList.toggle("active", state.isReverse);
  }

  function toggleFreeze() {
    state.isFrozen = !state.isFrozen;
    if (state.isFrozen) {
      state.isPlaying = false;
      btnPlay.textContent = "PLAY";
      btnFreeze.classList.add("active");
    } else {
      btnFreeze.classList.remove("active");
    }
  }

  // --- Update HUD & Metrics (Section 24) ---
  function updateTelemetryHUD() {
    if (!state.telemetry || !state.telemetry.frames) return;
    const frameData = state.telemetry.frames[state.currentFrameIndex];
    if (!frameData) return;

    const frameNum = frameData.frame;
    const timeSec = (state.startFrame / state.fps + frameData.time).toFixed(2);
    const normCursor = (state.currentFrameIndex / (state.totalFrames - 1)).toFixed(3);

    valFrame.textContent = `${frameNum} / ${state.endFrame}`;
    valTime.textContent = `${timeSec} s`;
    valCursor.textContent = normCursor;

    if (frameData.key_action) {
      valAction.textContent = `${frameData.key_action.type} (${frameData.key_action.label})`;
    } else {
      valAction.textContent = "TACTICAL STANCE";
    }
  }

  // --- Update Timeline UI & Markers (Section 12, 13) ---
  function updateTimelineUI() {
    const norm = state.currentFrameIndex / (state.totalFrames - 1);
    const pct = norm * 100;
    axisProgress.style.width = `${pct}%`;
    axisCursor.style.left = `${pct}%`;

    // Render Historical Pose Markers on Axis Rail (Section 13)
    historyMarkersLayer.innerHTML = "";
    for (let k = 1; k < state.historyCount; k++) {
      const histFrameIdx = state.currentFrameIndex - k * state.poseDelayFrames;
      if (histFrameIdx >= 0) {
        const histNorm = histFrameIdx / (state.totalFrames - 1);
        const dot = document.createElement("div");
        dot.className = "h-marker-dot";
        dot.style.left = `${histNorm * 100}%`;
        dot.title = `Historical State T-${(k * state.poseDelaySec).toFixed(1)}s (Frame ${state.startFrame + histFrameIdx})`;
        dot.onclick = (e) => {
          e.stopPropagation();
          jumpToFrameIndex(histFrameIdx);
        };
        historyMarkersLayer.appendChild(dot);
      }
    }
  }

  // --- Render 4D Temporal Motion Sculpture ---
  function renderTemporalSculpture() {
    if (!state.telemetry || !state.telemetry.frames) return;

    // Clear previous geometries
    bodyLayersGroup.clear();
    pointCloudGroup.clear();
    trajectoryGroup.clear();
    impactGroup.clear();

    const activeIndices = [];
    for (let k = 0; k < state.historyCount; k++) {
      const idx = state.currentFrameIndex - k * state.poseDelayFrames;
      if (idx >= 0) {
        activeIndices.push({ historyIndex: k, frameIndex: idx });
      }
    }

    // 1. Render Body Layers (Section 3, 4, 28)
    if (state.showBodies && state.visualMode !== "POINT_CLOUD" && state.visualMode !== "TRAJECTORY") {
      activeIndices.forEach(({ historyIndex, frameIndex }) => {
        renderBodyPlane(historyIndex, frameIndex);
      });
    }

    // 2. Render Point Clouds / Splat Particles (Section 4, 33)
    if (state.showPoints && (state.visualMode === "POINT_CLOUD" || state.visualMode === "HYBRID" || state.visualMode === "GHOST")) {
      activeIndices.forEach(({ historyIndex, frameIndex }) => {
        renderPointCloud(historyIndex, frameIndex);
      });
    }

    // 3. Render 3D Motion Trajectories (Section 6, 31)
    if (state.showTrajectories && (state.visualMode === "TRAJECTORY" || state.visualMode === "HYBRID" || state.visualMode === "BODY")) {
      renderMotionTrajectories(activeIndices);
    }

    // 4. Render Collision Contact Points (Section 7, 32)
    if (state.showImpacts) {
      renderCollisionMemory(activeIndices);
    }
  }

  // --- Helper: Render Textured 3D Body Plane Layer ---
  function renderBodyPlane(historyIndex, frameIndex) {
    const zPos = -historyIndex * state.timeDepth;
    const opacity = OPACITY_LADDER[Math.min(historyIndex, OPACITY_LADDER.length - 1)];

    // Geometry: plane aspect matching video (874:400) -> w: 4.37, h: 2.0
    const geom = new THREE.PlaneGeometry(4.37, 2.0);

    // Compute UV offsets in sprite atlas (10x10 grid)
    const atlasIdx = Math.floor(frameIndex / 100);
    const subIdx = frameIndex % 100;
    const col = subIdx % 10;
    const row = Math.floor(subIdx / 10);

    const atlasTexture = atlasTextures.combined[atlasIdx];
    let mat;

    if (atlasTexture) {
      // Clone texture UV mapping
      const tex = atlasTexture.clone();
      tex.needsUpdate = true;
      tex.repeat.set(0.1, 0.1);
      tex.offset.set(col * 0.1, (9 - row) * 0.1); // OpenGL bottom-left origin

      mat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: opacity,
        depthWrite: historyIndex === 0,
        side: THREE.DoubleSide
      });
    } else {
      // Fallback procedural silhouette material
      mat = new THREE.MeshBasicMaterial({
        color: historyIndex === 0 ? 0x00e5ff : 0x38bdf8,
        transparent: true,
        opacity: opacity * 0.5,
        wireframe: true
      });
    }

    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(0, 1.0, zPos);
    bodyLayersGroup.add(mesh);
  }

  // --- Helper: Render Particle Point Clouds / Splats ---
  function renderPointCloud(historyIndex, frameIndex) {
    const frameData = state.telemetry.frames[frameIndex];
    if (!frameData || !frameData.points || frameData.points.length === 0) return;

    const zPos = -historyIndex * state.timeDepth;
    const baseOpacity = OPACITY_LADDER[Math.min(historyIndex, OPACITY_LADDER.length - 1)];

    const points = frameData.points;
    const count = points.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    // Parse configurable fighter colors
    const colA = new THREE.Color(state.colorFighterA);
    const colB = new THREE.Color(state.colorFighterB);

    // Dispersion increases with temporal age (FIBERS / DUST)
    const dispersion = historyIndex * 0.02;

    for (let i = 0; i < count; i++) {
      const p = points[i]; // [x, y, r, g, b, is_fighter_b]
      const isB = p[5] === 1;

      // Jitter / disperse older poses
      const jx = (Math.random() - 0.5) * dispersion;
      const jy = (Math.random() - 0.5) * dispersion;
      const jz = (Math.random() - 0.5) * dispersion;

      positions[i * 3 + 0] = p[0] + jx;
      positions[i * 3 + 1] = p[1] + jy;
      positions[i * 3 + 2] = zPos + jz;

      // Blend video rgb with fighter chromatic aura
      const baseCol = isB ? colB : colA;
      const vr = p[2] / 255;
      const vg = p[3] / 255;
      const vb = p[4] / 255;

      colors[i * 3 + 0] = THREE.MathUtils.lerp(vr, baseCol.r, 0.45);
      colors[i * 3 + 1] = THREE.MathUtils.lerp(vg, baseCol.g, 0.45);
      colors[i * 3 + 2] = THREE.MathUtils.lerp(vb, baseCol.b, 0.45);
    }

    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    pGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const pMat = new THREE.PointsMaterial({
      size: Math.max(0.015, 0.035 - historyIndex * 0.003),
      vertexColors: true,
      transparent: true,
      opacity: baseOpacity * 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const pMesh = new THREE.Points(pGeom, pMat);
    pointCloudGroup.add(pMesh);
  }

  // --- Helper: Render 3D Limb Motion Trajectories (Section 6, 31) ---
  function renderMotionTrajectories(activeIndices) {
    if (activeIndices.length < 2) return;

    const limbs = [
      { name: "left_hand", color: state.colorFighterA, fighter: "A" },
      { name: "right_hand", color: state.colorFighterA, fighter: "A" },
      { name: "left_foot", color: state.colorFighterA, fighter: "A" },
      { name: "right_foot", color: state.colorFighterA, fighter: "A" },
      { name: "head", color: state.colorFighterA, fighter: "A" },
      { name: "left_hand", color: state.colorFighterB, fighter: "B" },
      { name: "right_hand", color: state.colorFighterB, fighter: "B" },
      { name: "left_foot", color: state.colorFighterB, fighter: "B" },
      { name: "right_foot", color: state.colorFighterB, fighter: "B" },
      { name: "head", color: state.colorFighterB, fighter: "B" }
    ];

    limbs.forEach((limb) => {
      const curvePts = [];

      activeIndices.forEach(({ historyIndex, frameIndex }) => {
        const frameData = state.telemetry.frames[frameIndex];
        if (!frameData) return;
        const joints = limb.fighter === "A" ? frameData.joints_a : frameData.joints_b;
        if (joints && joints[limb.name]) {
          const pt = joints[limb.name];
          const zPos = -historyIndex * state.timeDepth + pt[2];
          curvePts.push(new THREE.Vector3(pt[0], pt[1], zPos));
        }
      });

      if (curvePts.length >= 2) {
        const curve = new THREE.CatmullRomCurve3(curvePts);
        const tubeGeom = new THREE.TubeGeometry(curve, 32, 0.018, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(limb.color),
          transparent: true,
          opacity: 0.85,
          wireframe: false
        });
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        trajectoryGroup.add(tubeMesh);
      }
    });
  }

  // --- Helper: Render Collision Contacts & Shockwaves (Section 7, 32) ---
  function renderCollisionMemory(activeIndices) {
    activeIndices.forEach(({ historyIndex, frameIndex }) => {
      const frameData = state.telemetry.frames[frameIndex];
      if (!frameData || !frameData.contacts || frameData.contacts.length === 0) return;

      const zBase = -historyIndex * state.timeDepth;

      frameData.contacts.forEach((c) => {
        const pos = c.pos;
        const worldPos = new THREE.Vector3(pos[0], pos[1], zBase + pos[2]);

        // Expanding Shockwave Ring
        const ringGeom = new THREE.RingGeometry(0.08, 0.14, 24);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xef4444,
          transparent: true,
          opacity: THREE.MathUtils.lerp(0.8, 0.2, historyIndex / state.historyCount),
          side: THREE.DoubleSide
        });
        const ringMesh = new THREE.Mesh(ringGeom, ringMat);
        ringMesh.position.copy(worldPos);
        impactGroup.add(ringMesh);

        // Core Glowing Node
        const sphereGeom = new THREE.SphereGeometry(0.04, 12, 12);
        const sphereMat = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.95
        });
        const sphereMesh = new THREE.Mesh(sphereGeom, sphereMat);
        sphereMesh.position.copy(worldPos);
        impactGroup.add(sphereMesh);
      });
    });
  }

  // --- Animation & Playback Engine (Section 21) ---
  let lastTime = 0;
  function animate(now) {
    requestAnimationFrame(animate);

    if (now && state.isPlaying && !state.isFrozen) {
      const dt = (now - lastTime) / 1000;
      const framesToAdvance = dt * state.fps * state.playbackSpeed;

      if (framesToAdvance >= 1.0) {
        lastTime = now;
        const delta = state.isReverse ? -1 : 1;
        let nextIdx = state.currentFrameIndex + delta;

        if (nextIdx >= state.totalFrames) {
          nextIdx = 0; // loop
        } else if (nextIdx < 0) {
          nextIdx = state.totalFrames - 1;
        }
        jumpToFrameIndex(nextIdx);
      }
    } else {
      lastTime = now;
    }

    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // Auto-launch on DOM ready
  window.addEventListener("DOMContentLoaded", init);
})();
