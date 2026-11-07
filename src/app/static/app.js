/**
 * AI Biometric Fingerprint Recognition & Forensic Classification System
 * Stage 10 Interactive Web Dashboard Application Logic
 */

(function () {
  "use strict";

  // Bulletproof DOM helper
  function setText(elem, text) {
    if (elem) elem.textContent = (text !== undefined && text !== null) ? String(text) : "";
  }

  // Application State
  const state = {
    currentResult: null,
    rawImage: null,
    compositeImage: null,
    damageImage: null,
    damageMaskImage: null,
    selectedFile: null,
    selectedSampleId: null,
    isAnalyzing: false,
  };

  // DOM Elements
  const el = {
    // Hardware & Header
    hardwareBadge: document.getElementById("hardwareBadge"),
    hardwareText: document.getElementById("hardwareText"),
    sampleSelect: document.getElementById("sampleSelect"),
    btnLoadSample: document.getElementById("btnLoadSample"),
    groupNistClean: document.getElementById("groupNistClean"),
    groupNistAmbiguous: document.getElementById("groupNistAmbiguous"),
    groupSocofingReal: document.getElementById("groupSocofingReal"),
    groupSocofingAltered: document.getElementById("groupSocofingAltered"),

    // Ingestion & Toolbar
    btnUploadImage: document.getElementById("btnUploadImage"),
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    handSelect: document.getElementById("handSelect"),
    digitSelect: document.getElementById("digitSelect"),
    btnAnalyze: document.getElementById("btnAnalyze"),
    analyzeSpinner: document.getElementById("analyzeSpinner"),
    analyzeBtnText: document.getElementById("analyzeBtnText"),
    inputStatusBadge: document.getElementById("inputStatusBadge"),

    // Canvas Viewer
    chipFilename: document.getElementById("chipFilename"),
    chipResolution: document.getElementById("chipResolution"),
    chipQuality: document.getElementById("chipQuality"),
    chipGate: document.getElementById("chipGate"),
    canvasViewport: document.getElementById("canvasViewport"),
    fingerprintCanvas: document.getElementById("fingerprintCanvas"),
    canvasPlaceholder: document.getElementById("canvasOverlayPlaceholder"),
    btnZoomFit: document.getElementById("btnZoomFit"),
    btnZoom100: document.getElementById("btnZoom100"),

    // Layer Toggles
    toggleCores: document.getElementById("toggleCores"),
    toggleDeltas: document.getElementById("toggleDeltas"),
    toggleStreamlines: document.getElementById("toggleStreamlines"),
    toggleDamage: document.getElementById("toggleDamage"),
    toggleComposite: document.getElementById("toggleComposite"),

    // Forensic Inspection Lightbox Modal
    imageModal: document.getElementById("imageModal"),
    modalImageMeta: document.getElementById("modalImageMeta"),
    modalViewport: document.getElementById("modalViewport"),
    modalCanvasContainer: document.getElementById("modalCanvasContainer"),
    modalCanvas: document.getElementById("modalCanvas"),
    modalZoomLevel: document.getElementById("modalZoomLevel"),
    modalBtnZoomIn: document.getElementById("modalBtnZoomIn"),
    modalBtnZoomOut: document.getElementById("modalBtnZoomOut"),
    modalBtnFit: document.getElementById("modalBtnFit"),
    modalBtn100: document.getElementById("modalBtn100"),
    modalBtnClose: document.getElementById("modalBtnClose"),
    modalToggleCores: document.getElementById("modalToggleCores"),
    modalToggleDeltas: document.getElementById("modalToggleDeltas"),
    modalToggleStreamlines: document.getElementById("modalToggleStreamlines"),
    modalToggleDamage: document.getElementById("modalToggleDamage"),

    // Results & Rationale
    heroBanner: document.getElementById("heroBanner"),
    primaryClassTitle: document.getElementById("primaryClassTitle"),
    nistSymbolBadge: document.getElementById("nistSymbolBadge"),
    primaryConfBadge: document.getElementById("primaryConfBadge"),
    confidenceBarFill: document.getElementById("confidenceBarFill"),
    confidencePercentageText: document.getElementById("confidencePercentageText"),
    classDistributionGrid: document.getElementById("classDistributionGrid"),

    subtypeStatusBadge: document.getElementById("subtypeStatusBadge"),
    subtypeName: document.getElementById("subtypeName"),
    subtypeCategory: document.getElementById("subtypeCategory"),
    verifiabilityBadge: document.getElementById("verifiabilityBadge"),
    subtypeReasonText: document.getElementById("subtypeReasonText"),

    pipelineLatency: document.getElementById("pipelineLatency"),
    valCoresCount: document.getElementById("valCoresCount"),
    valCoresCoord: document.getElementById("valCoresCoord"),
    valDeltasCount: document.getElementById("valDeltasCount"),
    valDeltasCoord: document.getElementById("valDeltasCoord"),
    valTruncation: document.getElementById("valTruncation"),
    valTruncationDetail: document.getElementById("valTruncationDetail"),
    valRidgeCount: document.getElementById("valRidgeCount"),
    valWavelength: document.getElementById("valWavelength"),
    valFrequency: document.getElementById("valFrequency"),
    valFlowDirection: document.getElementById("valFlowDirection"),
    valExitAngle: document.getElementById("valExitAngle"),
    valCoherence: document.getElementById("valCoherence"),
    valCoherenceTier: document.getElementById("valCoherenceTier"),

    qualityCard: document.getElementById("qualityCard"),
    qualityTierBadge: document.getElementById("qualityTierBadge"),
    overallQualityScore: document.getElementById("overallQualityScore"),
    qUsableArea: document.getElementById("qUsableArea"),
    qClarity: document.getElementById("qClarity"),
    qFreqCont: document.getElementById("qFreqCont"),
    qDamageStatus: document.getElementById("qDamageStatus"),
    qualityAlertBanner: document.getElementById("qualityAlertBanner"),
    qualityAlertText: document.getElementById("qualityAlertText"),

    explanationAccordion: document.getElementById("explanationAccordion"),

    btnExportReport: document.getElementById("btnExportReport"),
    btnExportJson: document.getElementById("btnExportJson"),
    btnPrintReport: document.getElementById("btnPrintReport"),
    btnResetView: document.getElementById("btnResetView"),

    // False Prediction & Feedback Modal
    btnFalsePrediction: document.getElementById("btnFalsePrediction"),
    feedbackModal: document.getElementById("feedbackModal"),
    btnCancelFeedbackX: document.getElementById("btnCancelFeedbackX"),
    btnCancelFeedback: document.getElementById("btnCancelFeedback"),
    btnSubmitFeedback: document.getElementById("btnSubmitFeedback"),
    feedbackAiSummary: document.getElementById("feedbackAiSummary"),
    feedbackWhorlWrap: document.getElementById("feedbackWhorlWrap"),
    feedbackWhorlSubtype: document.getElementById("feedbackWhorlSubtype"),
    feedbackCustomWrap: document.getElementById("feedbackCustomWrap"),
    feedbackCustomText: document.getElementById("feedbackCustomText"),
    feedbackNotes: document.getElementById("feedbackNotes"),

    // Review Vault
    btnOpenVault: document.getElementById("btnOpenVault"),
    vaultBadgeCount: document.getElementById("vaultBadgeCount"),
    vaultModal: document.getElementById("vaultModal"),
    btnCloseVault: document.getElementById("btnCloseVault"),
    vaultFilterPills: document.getElementById("vaultFilterPills"),
    vaultListContainer: document.getElementById("vaultListContainer"),
    vaultEmptyState: document.getElementById("vaultEmptyState"),
    vaultDetailContent: document.getElementById("vaultDetailContent"),
    btnDownloadAllVault: document.getElementById("btnDownloadAllVault"),
    vaultConfirmModal: document.getElementById("vaultConfirmModal"),
    vaultConfirmTitle: document.getElementById("vaultConfirmTitle"),
    vaultConfirmMessage: document.getElementById("vaultConfirmMessage"),
    btnCancelVaultConfirm: document.getElementById("btnCancelVaultConfirm"),
    btnAcceptVaultConfirm: document.getElementById("btnAcceptVaultConfirm"),
  };

  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  async function init() {
    setupEventListeners();
    await checkHealth();
    await fetchVaultCount();
    if (el.groupNistClean) {
      await loadSampleGallery();
    }

    // Auto-analyze default clean NIST whorl sample on initial load
    state.selectedSampleId = "nist_whorl";
    analyzeCurrent();
  }

  // ------------------------------------------------------------------------
  // Health & System Status
  // ------------------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      if (data.mps_active) {
        if (el.hardwareText) setText(el.hardwareText, "MPS GPU ACTIVE");
        if (el.hardwareBadge) {
          el.hardwareBadge.style.color = "var(--emerald-bright)";
          el.hardwareBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
        }
      } else {
        if (el.hardwareText) setText(el.hardwareText, `DEVICE: ${data.device.toUpperCase()}`);
      }
    } catch (err) {
      if (el.hardwareText) setText(el.hardwareText, "BACKEND DISCONNECTED");
      if (el.hardwareBadge) el.hardwareBadge.style.color = "var(--rose-bright)";
    }
  }

  // ------------------------------------------------------------------------
  // Sample Gallery (guarded for headless or custom use)
  // ------------------------------------------------------------------------
  async function loadSampleGallery() {
    if (!el.groupNistClean) return;
    try {
      const res = await fetch("/api/sample-gallery");
      if (!res.ok) return;
      const data = await res.json();
      const samples = data.samples || [];

      // Clear groups
      if (el.groupNistClean) el.groupNistClean.innerHTML = "";
      if (el.groupNistAmbiguous) el.groupNistAmbiguous.innerHTML = "";
      if (el.groupSocofingReal) el.groupSocofingReal.innerHTML = "";
      if (el.groupSocofingAltered) el.groupSocofingAltered.innerHTML = "";

      samples.forEach((sample) => {
        const opt = document.createElement("option");
        opt.value = sample.id;
        opt.textContent = `${sample.name} [${sample.expected_class}]`;
        opt.dataset.hand = sample.hand || "";
        opt.dataset.digit = sample.digit_code || "";

        if (sample.category.includes("NIST Clean") && el.groupNistClean) {
          el.groupNistClean.appendChild(opt);
        } else if (sample.category.includes("NIST Ambiguous") && el.groupNistAmbiguous) {
          el.groupNistAmbiguous.appendChild(opt);
        } else if (sample.category.includes("SOCOFing Real") && el.groupSocofingReal) {
          el.groupSocofingReal.appendChild(opt);
        } else if (sample.category.includes("SOCOFing Altered") && el.groupSocofingAltered) {
          el.groupSocofingAltered.appendChild(opt);
        }
      });
    } catch (err) {
      console.warn("Could not load sample gallery:", err);
    }
  }

  // ------------------------------------------------------------------------
  // Event Listeners
  // ------------------------------------------------------------------------
  function setupEventListeners() {
    // Upload New Image Button
    if (el.btnUploadImage) {
      el.btnUploadImage.addEventListener("click", () => {
        if (el.fileInput) el.fileInput.click();
      });
    }

    // Sample loader (guarded if elements exist)
    if (el.btnLoadSample && el.sampleSelect) {
      el.btnLoadSample.addEventListener("click", () => {
        const sid = el.sampleSelect.value;
        if (!sid) return;
        handleSampleSelection(sid);
      });
    }

    if (el.sampleSelect) {
      el.sampleSelect.addEventListener("change", () => {
        const sid = el.sampleSelect.value;
        if (!sid) return;
        handleSampleSelection(sid);
      });
    }

    // File Drag and Drop & Click to Browse
    if (el.dropZone) {
      el.dropZone.addEventListener("click", (e) => {
        if (el.fileInput && e.target !== el.fileInput) {
          el.fileInput.click();
        }
      });

      el.dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        el.dropZone.classList.add("dragover");
      });

      el.dropZone.addEventListener("dragleave", () => {
        el.dropZone.classList.remove("dragover");
      });

      el.dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        el.dropZone.classList.remove("dragover");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleFileSelection(e.dataTransfer.files[0]);
        }
      });
    }

    if (el.fileInput) {
      el.fileInput.addEventListener("click", () => {
        el.fileInput.value = "";
      });

      el.fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          handleFileSelection(e.target.files[0]);
        }
      });
    }

    // Prevent accidental browser navigation on window file drop
    window.addEventListener("dragover", (e) => e.preventDefault(), false);
    window.addEventListener("drop", (e) => e.preventDefault(), false);

    // Run Analysis Button
    if (el.btnAnalyze) {
      el.btnAnalyze.addEventListener("click", () => {
        analyzeCurrent();
      });
    }

    // Layer Toggles (Main)
    [el.toggleCores, el.toggleDeltas, el.toggleStreamlines, el.toggleDamage, el.toggleComposite].forEach(
      (toggle) => {
        if (!toggle) return;
        toggle.addEventListener("change", () => {
          if (el.modalToggleCores && el.toggleCores) el.modalToggleCores.checked = el.toggleCores.checked;
          if (el.modalToggleDeltas && el.toggleDeltas) el.modalToggleDeltas.checked = el.toggleDeltas.checked;
          if (el.modalToggleStreamlines && el.toggleStreamlines) el.modalToggleStreamlines.checked = el.toggleStreamlines.checked;
          if (el.modalToggleDamage && el.toggleDamage) el.modalToggleDamage.checked = el.toggleDamage.checked;
          renderCanvas();
          if (el.imageModal && !el.imageModal.classList.contains("hidden")) {
            renderModalCanvas();
          }
        });
      }
    );

    // Zoom Controls (Main Dashboard Canvas)
    if (el.btnZoomFit) {
      el.btnZoomFit.addEventListener("click", () => {
        if (el.fingerprintCanvas) {
          el.fingerprintCanvas.style.maxWidth = "100%";
          el.fingerprintCanvas.style.maxHeight = "460px";
        }
      });
    }

    if (el.btnZoom100) {
      el.btnZoom100.addEventListener("click", () => {
        if (state.currentResult && state.currentResult.input_shape && el.fingerprintCanvas) {
          const [h, w] = state.currentResult.input_shape;
          el.fingerprintCanvas.style.maxWidth = `${w}px`;
          el.fingerprintCanvas.style.maxHeight = `${h}px`;
        }
      });
    }

    // Open Modal on Clicking Canvas or Viewport
    if (el.canvasViewport) {
      el.canvasViewport.addEventListener("click", () => {
        if (state.rawImage) {
          openImageModal();
        }
      });
    }

    if (el.fingerprintCanvas) {
      el.fingerprintCanvas.addEventListener("click", (e) => {
        if (state.rawImage) {
          e.stopPropagation();
          openImageModal();
        }
      });
    }

    // Modal Zoom Toolbar Buttons
    if (el.modalBtnZoomIn) {
      el.modalBtnZoomIn.addEventListener("click", () => {
        zoomModalAt(modalZoom * 1.25);
      });
    }

    if (el.modalBtnZoomOut) {
      el.modalBtnZoomOut.addEventListener("click", () => {
        zoomModalAt(modalZoom * 0.8);
      });
    }

    if (el.modalBtnFit) {
      el.modalBtnFit.addEventListener("click", () => {
        fitModalToViewport();
      });
    }

    if (el.modalBtn100) {
      el.modalBtn100.addEventListener("click", () => {
        setModalTransform(1.0, 0, 0);
      });
    }

    if (el.modalBtnClose) {
      el.modalBtnClose.addEventListener("click", () => {
        closeImageModal();
      });
    }

    // Modal Background Click (Outside Window)
    if (el.imageModal) {
      el.imageModal.addEventListener("click", (e) => {
        if (e.target === el.imageModal) {
          closeImageModal();
        }
      });
    }

    // Modal Layer Toggles
    if (el.modalToggleCores) {
      el.modalToggleCores.addEventListener("change", () => {
        if (el.toggleCores) el.toggleCores.checked = el.modalToggleCores.checked;
        renderModalCanvas();
        renderCanvas();
      });
    }

    if (el.modalToggleDeltas) {
      el.modalToggleDeltas.addEventListener("change", () => {
        if (el.toggleDeltas) el.toggleDeltas.checked = el.modalToggleDeltas.checked;
        renderModalCanvas();
        renderCanvas();
      });
    }

    if (el.modalToggleStreamlines) {
      el.modalToggleStreamlines.addEventListener("change", () => {
        if (el.toggleStreamlines) el.toggleStreamlines.checked = el.modalToggleStreamlines.checked;
        renderModalCanvas();
        renderCanvas();
      });
    }

    if (el.modalToggleDamage) {
      el.modalToggleDamage.addEventListener("change", () => {
        if (el.toggleDamage) el.toggleDamage.checked = el.modalToggleDamage.checked;
        renderModalCanvas();
        renderCanvas();
      });
    }

    // Modal Viewport Wheel Zoom & Drag Pan
    if (el.modalViewport) {
      el.modalViewport.addEventListener("wheel", (e) => {
        e.preventDefault();
        let factor;
        if (e.ctrlKey) {
          // macOS Apple Silicon Trackpad Pinch Gesture
          factor = Math.exp(-e.deltaY * 0.01);
        } else {
          // Discrete mouse wheel or smooth trackpad two-finger scroll
          const clamped = Math.max(-80, Math.min(80, e.deltaY));
          factor = Math.exp(-clamped * 0.0025);
        }
        zoomModalAt(modalZoom * factor, e.clientX, e.clientY);
      }, { passive: false });

      // Intercept Safari macOS trackpad gesture events to prevent native page zoom
      el.modalViewport.addEventListener("gesturestart", (e) => e.preventDefault());
      el.modalViewport.addEventListener("gesturechange", (e) => e.preventDefault());
      el.modalViewport.addEventListener("gestureend", (e) => e.preventDefault());

      el.modalViewport.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        if (e.target.closest("button") || e.target.classList.contains("modal-hint-bar")) return;
        e.preventDefault();

        isDraggingModal = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragStartPanX = modalPanX;
        dragStartPanY = modalPanY;
        el.modalViewport.classList.add("is-dragging");
      });
    }

    window.addEventListener("mousemove", (e) => {
      if (!isDraggingModal) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      setModalTransform(modalZoom, dragStartPanX + dx, dragStartPanY + dy);
    });

    window.addEventListener("mouseup", () => {
      if (isDraggingModal) {
        isDraggingModal = false;
        if (el.modalViewport) el.modalViewport.classList.remove("is-dragging");
      }
    });

    // Window Resize Handler for Modal Viewport
    window.addEventListener("resize", () => {
      if (el.imageModal && !el.imageModal.classList.contains("hidden")) {
        fitModalToViewport();
      }
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (el.vaultConfirmModal && !el.vaultConfirmModal.classList.contains("hidden")) {
          e.preventDefault();
          closeVaultConfirm();
          return;
        }
        if (el.vaultModal && !el.vaultModal.classList.contains("hidden")) {
          e.preventDefault();
          closeVaultModal();
          return;
        }
        if (el.feedbackModal && !el.feedbackModal.classList.contains("hidden")) {
          e.preventDefault();
          closeFeedbackModal();
          return;
        }
        if (el.imageModal && !el.imageModal.classList.contains("hidden")) {
          e.preventDefault();
          closeImageModal();
          return;
        }
      }

      if (!el.imageModal || el.imageModal.classList.contains("hidden")) return;
      if (e.target && (e.target.tagName === "INPUT" && e.target.type !== "checkbox" || e.target.tagName === "TEXTAREA")) return;

      if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomModalAt(modalZoom * 1.25);
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomModalAt(modalZoom * 0.8);
      } else if (e.key === "0") {
        e.preventDefault();
        fitModalToViewport();
      }
    });

    // Safely guarded legacy export handlers (if present in DOM)
    if (el.btnExportReport) {
      el.btnExportReport.addEventListener("click", () => {
        window.print();
      });
    }

    if (el.btnExportJson) {
      el.btnExportJson.addEventListener("click", () => {
        if (!state.currentResult) return;
        const jsonStr = JSON.stringify(state.currentResult, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const fname = (state.currentResult.filename || "forensic_report").replace(/\.[^/.]+$/, "");
        const a = document.createElement("a");
        a.href = url;
        a.download = `${fname}_forensic_report.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    }

    // Print / Save PDF Report (Primary Action)
    if (el.btnPrintReport) {
      el.btnPrintReport.addEventListener("click", () => {
        window.print();
      });
    }

    // Reset Dashboard View (Secondary Action)
    if (el.btnResetView) {
      el.btnResetView.addEventListener("click", () => {
        resetDashboard();
      });
    }

    // False Prediction & Feedback Modal Listeners
    if (el.btnFalsePrediction) {
      el.btnFalsePrediction.addEventListener("click", () => {
        openFeedbackModal();
      });
    }

    if (el.btnCancelFeedback) {
      el.btnCancelFeedback.addEventListener("click", () => {
        closeFeedbackModal();
      });
    }

    if (el.btnCancelFeedbackX) {
      el.btnCancelFeedbackX.addEventListener("click", () => {
        closeFeedbackModal();
      });
    }

    if (el.feedbackModal) {
      el.feedbackModal.addEventListener("click", (e) => {
        if (e.target === el.feedbackModal) {
          closeFeedbackModal();
        }
      });
    }

    document.querySelectorAll('input[name="feedbackCategoryOption"]').forEach((radio) => {
      radio.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val === "Other") {
          if (el.feedbackCustomWrap) el.feedbackCustomWrap.classList.remove("hidden");
          if (el.feedbackWhorlWrap) el.feedbackWhorlWrap.classList.add("hidden");
          if (el.feedbackCustomText) el.feedbackCustomText.focus();
        } else if (val === "Whorl") {
          if (el.feedbackWhorlWrap) el.feedbackWhorlWrap.classList.remove("hidden");
          if (el.feedbackCustomWrap) el.feedbackCustomWrap.classList.add("hidden");
        } else {
          if (el.feedbackCustomWrap) el.feedbackCustomWrap.classList.add("hidden");
          if (el.feedbackWhorlWrap) el.feedbackWhorlWrap.classList.add("hidden");
        }
      });
    });

    if (el.btnSubmitFeedback) {
      el.btnSubmitFeedback.addEventListener("click", () => {
        submitFeedback();
      });
    }

    // Review Vault Modal Listeners
    if (el.btnOpenVault) {
      el.btnOpenVault.addEventListener("click", () => {
        openVaultModal();
      });
    }

    if (el.btnCloseVault) {
      el.btnCloseVault.addEventListener("click", () => {
        closeVaultModal();
      });
    }

    if (el.vaultModal) {
      el.vaultModal.addEventListener("click", (e) => {
        if (e.target === el.vaultModal) {
          closeVaultModal();
        }
      });
    }

    if (el.vaultFilterPills) {
      el.vaultFilterPills.querySelectorAll(".filter-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
          el.vaultFilterPills.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
          pill.classList.add("active");
          selectedVaultCategory = pill.dataset.category || "All";
          loadVaultData(selectedVaultCategory);
        });
      });
    }

    if (el.btnDownloadAllVault) {
      el.btnDownloadAllVault.addEventListener("click", () => {
        handleDownloadAllVault();
      });
    }

    if (el.btnCancelVaultConfirm) {
      el.btnCancelVaultConfirm.addEventListener("click", () => {
        closeVaultConfirm();
      });
    }

    if (el.btnAcceptVaultConfirm) {
      el.btnAcceptVaultConfirm.addEventListener("click", () => {
        const cb = pendingVaultConfirmCallback;
        closeVaultConfirm();
        if (cb) cb();
      });
    }

    if (el.vaultConfirmModal) {
      el.vaultConfirmModal.addEventListener("click", (e) => {
        if (e.target === el.vaultConfirmModal) {
          closeVaultConfirm();
        }
      });
    }
  }

  function handleSampleSelection(sid) {
    if (!sid) return;
    state.selectedSampleId = sid;
    state.selectedFile = null;

    if (el.sampleSelect) {
      const selectedOpt = el.sampleSelect.options ? el.sampleSelect.options[el.sampleSelect.selectedIndex] : null;
      if (selectedOpt && selectedOpt.dataset && selectedOpt.dataset.hand) {
        if (el.handSelect) el.handSelect.value = selectedOpt.dataset.hand;
      }
      if (selectedOpt && selectedOpt.dataset && selectedOpt.dataset.digit) {
        if (el.digitSelect) el.digitSelect.value = selectedOpt.dataset.digit;
      }
    }

    if (el.inputStatusBadge) {
      setText(el.inputStatusBadge, `Sample: ${sid}`);
      el.inputStatusBadge.className = "badge badge-info";
    }
    analyzeCurrent();
  }

  function handleFileSelection(file) {
    if (!file) return;
    state.selectedFile = file;
    state.selectedSampleId = null;
    if (el.inputStatusBadge) {
      setText(el.inputStatusBadge, `File: ${file.name}`);
      el.inputStatusBadge.className = "badge badge-status";
    }

    if (el.dropZone) {
      const p = el.dropZone.querySelector(".drop-text");
      if (p) p.innerHTML = `Loaded: <strong>${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
    }

    // Auto-analyze uploaded file immediately for responsive UX
    analyzeCurrent();
  }

  function setAnalyzingState(loading) {
    state.isAnalyzing = loading;
    if (loading) {
      if (el.btnAnalyze) el.btnAnalyze.disabled = true;
      if (el.analyzeSpinner) el.analyzeSpinner.classList.remove("hidden");
      if (el.analyzeBtnText) setText(el.analyzeBtnText, "Analyzing Ridge Geometry...");
    } else {
      if (el.btnAnalyze) el.btnAnalyze.disabled = false;
      if (el.analyzeSpinner) el.analyzeSpinner.classList.add("hidden");
      if (el.analyzeBtnText) setText(el.analyzeBtnText, "Run Forensic Analysis");
    }
  }

  // ------------------------------------------------------------------------
  // Pipeline Execution (POST /api/analyze)
  // ------------------------------------------------------------------------
  async function analyzeCurrent() {
    if (state.isAnalyzing) return;

    const formData = new FormData();
    if (state.selectedFile) {
      formData.append("file", state.selectedFile);
    } else if (state.selectedSampleId) {
      formData.append("sample_id", state.selectedSampleId);
    } else if (el.sampleSelect && el.sampleSelect.value) {
      formData.append("sample_id", el.sampleSelect.value);
    } else {
      alert("Please upload or drag & drop a fingerprint impression to inspect.");
      return;
    }

    if (el.handSelect && el.handSelect.value) {
      formData.append("hand", el.handSelect.value);
    }
    if (el.digitSelect && el.digitSelect.value) {
      formData.append("digit_code", el.digitSelect.value);
    }

    setAnalyzingState(true);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Analysis request failed (HTTP ${response.status})`);
      }

      const result = await response.json();
      state.currentResult = result;

      // Load visual assets into memory
      await loadImages(result.visualizations);

      // Update UI components
      updateDashboard(result);
      renderCanvas();
      if (el.imageModal && !el.imageModal.classList.contains("hidden")) {
        renderModalCanvas();
        if (el.modalImageMeta) {
          const filename = result.filename || "impression.png";
          const imgW = state.rawImage ? state.rawImage.width : 512;
          const imgH = state.rawImage ? state.rawImage.height : 512;
          const w = el.modalCanvas ? el.modalCanvas.width : (imgW || 512);
          const h = el.modalCanvas ? el.modalCanvas.height : (imgH || 512);
          setText(el.modalImageMeta, `${filename} • ${w} × ${h} px`);
        }
      }
    } catch (err) {
      console.error("Forensic analysis error:", err);
      alert(`Forensic Analysis Error: ${err.message}`);
    } finally {
      setAnalyzingState(false);
    }
  }

  // ------------------------------------------------------------------------
  // Image Asset Loader
  // ------------------------------------------------------------------------
  function loadImagePromise(src) {
    return new Promise((resolve) => {
      if (!src) return resolve(null);
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  async function loadImages(vis) {
    if (!vis) return;
    const [raw, composite, damage] = await Promise.all([
      loadImagePromise(vis.image_base64),
      loadImagePromise(vis.composite_overlay_base64),
      loadImagePromise(vis.damage_mask_base64),
    ]);
    state.rawImage = raw;
    state.compositeImage = composite;
    state.damageImage = damage;
    state.damageMaskImage = damage;
  }

  // ------------------------------------------------------------------------
  // UI Dashboard Updater
  // ------------------------------------------------------------------------
  function updateDashboard(res) {
    if (!res) return;

    // 1. Metadata Chips
    if (el.chipFilename) setText(el.chipFilename, res.filename || "impression.png");
    if (el.chipResolution) {
      if (res.input_shape) {
        setText(el.chipResolution, `${res.input_shape[0]} × ${res.input_shape[1]} px`);
      } else {
        setText(el.chipResolution, "-- × -- px");
      }
    }

    const q = res.quality || {};
    const qScore = q.overall_score !== undefined ? q.overall_score.toFixed(1) : "--";
    const qTier = q.quality_tier || "N/A";
    if (el.chipQuality) setText(el.chipQuality, `Q: ${qScore} (${qTier})`);

    const gate = res.decision_gate || "ACCEPTED";
    if (el.chipGate) {
      setText(el.chipGate, `GATE: ${gate}`);
      if (gate === "ACCEPTED") {
        el.chipGate.style.background = "rgba(16, 185, 129, 0.15)";
        el.chipGate.style.color = "var(--emerald-bright)";
        el.chipGate.style.borderColor = "rgba(16, 185, 129, 0.35)";
      } else if (gate === "ACCEPTED_WITH_WARNINGS") {
        el.chipGate.style.background = "rgba(245, 158, 11, 0.15)";
        el.chipGate.style.color = "var(--amber-bright)";
        el.chipGate.style.borderColor = "rgba(245, 158, 11, 0.35)";
      } else {
        el.chipGate.style.background = "rgba(244, 63, 94, 0.15)";
        el.chipGate.style.color = "var(--rose-bright)";
        el.chipGate.style.borderColor = "rgba(244, 63, 94, 0.35)";
      }
    }

    // 2. Macro Anatomical Class Banner
    const cls = res.classification || {};
    const anatLabel = cls.anatomical_label || "Unknown";
    const primClass = cls.primary_class || "?";
    const conf = cls.confidence !== undefined ? (cls.confidence * 100).toFixed(1) : "0.0";

    if (el.primaryClassTitle) setText(el.primaryClassTitle, anatLabel);
    if (el.nistSymbolBadge) setText(el.nistSymbolBadge, `[${primClass}]`);
    if (el.primaryConfBadge) setText(el.primaryConfBadge, `${conf}% Confidence`);
    if (el.confidenceBarFill) {
      const confVal = cls.confidence !== undefined ? cls.confidence * 100 : 0;
      el.confidenceBarFill.style.width = `${Math.min(Math.max(confVal, 0), 100)}%`;
    }
    if (el.confidencePercentageText) setText(el.confidencePercentageText, `${conf}%`);

    // Multi-class Softmax breakdown
    if (el.classDistributionGrid) {
      el.classDistributionGrid.innerHTML = "";
      const probs = cls.probabilities || {};
      const nistClasses = ["W", "L", "R", "A", "T"];
      const classNames = { W: "Whorl", L: "Left Loop", R: "Right Loop", A: "Arch", T: "Tented Arch" };

      nistClasses.forEach((c) => {
        const p = probs[c] !== undefined ? (probs[c] * 100).toFixed(1) : "0.0";
        const cell = document.createElement("div");
        cell.className = `prob-cell ${c === primClass ? "active-class" : ""}`;
        cell.innerHTML = `
          <span class="prob-class-code">${escapeHtml(c)} • ${escapeHtml(classNames[c] || c)}</span>
          <span class="prob-value">${escapeHtml(p)}%</span>
        `;
        el.classDistributionGrid.appendChild(cell);
      });
    }

    // 3. Subtype Refinement Card
    const sub = res.subtype || {};
    if (el.subtypeName) setText(el.subtypeName, sub.subtype || "Uncertain / Degraded");
    if (el.subtypeStatusBadge) {
      setText(el.subtypeStatusBadge, sub.status || "Uncertain");
      if (sub.is_uncertain) {
        el.subtypeStatusBadge.className = "badge badge-accent";
        el.subtypeStatusBadge.style.color = "var(--amber-bright)";
        el.subtypeStatusBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
      } else {
        el.subtypeStatusBadge.className = "badge badge-status";
        el.subtypeStatusBadge.style.color = "";
        el.subtypeStatusBadge.style.borderColor = "";
      }
    }

    if (el.subtypeCategory) setText(el.subtypeCategory, `Category: ${sub.primary_category || "Unclassified"}`);
    if (el.verifiabilityBadge) setText(el.verifiabilityBadge, sub.forensic_verifiability || "Expert Discretion");
    if (el.subtypeReasonText) setText(el.subtypeReasonText, sub.reason || "Subtype rule synthesis evaluated dermatoglyphic topology.");

    // 4. Dermatoglyphic Landmark Metrics
    const feat = res.features || {};
    const times = res.execution_time_ms || {};
    if (el.pipelineLatency) {
      setText(el.pipelineLatency, `Pipeline Latency: ${(times.total_ms || 0).toFixed(1)} ms`);
    }

    if (el.valCoresCount) setText(el.valCoresCount, feat.num_cores !== undefined ? feat.num_cores : 0);
    const coresList = feat.cores || [];
    if (el.valCoresCoord) {
      if (coresList.length === 0) {
        setText(el.valCoresCoord, "Zero cores detected");
        el.valCoresCoord.removeAttribute("title");
      } else {
        const formattedList = coresList.map((c) => `(${Math.round(c.x)}, ${Math.round(c.y)})`);
        const allCoordsStr = formattedList.join(", ");
        el.valCoresCoord.title = allCoordsStr;
        if (coresList.length <= 2) {
          setText(el.valCoresCoord, `Coords: ${allCoordsStr}`);
        } else {
          const firstTwo = formattedList.slice(0, 2).join(", ");
          setText(el.valCoresCoord, `Coords: ${firstTwo} (+${coresList.length - 2} more)`);
        }
      }
    }

    if (el.valDeltasCount) setText(el.valDeltasCount, feat.num_deltas !== undefined ? feat.num_deltas : 0);
    const deltasList = feat.deltas || [];
    if (el.valDeltasCoord) {
      if (deltasList.length === 0) {
        setText(el.valDeltasCoord, "Zero deltas detected");
        el.valDeltasCoord.removeAttribute("title");
      } else {
        const formattedDeltas = deltasList.map((d) => `(${Math.round(d.x)}, ${Math.round(d.y)})`);
        const allDeltasStr = formattedDeltas.join(", ");
        el.valDeltasCoord.title = allDeltasStr;
        if (deltasList.length <= 2) {
          setText(el.valDeltasCoord, `Coords: ${allDeltasStr}`);
        } else {
          const firstTwo = formattedDeltas.slice(0, 2).join(", ");
          setText(el.valDeltasCoord, `Coords: ${firstTwo} (+${deltasList.length - 2} more)`);
        }
      }
    }

    if (el.valTruncation) {
      if (feat.delta_truncated) {
        setText(el.valTruncation, "Margin Truncated");
        el.valTruncation.style.color = "var(--amber-bright)";
      } else {
        setText(el.valTruncation, "Enclosed");
        el.valTruncation.style.color = "#ffffff";
      }
    }
    if (el.valTruncationDetail) {
      if (feat.delta_truncated) {
        setText(el.valTruncationDetail, `Sector: ${feat.expected_delta_sector || "Perimeter"}`);
      } else {
        setText(el.valTruncationDetail, "All landmarks within impression");
      }
    }

    // Dermatoglyphic Ridge Count
    if (el.valRidgeCount) {
      if (feat.num_cores === 0 && feat.num_deltas === 0) {
        setText(el.valRidgeCount, "0.0 (Arch)");
      } else if (feat.ridge_count !== undefined && feat.ridge_count !== null) {
        setText(el.valRidgeCount, `${feat.ridge_count.toFixed(1)} ridges`);
      } else {
        setText(el.valRidgeCount, "--");
      }
    }

    if (el.valWavelength) {
      setText(el.valWavelength, feat.median_wavelength ? `Wavelength: ${feat.median_wavelength.toFixed(1)} px` : "--");
    }
    if (el.valFrequency) {
      setText(el.valFrequency, feat.median_frequency ? `Freq: ${feat.median_frequency.toFixed(3)} cyc/px` : "--");
    }

    if (el.valFlowDirection) setText(el.valFlowDirection, feat.flow_direction || "--");
    if (el.valExitAngle) {
      setText(el.valExitAngle, feat.exit_angle_deg !== undefined ? `${feat.exit_angle_deg.toFixed(1)}° Drift` : "--");
    }

    const coh = feat.mean_coherence !== undefined ? feat.mean_coherence.toFixed(2) : "--";
    if (el.valCoherence) setText(el.valCoherence, coh);
    if (el.valCoherenceTier) {
      setText(el.valCoherenceTier, feat.mean_coherence > 0.6 ? "High Ridge Coherence" : feat.mean_coherence > 0.35 ? "Moderate Parallelism" : "Low Ridge Coherence");
    }

    // 5. Quality & Defect Diagnostics Card
    if (el.overallQualityScore) setText(el.overallQualityScore, qScore);
    if (el.qualityTierBadge) setText(el.qualityTierBadge, `${qTier} Tier`);
    if (el.qUsableArea) {
      setText(el.qUsableArea, q.usable_area_percentage !== undefined ? `${q.usable_area_percentage.toFixed(1)}%` : "--");
    }

    const metrics = q.metrics || {};
    if (el.qClarity) setText(el.qClarity, metrics.clarity !== undefined ? `${metrics.clarity.toFixed(1)} / 100` : "--");
    if (el.qFreqCont) setText(el.qFreqCont, metrics.frequency_continuity !== undefined ? `${metrics.frequency_continuity.toFixed(1)} / 100` : "--");

    if (q.damage_detected) {
      const dmgTypes = (q.damage_types || []).join(", ") || "Physical Alteration";
      const dmgPct = q.damage_percentage !== undefined ? `${q.damage_percentage.toFixed(1)}%` : "";
      if (el.qDamageStatus) {
        setText(el.qDamageStatus, `${dmgTypes} (${dmgPct})`);
        el.qDamageStatus.style.color = "var(--rose-bright)";
      }

      if (el.qualityAlertBanner) {
        el.qualityAlertBanner.className = "quality-alert-banner danger";
        const alertIcon = el.qualityAlertBanner.querySelector(".alert-icon");
        if (alertIcon) setText(alertIcon, "⚠");
        const alertText = el.qualityAlertText || el.qualityAlertBanner.querySelector(".alert-text");
        if (alertText) setText(alertText, `Warning: ${dmgTypes} defect localized in impression. Special forensic audit required.`);
      }
    } else if (res.decision_gate === "ACCEPTED_WITH_WARNINGS") {
      if (el.qDamageStatus) {
        setText(el.qDamageStatus, "Intact (Marginal Warnings)");
        el.qDamageStatus.style.color = "var(--amber-bright)";
      }

      if (el.qualityAlertBanner) {
        el.qualityAlertBanner.className = "quality-alert-banner warning";
        const alertIcon = el.qualityAlertBanner.querySelector(".alert-icon");
        if (alertIcon) setText(alertIcon, "⚠");
        const alertText = el.qualityAlertText || el.qualityAlertBanner.querySelector(".alert-text");
        if (alertText) setText(alertText, q.quality_warning || "Impression accepted with operational cautions.");
      }
    } else {
      if (el.qDamageStatus) {
        setText(el.qDamageStatus, "Intact (Zero Defects)");
        el.qDamageStatus.style.color = "var(--emerald-bright)";
      }

      if (el.qualityAlertBanner) {
        el.qualityAlertBanner.className = "quality-alert-banner";
        const alertIcon = el.qualityAlertBanner.querySelector(".alert-icon");
        if (alertIcon) setText(alertIcon, "✓");
        const alertText = el.qualityAlertText || el.qualityAlertBanner.querySelector(".alert-text");
        if (alertText) setText(alertText, "Impression meets ISO/NIST biometric quality standards for automated filing.");
      }
    }

    // 6. Forensic Explanation (Safely guarded if card is present)
    if (el.explanationAccordion) {
      el.explanationAccordion.innerHTML = "";
      const explanationList = res.explanation || [];
      explanationList.forEach((item, idx) => {
        const stepCard = document.createElement("div");
        stepCard.className = "step-card";
        stepCard.innerHTML = `
          <div class="step-header">
            <span class="step-num-badge">${idx + 1}</span>
            <span class="step-title">${escapeHtml(item.title || item.step)}</span>
          </div>
          <div class="step-detail">${escapeHtml(item.detail || "")}</div>
        `;
        el.explanationAccordion.appendChild(stepCard);
      });
    }

    // Hide placeholder
    if (el.canvasPlaceholder) {
      el.canvasPlaceholder.classList.add("hidden");
    }
  }

  // ------------------------------------------------------------------------
  // HTML5 Canvas Multi-Layer Vector Renderer
  // ------------------------------------------------------------------------
  function drawFingerprint(targetCanvas, layerToggles) {
    if (!targetCanvas || !state.currentResult || !state.rawImage) return;

    const ctx = targetCanvas.getContext("2d");
    const img = state.rawImage;
    const w = img.width || 512;
    const h = img.height || 512;

    if (targetCanvas.width !== w || targetCanvas.height !== h) {
      targetCanvas.width = w;
      targetCanvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    const toggles = layerToggles || {};
    const showComposite = !!toggles.composite;
    const showDamage = toggles.damage !== false;
    const showStreamlines = toggles.streamlines !== false;
    const showCores = toggles.cores !== false;
    const showDeltas = toggles.deltas !== false;

    // MODE A: Pre-rendered Composite Overlay
    if (showComposite && state.compositeImage) {
      ctx.drawImage(state.compositeImage, 0, 0, w, h);
      return;
    }

    // MODE B: Dynamic Layered HTML5 Canvas Rendering
    // Layer 0: Original Grayscale Fingerprint Bitmap
    ctx.drawImage(img, 0, 0, w, h);

    // Layer 1: Damage / Alteration Mask
    const dmg = state.damageMaskImage || state.damageImage;
    if (showDamage && dmg) {
      ctx.drawImage(dmg, 0, 0, w, h);
    }

    const vis = state.currentResult.visualizations || {};

    // Layer 2: Ridge Flow Streamlines (Cyan)
    if (showStreamlines && vis.streamlines && vis.streamlines.length > 0) {
      ctx.save();
      ctx.strokeStyle = "rgba(6, 182, 212, 0.75)";
      ctx.lineWidth = 1.6;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      vis.streamlines.forEach((sl) => {
        if (sl.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(sl[0][0], sl[0][1]);
        for (let i = 1; i < sl.length; i++) {
          ctx.lineTo(sl[i][0], sl[i][1]);
        }
        ctx.stroke();
      });
      ctx.restore();
    }

    // Layer 3: Singular Points - Cores (Red Circle & Crosshair)
    if (showCores && vis.cores) {
      vis.cores.forEach((core) => {
        const cx = core.x !== undefined ? core.x : (Array.isArray(core) ? core[0] : null);
        const cy = core.y !== undefined ? core.y : (Array.isArray(core) ? core[1] : null);
        if (cx === null || cy === null || cx < 0 || cy < 0 || cx > w || cy > h) return;

        ctx.save();
        // Crisp vector ring in Red
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, Math.PI * 2);
        ctx.stroke();

        // Crosshair lines
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 14, cy);
        ctx.lineTo(cx + 14, cy);
        ctx.moveTo(cx, cy - 14);
        ctx.lineTo(cx, cy + 14);
        ctx.stroke();

        // Center white dot
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });
    }

    // Layer 4: Singular Points - Deltas (Blue Triradius Triangle)
    if (showDeltas && vis.deltas) {
      vis.deltas.forEach((delta) => {
        const dx = delta.x !== undefined ? delta.x : (Array.isArray(delta) ? delta[0] : null);
        const dy = delta.y !== undefined ? delta.y : (Array.isArray(delta) ? delta[1] : null);
        if (dx === null || dy === null || dx < 0 || dy < 0 || dx > w || dy > h) return;

        ctx.save();
        const r = 10;
        const p1 = [dx, dy - r];
        const p2 = [dx - r * 0.866, dy + r * 0.5];
        const p3 = [dx + r * 0.866, dy + r * 0.5];

        ctx.strokeStyle = "#3b82f6";
        ctx.fillStyle = "rgba(59, 130, 246, 0.35)";
        ctx.lineWidth = 2.2;

        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.lineTo(p3[0], p3[1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Center anchor dot
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });
    }
  }

  function renderCanvas() {
    const canvas = el.fingerprintCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!state.currentResult || !state.rawImage) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (el.canvasPlaceholder) el.canvasPlaceholder.classList.remove("hidden");
      return;
    }

    if (el.canvasPlaceholder) el.canvasPlaceholder.classList.add("hidden");
    drawFingerprint(canvas, {
      cores: el.toggleCores ? el.toggleCores.checked : true,
      deltas: el.toggleDeltas ? el.toggleDeltas.checked : true,
      streamlines: el.toggleStreamlines ? el.toggleStreamlines.checked : true,
      damage: el.toggleDamage ? el.toggleDamage.checked : true,
      composite: el.toggleComposite ? el.toggleComposite.checked : false,
    });
  }

  // ------------------------------------------------------------------------
  // Forensic Inspection Lightbox Modal Controls & Pan/Zoom Logic
  // ------------------------------------------------------------------------
  let modalZoom = 1.0;
  let modalPanX = 0;
  let modalPanY = 0;
  let isDraggingModal = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartPanX = 0;
  let dragStartPanY = 0;

  function setModalTransform(newZoom, newPanX, newPanY) {
    modalZoom = Math.max(0.2, Math.min(10.0, newZoom));
    modalPanX = newPanX;
    modalPanY = newPanY;

    if (el.modalCanvasContainer) {
      el.modalCanvasContainer.style.transform = `translate(${modalPanX}px, ${modalPanY}px) scale(${modalZoom})`;
    }
    if (el.modalZoomLevel) {
      setText(el.modalZoomLevel, `${Math.round(modalZoom * 100)}%`);
    }
  }

  function zoomModalAt(targetZoom, clientX, clientY) {
    const oldZoom = modalZoom;
    const newZoom = Math.max(0.2, Math.min(10.0, targetZoom));
    if (Math.abs(newZoom - oldZoom) < 0.0001) return;

    if (!el.modalViewport) return;
    const rect = el.modalViewport.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const mx = (clientX !== undefined ? clientX - rect.left : cx) - cx;
    const my = (clientY !== undefined ? clientY - rect.top : cy) - cy;

    const scaleRatio = newZoom / oldZoom;
    const newPanX = modalPanX - (mx - modalPanX) * (scaleRatio - 1);
    const newPanY = modalPanY - (my - modalPanY) * (scaleRatio - 1);

    setModalTransform(newZoom, newPanX, newPanY);
  }

  function fitModalToViewport() {
    if (!state.rawImage || !el.modalViewport) return;
    const vpRect = el.modalViewport.getBoundingClientRect();
    const vpW = vpRect.width || 800;
    const vpH = vpRect.height || 600;
    const pad = 40; // 20px padding around
    const availW = Math.max(100, vpW - pad);
    const availH = Math.max(100, vpH - pad);

    const imgW = state.rawImage.width || 512;
    const imgH = state.rawImage.height || 512;

    const scaleX = availW / imgW;
    const scaleY = availH / imgH;
    const fitScale = Math.min(scaleX, scaleY);

    setModalTransform(fitScale, 0, 0);
  }

  function renderModalCanvas() {
    if (!state.currentResult || !state.rawImage || !el.modalCanvas) return;
    drawFingerprint(el.modalCanvas, {
      cores: el.modalToggleCores ? el.modalToggleCores.checked : true,
      deltas: el.modalToggleDeltas ? el.modalToggleDeltas.checked : true,
      streamlines: el.modalToggleStreamlines ? el.modalToggleStreamlines.checked : true,
      damage: el.modalToggleDamage ? el.modalToggleDamage.checked : true,
      composite: false,
    });
  }

  function openImageModal() {
    if (!state.rawImage || !state.currentResult || !el.imageModal) return;

    // Enable all inspection toggles by default (Cores, Deltas, Streamlines, Damage)
    if (el.modalToggleCores) el.modalToggleCores.checked = true;
    if (el.modalToggleDeltas) el.modalToggleDeltas.checked = true;
    if (el.modalToggleStreamlines) el.modalToggleStreamlines.checked = true;
    if (el.modalToggleDamage) el.modalToggleDamage.checked = true;
    if (el.toggleCores) el.toggleCores.checked = true;
    if (el.toggleDeltas) el.toggleDeltas.checked = true;
    if (el.toggleStreamlines) el.toggleStreamlines.checked = true;
    if (el.toggleDamage) el.toggleDamage.checked = true;

    if (el.modalCanvas) {
      el.modalCanvas.width = state.rawImage.width || 512;
      el.modalCanvas.height = state.rawImage.height || 512;
    }

    if (el.modalImageMeta) {
      const filename = state.currentResult.filename || "impression.png";
      const w = el.modalCanvas ? el.modalCanvas.width : (state.rawImage.width || 512);
      const h = el.modalCanvas ? el.modalCanvas.height : (state.rawImage.height || 512);
      el.modalImageMeta.textContent = `${filename} • ${w} × ${h} px`;
    }

    renderModalCanvas();

    el.imageModal.classList.remove("hidden");
    fitModalToViewport();

    requestAnimationFrame(() => {
      fitModalToViewport();
    });
  }

  function closeImageModal() {
    if (el.imageModal) {
      el.imageModal.classList.add("hidden");
    }
    isDraggingModal = false;
    if (el.modalViewport) {
      el.modalViewport.classList.remove("is-dragging");
    }
  }

  // ------------------------------------------------------------------------
  // Reset Dashboard
  // ------------------------------------------------------------------------
  function resetDashboard() {
    state.currentResult = null;
    state.rawImage = null;
    state.compositeImage = null;
    state.damageImage = null;
    state.damageMaskImage = null;
    state.selectedFile = null;
    state.selectedSampleId = null;

    closeImageModal();
    closeFeedbackModal();
    closeVaultModal();
    if (el.modalCanvas) {
      const mctx = el.modalCanvas.getContext("2d");
      if (mctx) mctx.clearRect(0, 0, el.modalCanvas.width, el.modalCanvas.height);
    }
    if (el.modalImageMeta) {
      setText(el.modalImageMeta, "Native Resolution");
    }

    if (el.fileInput) el.fileInput.value = "";
    if (el.sampleSelect) el.sampleSelect.value = "";
    if (el.handSelect) el.handSelect.value = "";
    if (el.digitSelect) el.digitSelect.value = "";

    if (el.inputStatusBadge) {
      setText(el.inputStatusBadge, "Ready for Input");
      el.inputStatusBadge.className = "badge badge-info";
    }

    if (el.dropZone) {
      const p = el.dropZone.querySelector(".drop-text");
      if (p) p.innerHTML = "<strong>Drag & Drop</strong> fingerprint impression here or <strong>browse</strong>";
    }

    if (el.fingerprintCanvas) {
      const ctx = el.fingerprintCanvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, el.fingerprintCanvas.width, el.fingerprintCanvas.height);
    }
    if (el.canvasPlaceholder) el.canvasPlaceholder.classList.remove("hidden");

    if (el.chipFilename) setText(el.chipFilename, "No Image Loaded");
    if (el.chipResolution) setText(el.chipResolution, "-- × -- px");
    if (el.chipQuality) setText(el.chipQuality, "Q: --");
    if (el.chipGate) {
      setText(el.chipGate, "GATE: IDLE");
      el.chipGate.style.background = "";
      el.chipGate.style.color = "";
      el.chipGate.style.borderColor = "";
    }
    if (el.primaryClassTitle) setText(el.primaryClassTitle, "AWAITING IMPRESSION");
    if (el.nistSymbolBadge) setText(el.nistSymbolBadge, "[--]");
    if (el.primaryConfBadge) setText(el.primaryConfBadge, "0.0%");
    if (el.confidenceBarFill) el.confidenceBarFill.style.width = "0%";
    if (el.confidencePercentageText) setText(el.confidencePercentageText, "0.0%");
    if (el.classDistributionGrid) el.classDistributionGrid.innerHTML = "";

    if (el.subtypeStatusBadge) {
      setText(el.subtypeStatusBadge, "Idle");
      el.subtypeStatusBadge.className = "badge badge-info";
    }
    if (el.subtypeName) setText(el.subtypeName, "Awaiting Impression");
    if (el.subtypeCategory) setText(el.subtypeCategory, "Category: --");
    if (el.verifiabilityBadge) setText(el.verifiabilityBadge, "Pending");
    if (el.subtypeReasonText) setText(el.subtypeReasonText, "Upload an impression to evaluate dermatoglyphic pattern and landmarks.");

    if (el.pipelineLatency) setText(el.pipelineLatency, "Execution: -- ms");
    if (el.valCoresCount) setText(el.valCoresCount, "0");
    if (el.valCoresCoord) {
      setText(el.valCoresCoord, "No impression loaded");
      el.valCoresCoord.removeAttribute("title");
    }
    if (el.valDeltasCount) setText(el.valDeltasCount, "0");
    if (el.valDeltasCoord) {
      setText(el.valDeltasCoord, "No impression loaded");
      el.valDeltasCoord.removeAttribute("title");
    }
    if (el.valTruncation) {
      setText(el.valTruncation, "Idle");
      el.valTruncation.style.color = "";
    }
    if (el.valTruncationDetail) setText(el.valTruncationDetail, "--");
    if (el.valRidgeCount) setText(el.valRidgeCount, "--");
    if (el.valWavelength) setText(el.valWavelength, "--");
    if (el.valFrequency) setText(el.valFrequency, "--");
    if (el.valFlowDirection) setText(el.valFlowDirection, "--");
    if (el.valExitAngle) setText(el.valExitAngle, "--");
    if (el.valCoherence) setText(el.valCoherence, "--");
    if (el.valCoherenceTier) setText(el.valCoherenceTier, "--");

    if (el.overallQualityScore) setText(el.overallQualityScore, "--");
    if (el.qualityTierBadge) setText(el.qualityTierBadge, "Idle");
    if (el.qUsableArea) setText(el.qUsableArea, "--%");
    if (el.qClarity) setText(el.qClarity, "-- / 100");
    if (el.qFreqCont) setText(el.qFreqCont, "-- / 100");
    if (el.qDamageStatus) {
      setText(el.qDamageStatus, "--");
      el.qDamageStatus.style.color = "";
    }
    if (el.qualityAlertBanner) {
      el.qualityAlertBanner.className = "quality-alert-banner";
      const icon = el.qualityAlertBanner.querySelector(".alert-icon");
      if (icon) setText(icon, "ℹ");
      const alertText = el.qualityAlertText || el.qualityAlertBanner.querySelector(".alert-text");
      if (alertText) setText(alertText, "Upload or drag & drop a fingerprint impression to run forensic classification.");
    }

    if (el.explanationAccordion) el.explanationAccordion.innerHTML = "";
  }

  // ------------------------------------------------------------------------
  // False Prediction Reporting & Modal
  // ------------------------------------------------------------------------
  function openFeedbackModal() {
    if (!state.currentResult) {
      showToast("Please analyze a fingerprint before reporting a false prediction.", "warning");
      return;
    }

    // Reset feedback options
    const radios = document.querySelectorAll('input[name="feedbackCategoryOption"]');
    radios.forEach((r) => (r.checked = false));

    if (el.feedbackWhorlWrap) el.feedbackWhorlWrap.classList.add("hidden");
    if (el.feedbackCustomWrap) el.feedbackCustomWrap.classList.add("hidden");
    if (el.feedbackCustomText) el.feedbackCustomText.value = "";
    if (el.feedbackNotes) el.feedbackNotes.value = "";

    // Populate current AI prediction summary
    const cls = state.currentResult.classification || {};
    const sub = state.currentResult.subtype || {};
    const macroLabel = cls.anatomical_label || cls.primary_class || "Unknown";
    const subLabel = sub.subtype || "Unresolved";
    const confVal = cls.confidence ? (cls.confidence * 100).toFixed(1) : "--";
    if (el.feedbackAiSummary) {
      setText(el.feedbackAiSummary, `${macroLabel} (${subLabel}) — ${confVal}% Conf`);
    }

    if (el.feedbackModal) {
      el.feedbackModal.classList.remove("hidden");
    }
  }

  function closeFeedbackModal() {
    if (el.feedbackModal) {
      el.feedbackModal.classList.add("hidden");
    }
  }

  async function submitFeedbackRecord() {
    if (!state.currentResult) return;

    const selected = document.querySelector('input[name="feedbackCategoryOption"]:checked');
    if (!selected) {
      showToast("Please select the correct anatomical classification before saving.", "warning");
      return;
    }

    let category = selected.dataset.category || selected.value;
    let userCorrection = selected.value;

    if (selected.value === "Whorl") {
      userCorrection = el.feedbackWhorlSubtype ? el.feedbackWhorlSubtype.value : "Target Centric Whorl";
      category = "Whorl";
    } else if (selected.value === "Other") {
      const customVal = (el.feedbackCustomText ? el.feedbackCustomText.value : "").trim();
      if (!customVal) {
        showToast("Please enter the specific custom classification or subtype.", "warning");
        if (el.feedbackCustomText) el.feedbackCustomText.focus();
        return;
      }
      userCorrection = customVal;
      category = "Other";
    }

    // Extract image snapshot from result visualizations or canvas
    const vis = state.currentResult.visualizations || {};
    const imageBase64 = vis.original || vis.image_base64 || (el.fingerprintCanvas ? el.fingerprintCanvas.toDataURL("image/png") : null);

    const cls = state.currentResult.classification || {};
    const sub = state.currentResult.subtype || {};
    const q = state.currentResult.quality || {};
    const feat = state.currentResult.features || {};

    const aiPrediction = {
      primary_class: cls.primary_class || "",
      anatomical_label: cls.anatomical_label || "",
      subtype: sub.subtype || "",
      confidence: cls.confidence || 0,
      quality_score: q.overall_score || 0,
    };

    const metrics = {
      cores_count: feat.num_cores ?? (feat.cores ? feat.cores.length : 0),
      deltas_count: feat.num_deltas ?? (feat.deltas ? feat.deltas.length : 0),
      quality_score: q.overall_score || 0,
      usable_area: q.usable_area_percentage || 0,
      ridge_count: feat.ridge_count || 0,
    };

    const payload = {
      image_base64: imageBase64,
      filename: state.currentResult.filename || "impression.png",
      ai_prediction: aiPrediction,
      user_correction: userCorrection,
      category: category,
      notes: (el.feedbackNotes ? el.feedbackNotes.value : "").trim(),
      metrics: metrics,
      visualizations: state.currentResult.visualizations || null,
      analysis_result: state.currentResult || null,
    };

    if (el.btnSubmitFeedback) {
      el.btnSubmitFeedback.disabled = true;
      setText(el.btnSubmitFeedback, "Saving...");
    }

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Server error while saving correction.");
      }

      const resData = await res.json();
      updateVaultBadge(resData.total_corrections);
      closeFeedbackModal();
      showToast("Correction saved to Review Vault archive.", "success");
    } catch (err) {
      showToast("Error saving correction: " + err.message, "error");
    } finally {
      if (el.btnSubmitFeedback) {
        el.btnSubmitFeedback.disabled = false;
        setText(el.btnSubmitFeedback, "Save to Feedback Archive");
      }
    }
  }

  const submitFeedback = submitFeedbackRecord;

  // ------------------------------------------------------------------------
  // Correction Review Vault Drawer / Modal
  // ------------------------------------------------------------------------
  let currentVaultCorrections = [];
  let selectedVaultCategory = "All";
  let selectedCorrectionId = null;

  async function fetchVaultCount() {
    try {
      const res = await fetch("/api/feedback/count");
      if (!res.ok) return;
      const data = await res.json();
      updateVaultBadge(data.count);
    } catch (err) {
      console.warn("Could not fetch feedback count:", err);
    }
  }

  function updateVaultBadge(count) {
    if (el.vaultBadgeCount) {
      setText(el.vaultBadgeCount, count !== undefined ? String(count) : "0");
    }
  }

  function openVaultModal() {
    if (el.vaultModal) {
      el.vaultModal.classList.remove("hidden");
    }
    loadVaultData(selectedVaultCategory);
  }

  function closeVaultModal() {
    if (el.vaultModal) {
      el.vaultModal.classList.add("hidden");
    }
  }

  async function loadVaultData(category = selectedVaultCategory) {
    if (!el.vaultListContainer) return;
    el.vaultListContainer.innerHTML = '<div class="vault-no-records"><p>Loading correction records...</p></div>';

    try {
      let url = "/api/feedback";
      if (category && category !== "All") {
        url += `?category=${encodeURIComponent(category)}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load feedback records");
      const data = await res.json();
      currentVaultCorrections = data.corrections || [];
      renderVaultList(currentVaultCorrections);

      if (currentVaultCorrections.length > 0) {
        const matchExists = currentVaultCorrections.some((c) => c.id === selectedCorrectionId);
        const toSelect = matchExists ? selectedCorrectionId : currentVaultCorrections[0].id;
        selectVaultCorrection(toSelect);
      } else {
        renderVaultEmptyDetail();
      }
    } catch (err) {
      console.error("Vault load error:", err);
      el.vaultListContainer.innerHTML = `<div class="vault-no-records"><p>Error: ${escapeHtml(err.message)}</p></div>`;
      renderVaultEmptyDetail();
    }
  }

  let pendingVaultConfirmCallback = null;

  function showVaultConfirm(title, message, confirmText, confirmClass, onConfirm) {
    if (!el.vaultConfirmModal) {
      if (window.confirm(message)) onConfirm();
      return;
    }
    if (el.vaultConfirmTitle) setText(el.vaultConfirmTitle, title);
    if (el.vaultConfirmMessage) setText(el.vaultConfirmMessage, message);
    if (el.btnAcceptVaultConfirm) {
      setText(el.btnAcceptVaultConfirm, confirmText || "Confirm");
      el.btnAcceptVaultConfirm.className = `btn ${confirmClass || "btn-danger"} btn-sm`;
    }
    pendingVaultConfirmCallback = onConfirm;
    el.vaultConfirmModal.classList.remove("hidden");
  }

  function closeVaultConfirm() {
    if (el.vaultConfirmModal) {
      el.vaultConfirmModal.classList.add("hidden");
    }
    pendingVaultConfirmCallback = null;
  }

  function handleDownloadAllVault() {
    if (!currentVaultCorrections || currentVaultCorrections.length === 0) {
      showToast("Archive is empty", "warning");
      return;
    }
    showVaultConfirm(
      "Download Archive & Clear Queue?",
      "This will export all logged corrections and biometric images as a ZIP bundle, and clear the feedback queue.",
      "Download & Clear",
      "btn-primary",
      async () => {
        try {
          const link = document.createElement("a");
          link.href = "/api/feedback/export?clear=true&format=zip";
          link.download = "";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          currentVaultCorrections = [];
          selectedCorrectionId = null;
          renderVaultList([]);
          renderVaultEmptyDetail();
          await fetchVaultCount();
          showToast("Archive exported and cleared successfully.", "success");
        } catch (err) {
          showToast("Export failed: " + err.message, "danger");
        }
      }
    );
  }

  function handleDownloadSingleRecord(id) {
    if (!id) return;
    const link = document.createElement("a");
    link.href = `/api/feedback/${encodeURIComponent(id)}/download`;
    link.download = `correction_${id}_bundle.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function handleDeleteSingleRecord(id) {
    if (!id) return;
    showVaultConfirm(
      "Delete Correction Record?",
      "Delete this correction from archive? This action cannot be undone.",
      "Delete Record",
      "btn-danger",
      async () => {
        try {
          const res = await fetch(`/api/feedback/${encodeURIComponent(id)}`, {
            method: "DELETE",
          });
          if (!res.ok) {
            throw new Error(`Failed to delete record (HTTP ${res.status})`);
          }
          const data = await res.json();
          updateVaultBadge(data.remaining);

          currentVaultCorrections = currentVaultCorrections.filter((c) => c.id !== id);
          renderVaultList(currentVaultCorrections);

          if (selectedCorrectionId === id) {
            if (currentVaultCorrections.length > 0) {
              selectVaultCorrection(currentVaultCorrections[0].id);
            } else {
              selectedCorrectionId = null;
              renderVaultEmptyDetail();
            }
          }
          showToast("Correction record deleted.", "success");
        } catch (err) {
          showToast("Delete failed: " + err.message, "danger");
        }
      }
    );
  }

  function renderVaultList(corrections) {
    if (!el.vaultListContainer) return;
    if (!corrections || corrections.length === 0) {
      el.vaultListContainer.innerHTML = '<div class="vault-no-records"><p>No corrections logged for this filter category.</p></div>';
      return;
    }

    el.vaultListContainer.innerHTML = corrections
      .map((c) => {
        const isActive = c.id === selectedCorrectionId ? " active" : "";
        const ai = c.ai_prediction || {};
        const aiName = ai.anatomical_label || ai.primary_class || "AI";
        const userCorr = c.user_correction || "Verified";
        const confText = ai.confidence ? `${(ai.confidence * 100).toFixed(0)}% conf` : "";
        const imgSrc = c.image_url || (c.image_filename ? `/api/feedback/image/${c.image_filename}` : "");
        const timeFormatted = c.timestamp ? formatTimestamp(c.timestamp) : "";

        return `
          <div class="vault-card${isActive}" data-id="${c.id}">
            ${imgSrc ? `<img src="${imgSrc}" class="vault-card-thumb" alt="Thumbnail" />` : '<div class="vault-card-thumb"></div>'}
            <div class="vault-card-info">
              <div class="vault-card-top">
                <span class="vault-card-filename" title="${escapeHtml(c.filename || "")}">${escapeHtml(c.filename || "impression.png")}</span>
                <div class="vault-card-meta-right">
                  <span class="vault-card-time">${timeFormatted}</span>
                  <button type="button" class="vault-card-delete" title="Delete record" data-id="${c.id}">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </button>
                </div>
              </div>
              <div class="vault-card-diff">
                <span class="vault-diff-from">${escapeHtml(aiName)}</span>
                <span class="vault-diff-arrow">→</span>
                <span class="vault-diff-to">${escapeHtml(userCorr)}</span>
              </div>
              ${confText ? `<span class="vault-card-conf">${confText}</span>` : ""}
            </div>
          </div>
        `;
      })
      .join("");

    el.vaultListContainer.querySelectorAll(".vault-card").forEach((card) => {
      card.addEventListener("click", () => {
        selectVaultCorrection(card.dataset.id);
      });
    });

    el.vaultListContainer.querySelectorAll(".vault-card-delete").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = btn.dataset.id;
        if (id) handleDeleteSingleRecord(id);
      });
    });
  }

  function selectVaultCorrection(id) {
    selectedCorrectionId = id;
    if (el.vaultListContainer) {
      el.vaultListContainer.querySelectorAll(".vault-card").forEach((card) => {
        card.classList.toggle("active", card.dataset.id === id);
      });
    }

    const rec = currentVaultCorrections.find((c) => c.id === id);
    if (rec) {
      renderVaultDetail(rec);
    } else if (id) {
      fetch(`/api/feedback/${encodeURIComponent(id)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data) renderVaultDetail(data);
          else renderVaultEmptyDetail();
        })
        .catch(() => renderVaultEmptyDetail());
    } else {
      renderVaultEmptyDetail();
    }
  }

  function renderVaultDetail(c) {
    if (!el.vaultDetailContent || !el.vaultEmptyState) return;
    el.vaultEmptyState.classList.add("hidden");
    el.vaultDetailContent.classList.remove("hidden");

    const ai = c.ai_prediction || {};
    const metrics = c.metrics || {};
    let imgSrc = c.image_url || (c.image_filename ? `/api/feedback/image/${c.image_filename}` : "") || (c.visualizations && (c.visualizations.original || c.visualizations.image_base64)) || c.image_base64 || "";
    if (imgSrc && !imgSrc.startsWith("http") && !imgSrc.startsWith("/") && !imgSrc.startsWith("data:")) {
      imgSrc = `data:image/png;base64,${imgSrc}`;
    }
    const timeFormatted = c.timestamp ? formatTimestamp(c.timestamp) : "--";
    const qScore = metrics.quality_score !== undefined ? Number(metrics.quality_score).toFixed(1) : (ai.quality_score ? Number(ai.quality_score).toFixed(1) : "--");

    el.vaultDetailContent.innerHTML = `
      <div class="vault-detail-header">
        ${imgSrc ? `
          <div class="vault-detail-img-wrap" id="vaultDetailImgWrap" title="Click to inspect fingerprint impression, ridge flow streamlines, cores &amp; deltas">
            <img src="${imgSrc}" class="vault-detail-img" alt="Impression Snapshot" onerror="this.style.display='none'" />
            <span class="vault-img-hover-badge">🔍 Inspect Overlays &amp; Zoom</span>
          </div>
        ` : ""}
        <div class="vault-detail-meta">
          <div class="vault-detail-title">${escapeHtml(c.filename || "Impression Snapshot")}</div>
          <div class="vault-detail-subtitle">ID: ${escapeHtml(c.id)} • Category: ${escapeHtml(c.category || "Other")} • Logged: ${timeFormatted}</div>
        </div>
        <div class="vault-detail-actions">
          <button type="button" id="btnInspectVaultPrint" class="btn btn-primary btn-sm" title="Open full-screen forensic inspection with ridge flow lines, cores and deltas">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Inspect Print &amp; Overlays
          </button>
          <button type="button" id="btnDownloadSingleVault" class="btn btn-secondary btn-sm" title="Download record bundle as ZIP (JSON, PDF report, impression image)">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download Record
          </button>
          <button type="button" id="btnDeleteSingleVault" class="btn btn-ghost btn-sm text-danger" title="Delete record">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Delete Record
          </button>
        </div>
      </div>

      <div class="diff-split">
        <!-- Left Sub-Card: Original AI Prediction -->
        <div class="diff-column-ai">
          <div class="diff-column-header">
            <span class="diff-column-title">Original AI Prediction</span>
            <span class="badge badge-warning">${ai.confidence ? (ai.confidence * 100).toFixed(1) + "% Conf" : "--"}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Macro Pattern:</span>
            <span class="diff-item-val">${escapeHtml(ai.anatomical_label || ai.primary_class || "--")}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Subtype:</span>
            <span class="diff-item-val">${escapeHtml(ai.subtype || "--")}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Landmark Cores:</span>
            <span class="diff-item-val">${metrics.cores_count ?? "--"}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Delta Triradii:</span>
            <span class="diff-item-val">${metrics.deltas_count ?? "--"}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Quality Score (Q):</span>
            <span class="diff-item-val">${qScore}</span>
          </div>
        </div>

        <!-- Right Sub-Card: Human Verified Correction -->
        <div class="diff-column-human">
          <div class="diff-column-header">
            <span class="diff-column-title">Human Verified Correction</span>
            <span class="badge badge-status">Examiner Audited</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Verified Class / Subtype:</span>
            <span class="diff-item-val" style="color: var(--emerald-bright); font-size: 0.95rem;">${escapeHtml(c.user_correction || "--")}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Assigned Category:</span>
            <span class="diff-item-val">${escapeHtml(c.category || "--")}</span>
          </div>
          <div class="diff-item">
            <span class="diff-item-label">Timestamp:</span>
            <span class="diff-item-val">${timeFormatted}</span>
          </div>
          <div class="diff-notes-block">
            <div class="diff-notes-title">Examiner Explanation & Notes:</div>
            <div>${c.notes ? escapeHtml(c.notes) : '<em style="color: var(--text-muted);">No notes recorded.</em>'}</div>
          </div>
        </div>
      </div>
    `;

    const imgWrap = el.vaultDetailContent.querySelector("#vaultDetailImgWrap");
    if (imgWrap) {
      imgWrap.addEventListener("click", () => openVaultInspectionModal(c));
    }
    const btnInspect = el.vaultDetailContent.querySelector("#btnInspectVaultPrint");
    if (btnInspect) {
      btnInspect.addEventListener("click", () => openVaultInspectionModal(c));
    }
    const btnDown = el.vaultDetailContent.querySelector("#btnDownloadSingleVault");
    if (btnDown) {
      btnDown.addEventListener("click", () => handleDownloadSingleRecord(c.id));
    }
    const btnDel = el.vaultDetailContent.querySelector("#btnDeleteSingleVault");
    if (btnDel) {
      btnDel.addEventListener("click", () => handleDeleteSingleRecord(c.id));
    }
  }

  async function openVaultInspectionModal(c) {
    if (!c) return;
    const vis = c.visualizations || (c.analysis_result && c.analysis_result.visualizations);
    const hasStoredVis = Boolean(vis && (
      (vis.streamlines && vis.streamlines.length > 0) ||
      (Array.isArray(vis.cores)) ||
      (Array.isArray(vis.deltas)) ||
      vis.composite_overlay_base64 ||
      vis.damage_mask_base64 ||
      vis.image_base64 ||
      vis.original
    ));

    if (hasStoredVis) {
      const mergedVis = Object.assign({}, vis, {
        cores: (vis && vis.cores) || (c.analysis_result && c.analysis_result.features && c.analysis_result.features.cores) || [],
        deltas: (vis && vis.deltas) || (c.analysis_result && c.analysis_result.features && c.analysis_result.features.deltas) || [],
        streamlines: (vis && vis.streamlines) || [],
      });
      state.currentResult = c.analysis_result || Object.assign({}, c, {
        filename: c.filename || (c.analysis_result && c.analysis_result.filename) || "impression.png",
        input_shape: (c.analysis_result && c.analysis_result.input_shape) || [512, 512],
        visualizations: mergedVis,
        classification: c.ai_prediction || {},
        quality: c.metrics || {},
        features: {
          cores: mergedVis.cores,
          deltas: mergedVis.deltas,
        },
      });
      if (!state.currentResult.visualizations) {
        state.currentResult.visualizations = mergedVis;
      }

      let rawSrc = c.image_url || (vis && (vis.original || vis.image_base64)) || c.image_base64 || (c.image_filename ? `/api/feedback/image/${c.image_filename}` : "");
      if (rawSrc && !rawSrc.startsWith("http") && !rawSrc.startsWith("/") && !rawSrc.startsWith("data:")) {
        rawSrc = `data:image/png;base64,${rawSrc}`;
      }

      const compositeSrc = vis && vis.composite_overlay_base64;
      const damageSrc = vis && vis.damage_mask_base64;

      const [raw, composite, damage] = await Promise.all([
        loadImagePromise(rawSrc),
        loadImagePromise(compositeSrc),
        loadImagePromise(damageSrc),
      ]);
      state.rawImage = raw;
      state.compositeImage = composite;
      state.damageImage = damage;
      state.damageMaskImage = damage;

      if (!state.rawImage) {
        showToast("Could not load impression image for inspection.", "warning");
        return;
      }
      openImageModal();
      return;
    }

    let imgSrc = c.image_url || (c.image_filename ? `/api/feedback/image/${c.image_filename}` : "") || c.image_base64;
    if (imgSrc && !imgSrc.startsWith("http") && !imgSrc.startsWith("/") && !imgSrc.startsWith("data:")) {
      imgSrc = `data:image/png;base64,${imgSrc}`;
    }

    if (imgSrc) {
      showToast("Analyzing impression overlays on-the-fly...", "info");
      try {
        const resp = await fetch(imgSrc);
        if (!resp.ok) throw new Error("Could not fetch impression image.");
        const blob = await resp.blob();
        const formData = new FormData();
        formData.append("file", blob, c.filename || "impression.png");
        const analyzeRes = await fetch("/api/analyze", {
          method: "POST",
          body: formData,
        });
        if (!analyzeRes.ok) {
          const errJson = await analyzeRes.json().catch(() => ({}));
          throw new Error(errJson.detail || "Analysis request failed.");
        }
        const result = await analyzeRes.json();
        state.currentResult = result;
        await loadImages(result.visualizations);
        state.damageMaskImage = state.damageImage;
        openImageModal();
        showToast("Forensic overlays & streamlines loaded.", "success");
      } catch (err) {
        console.error("Failed to analyze vault image:", err);
        showToast("Could not inspect overlays: " + err.message, "danger");
      }
    } else {
      showToast("No impression image available for inspection.", "warning");
    }
  }

  function renderVaultEmptyDetail() {
    if (el.vaultEmptyState) {
      el.vaultEmptyState.classList.remove("hidden");
      const p = el.vaultEmptyState.querySelector("p");
      if (p) {
        if (!currentVaultCorrections || currentVaultCorrections.length === 0) {
          setText(p, "No corrections logged yet. Use 'False Prediction?' to record discrepancies.");
        } else {
          setText(p, "Select a correction from the archive list to view side-by-side comparison");
        }
      }
    }
    if (el.vaultDetailContent) el.vaultDetailContent.classList.add("hidden");
  }

  function formatTimestamp(isoStr) {
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    } catch {
      return isoStr;
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showToast(message, type = "success") {
    let container = document.getElementById("toastContainer");
    if (!container) {
      container = document.createElement("div");
      container.id = "toastContainer";
      container.className = "toast-container";
      document.body.appendChild(container);
    }
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    setText(toast, message);
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      setTimeout(() => {
        if (toast.parentElement) toast.parentElement.removeChild(toast);
      }, 250);
    }, 3500);
  }

  // Run on page load
  window.addEventListener("DOMContentLoaded", init);
})();
