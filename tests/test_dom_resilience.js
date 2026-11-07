/**
 * Node.js DOM Mock Resilience Verification
 * Verifies that app.js never throws 'Cannot set properties of null' or any DOM error
 * under various missing element conditions (including the user's reported bug).
 *
 * Fully asynchronous with an element instance cache so that DOM event listeners
 * registered by app.js are genuinely invoked by dispatchEvent / click.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const appJsCode = fs.readFileSync(path.join(__dirname, '../src/app/static/app.js'), 'utf8');

function createMockElement(id = '', tag = 'div') {
  const listeners = {};
  const el = {
    id: id,
    tagName: tag.toUpperCase(),
    textContent: '',
    innerHTML: '',
    value: '',
    checked: true,
    disabled: false,
    className: '',
    style: {},
    dataset: {},
    classList: {
      _classes: new Set(),
      add: function (...names) { names.forEach(n => this._classes.add(n)); },
      remove: function (...names) { names.forEach(n => this._classes.delete(n)); },
      contains: function (name) { return this._classes.has(name); },
      toggle: function (name, force) {
        if (force === undefined) {
          if (this._classes.has(name)) this._classes.delete(name);
          else this._classes.add(name);
        } else if (force) {
          this._classes.add(name);
        } else {
          this._classes.delete(name);
        }
      }
    },
    addEventListener: function (evt, fn) {
      if (!listeners[evt]) listeners[evt] = [];
      listeners[evt].push(fn);
    },
    dispatchEvent: function (evt, data = {}) {
      if (listeners[evt]) {
        listeners[evt].forEach(fn => fn(data));
      }
    },
    removeAttribute: function () {},
    setAttribute: function () {},
    getAttribute: function () { return null; },
    focus: function () {},
    click: function () {
      if (listeners['click']) {
        listeners['click'].forEach(fn => fn({ preventDefault: () => {}, target: el, stopPropagation: () => {} }));
      }
    },
    getContext: function () {
      return {
        clearRect: () => {},
        drawImage: () => {},
        beginPath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        arc: () => {},
        stroke: () => {},
        fill: () => {},
        save: () => {},
        restore: () => {},
        closePath: () => {},
      };
    },
    querySelector: function (selector) {
      if (selector === '.alert-icon') {
        return createMockElement('', 'span');
      }
      if (selector === '.alert-text') {
        return createMockElement('', 'span');
      }
      if (selector === '.drop-text') {
        return createMockElement('', 'p');
      }
      if (selector.startsWith('#')) {
        return createMockElement(selector.slice(1), 'div');
      }
      return createMockElement('', 'div');
    },
    querySelectorAll: function () {
      return [];
    },
    appendChild: function (child) {
      return child;
    },
    removeChild: function (child) {
      return child;
    },
    getBoundingClientRect: function () {
      return { left: 0, top: 0, width: 800, height: 600 };
    },
    toDataURL: function () {
      return 'data:image/png;base64,mock';
    }
  };
  return el;
}

const mockAnalysisResult = {
  filename: "sample_print.png",
  input_shape: [512, 512],
  classification: {
    primary_class: "W",
    anatomical_label: "Whorl",
    confidence: 0.985,
    probabilities: { W: 0.985, L: 0.005, R: 0.005, A: 0.003, T: 0.002 }
  },
  subtype: {
    subtype: "Plain Whorl",
    primary_category: "Whorl",
    status: "Deterministic",
    is_uncertain: false,
    forensic_verifiability: "Deterministic (Strict)",
    reason: "Bilateral triradii detected."
  },
  features: {
    num_cores: 2,
    num_deltas: 2,
    cores: [{ x: 250, y: 250 }, { x: 260, y: 270 }],
    deltas: [{ x: 120, y: 380 }, { x: 390, y: 375 }],
    delta_truncated: false,
    expected_delta_sector: null,
    ridge_count: 14.5,
    median_wavelength: 9.8,
    median_frequency: 0.102,
    flow_direction: "Whorl Circular Flow",
    exit_angle_deg: 135.0,
    mean_coherence: 0.82
  },
  quality: {
    overall_score: 82.5,
    quality_tier: "Good",
    usable_area_percentage: 89.2,
    metrics: { clarity: 85.0, frequency_continuity: 80.0 },
    damage_detected: true,
    damage_types: ["Obliteration"],
    damage_percentage: 4.2,
    quality_warning: "Minor scarring detected"
  },
  decision_gate: "ACCEPTED_WITH_WARNINGS",
  execution_time_ms: { total_ms: 45.2 },
  visualizations: {
    image_base64: "data:image/png;base64,mock",
    composite_overlay_base64: "data:image/png;base64,mock",
    damage_mask_base64: "data:image/png;base64,mock",
    streamlines: [[[10, 10], [20, 20]]],
    cores: [{ x: 250, y: 250 }],
    deltas: [{ x: 120, y: 380 }]
  },
  explanation: [
    { step: "Quality Assessment", title: "Quality Check", detail: "Good quality impression." },
    { step: "Classification", title: "Macro Pattern", detail: "Identified as Whorl." }
  ]
};

async function runTestScenario(name, elementFilter, querySelectorMock) {
  console.log(`--- Running Test Scenario: ${name} ---`);

  // Element Cache ensures app.js and the test harness operate on the exact same element instances
  const elemCache = new Map();
  const getElementByIdMock = (id) => {
    if (elementFilter && elementFilter(id) === null) {
      return null;
    }
    if (!elemCache.has(id)) {
      elemCache.set(id, createMockElement(id));
    }
    return elemCache.get(id);
  };

  const alerts = [];
  const consoleErrors = [];
  const mockWindowListeners = {};

  const sandbox = {
    console: {
      log: (...args) => {},
      warn: (...args) => {},
      error: (...args) => {
        consoleErrors.push(args.join(" "));
      }
    },
    setTimeout: (fn, delay = 0) => setTimeout(fn, Math.min(delay, 5)),
    clearTimeout: (id) => clearTimeout(id),
    requestAnimationFrame: (fn) => setTimeout(fn, 1),
    alert: (msg) => {
      alerts.push(String(msg));
    },
    Blob: class {
      constructor(parts = [], opts = {}) {
        this.parts = parts;
        this.opts = opts;
      }
    },
    URL: {
      createObjectURL: () => 'blob:mock',
      revokeObjectURL: () => {}
    },
    FormData: class {
      constructor() {
        this._entries = {};
      }
      append(key, val) {
        this._entries[key] = val;
      }
    },
    Image: class {
      constructor() {
        this.width = 512;
        this.height = 512;
        setTimeout(() => {
          if (this.onload) this.onload();
        }, 2);
      }
    },
    fetch: (url) => Promise.resolve({
      ok: true,
      status: 200,
      json: () => {
        if (typeof url === 'string' && url.includes('/api/health')) {
          return Promise.resolve({ status: 'ok', mps_active: true, device: 'mps' });
        }
        if (typeof url === 'string' && url.includes('/api/sample-gallery')) {
          return Promise.resolve({
            samples: [
              { id: 'nist_whorl', name: 'NIST Whorl', category: 'NIST Clean', expected_class: 'W', hand: 'Right', digit_code: 'R1' }
            ]
          });
        }
        if (typeof url === 'string' && url.includes('/api/feedback/count')) {
          return Promise.resolve({ count: 5 });
        }
        return Promise.resolve(mockAnalysisResult);
      },
      blob: () => Promise.resolve(new sandbox.Blob()),
      text: () => Promise.resolve('')
    }),
    window: {
      addEventListener: (evt, fn) => { mockWindowListeners[evt] = fn; },
      print: () => {},
      confirm: () => true
    },
    document: {
      getElementById: getElementByIdMock,
      querySelector: querySelectorMock || (() => null),
      querySelectorAll: () => [],
      createElement: (tag) => createMockElement('', tag),
      body: createMockElement('body', 'body')
    }
  };

  vm.createContext(sandbox);

  // 1. Evaluate app.js in sandbox
  try {
    vm.runInContext(appJsCode, sandbox);
  } catch (err) {
    throw new Error(`Failed during initial evaluation of app.js in scenario "${name}": ${err.stack}`);
  }

  // 2. Trigger DOMContentLoaded and allow initial async calls (init / auto-analyze) to settle
  if (mockWindowListeners['DOMContentLoaded']) {
    try {
      mockWindowListeners['DOMContentLoaded']();
    } catch (err) {
      throw new Error(`Failed during DOMContentLoaded in scenario "${name}": ${err.stack}`);
    }
  }

  // Wait for initial init() async promises
  await new Promise(r => setTimeout(r, 25));

  // 3. Simulate user actions: File selection via drop
  const dropZone = getElementByIdMock('dropZone');
  if (dropZone) {
    try {
      dropZone.dispatchEvent('drop', {
        preventDefault: () => {},
        dataTransfer: { files: [{ name: 'uploaded_print.png', size: 102400 }] }
      });
    } catch (err) {
      throw new Error(`Failed on dropZone 'drop' in scenario "${name}": ${err.stack}`);
    }
  }

  // Wait for drop handling and analysis to settle
  await new Promise(r => setTimeout(r, 25));

  // 4. Simulate file selection via file input
  const fileInput = getElementByIdMock('fileInput');
  if (fileInput) {
    try {
      fileInput.dispatchEvent('change', {
        target: { files: [{ name: 'selected_print.bmp', size: 204800 }] }
      });
    } catch (err) {
      throw new Error(`Failed on fileInput 'change' in scenario "${name}": ${err.stack}`);
    }
  }

  // Wait for fileInput analysis to settle
  await new Promise(r => setTimeout(r, 25));

  // 5. Simulate sample dropdown change and load
  const sampleSelect = getElementByIdMock('sampleSelect');
  if (sampleSelect) {
    sampleSelect.value = 'nist_whorl';
    sampleSelect.selectedIndex = 0;
    sampleSelect.options = [{ value: 'nist_whorl', dataset: { hand: 'Right', digit: 'R1' } }];
    try {
      sampleSelect.dispatchEvent('change');
    } catch (err) {
      throw new Error(`Failed on sampleSelect 'change' in scenario "${name}": ${err.stack}`);
    }
  }

  const btnLoadSample = getElementByIdMock('btnLoadSample');
  if (btnLoadSample && sampleSelect) {
    try {
      btnLoadSample.click();
    } catch (err) {
      throw new Error(`Failed on btnLoadSample 'click' in scenario "${name}": ${err.stack}`);
    }
  }

  await new Promise(r => setTimeout(r, 25));

  // 6. Simulate Analyze Button
  const btnAnalyze = getElementByIdMock('btnAnalyze');
  if (btnAnalyze) {
    try {
      btnAnalyze.click();
    } catch (err) {
      throw new Error(`Failed on btnAnalyze 'click' in scenario "${name}": ${err.stack}`);
    }
  }

  await new Promise(r => setTimeout(r, 25));

  // 7. Simulate Reset View
  const btnReset = getElementByIdMock('btnResetView');
  if (btnReset) {
    try {
      btnReset.click();
    } catch (err) {
      throw new Error(`Failed on btnResetView 'click' in scenario "${name}": ${err.stack}`);
    }
  }

  // 8. Simulate zoom toolbar buttons
  ['btnZoomFit', 'btnZoom100', 'modalBtnZoomIn', 'modalBtnZoomOut', 'modalBtnFit', 'modalBtn100'].forEach(btnId => {
    const b = getElementByIdMock(btnId);
    if (b) {
      try {
        b.click();
      } catch (err) {
        throw new Error(`Failed on ${btnId} 'click' in scenario "${name}": ${err.stack}`);
      }
    }
  });

  // 9. Verify no alerts reported errors
  const fatalAlerts = alerts.filter(a => a.toLowerCase().includes('error') || a.toLowerCase().includes('cannot set') || a.toLowerCase().includes('cannot read'));
  if (fatalAlerts.length > 0) {
    throw new Error(`Scenario "${name}" triggered fatal alert error(s): ${fatalAlerts.join('; ')}`);
  }

  // 10. Verify no console errors reported null property access
  const fatalConsoleErrors = consoleErrors.filter(e => e.includes('TypeError') || e.includes('Cannot set properties of null') || e.includes('Cannot read properties of null'));
  if (fatalConsoleErrors.length > 0) {
    throw new Error(`Scenario "${name}" triggered fatal console error(s): ${fatalConsoleErrors.join('; ')}`);
  }

  console.log(`[PASS] Scenario: ${name}`);
}

(async () => {
  try {
    // 1. Scenario: Real DOM reproduction with missing #inputStatusBadge & #sampleSelect (the reported bug)
    await runTestScenario("Missing inputStatusBadge and sampleSelect (Current Workstation)", (id) => {
      if (id === 'inputStatusBadge' || id === 'sampleSelect' || id === 'btnLoadSample' || id === 'btnUploadImage') {
        return null;
      }
      return undefined;
    });

    // 2. Scenario: Pathology - ALL elements return null (Headless / Total Reskin)
    await runTestScenario("Completely Null/Empty DOM (Headless or Total Reskin)", (id) => {
      return null;
    });

    // 3. Scenario: qualityAlertBanner exists but .alert-icon inside it is null
    await runTestScenario("Missing .alert-icon inside qualityAlertBanner", (id) => {
      if (id === 'inputStatusBadge') return null;
      return undefined;
    }, (sel) => {
      if (sel === '.alert-icon' || sel === '.alert-text') return null;
      return createMockElement('', 'div');
    });

    // 4. Scenario: All metric chips and value fields return null
    await runTestScenario("Missing All Data/Metric Chips", (id) => {
      const chips = [
        'chipFilename', 'chipResolution', 'chipQuality', 'chipGate',
        'primaryClassTitle', 'nistSymbolBadge', 'primaryConfBadge', 'confidenceBarFill',
        'confidencePercentageText', 'valCoresCount', 'valCoresCoord', 'valDeltasCount',
        'valDeltasCoord', 'valTruncation', 'valTruncationDetail', 'valRidgeCount',
        'valWavelength', 'valFrequency', 'valFlowDirection', 'valExitAngle',
        'valCoherence', 'valCoherenceTier', 'overallQualityScore', 'qualityTierBadge',
        'qUsableArea', 'qClarity', 'qFreqCont', 'qDamageStatus'
      ];
      if (chips.includes(id)) return null;
      return undefined;
    });

    // 5. Scenario: Missing canvas and modal elements
    await runTestScenario("Missing Canvas and Inspection Modal Elements", (id) => {
      const canvasElements = [
        'fingerprintCanvas', 'canvasOverlayPlaceholder', 'canvasViewport',
        'imageModal', 'modalCanvas', 'modalViewport', 'modalCanvasContainer', 'modalZoomLevel'
      ];
      if (canvasElements.includes(id)) return null;
      return undefined;
    });

    // 6. Scenario: Missing Feedback and Review Vault Elements
    await runTestScenario("Missing Feedback and Review Vault Elements", (id) => {
      const feedbackElements = [
        'btnFalsePrediction', 'feedbackModal', 'feedbackAiSummary',
        'btnSubmitFeedback', 'btnOpenVault', 'vaultBadgeCount', 'vaultModal',
        'vaultListContainer', 'vaultEmptyState', 'vaultDetailContent'
      ];
      if (feedbackElements.includes(id)) return null;
      return undefined;
    });

    console.log("\nALL NODE.JS DOM RESILIENCE TESTS PASSED CLEANLY!");
    process.exit(0);
  } catch (err) {
    console.error("\nTEST SUITE FAILED:", err);
    process.exit(1);
  }
})();
