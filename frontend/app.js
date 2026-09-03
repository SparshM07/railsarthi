/**
 * RailsArthi - Intelligent Railway Delay & Real-Time ETA Platform
 * Handles live telemetry polling, Leaflet map rendering, cascading station timelines,
 * and end-to-end journey delay simulations.
 */

// State
let map = null;
let trainMarker = null;
let routePolyline = null;
let stationMarkersLayer = null;
let renderedRouteSignature = null;
let currentTrainData = null;
let autoRefreshTimer = null;
let isAutoRefreshActive = true;
let availableCatalog = [];
let lastAlertedDelay = null;
let activePredictionSeq = 0;

// DOM Elements
const trainInput = document.getElementById("train-input");
const trainSearchForm = document.getElementById("train-search-form");
const quickTrainPills = document.getElementById("quick-train-pills");
const liveIstClock = document.getElementById("live-ist-clock");
const toggleAutoRefresh = document.getElementById("toggle-auto-refresh");
const systemStatusBadge = document.getElementById("system-status-badge");
const providerModeText = document.getElementById("provider-mode-text");
const btnRefresh = document.getElementById("btn-refresh");
const alertButton = document.getElementById("btn-enable-alerts");

function showToast(message, tone = "info") {
  const toast = document.getElementById("app-toast");
  if (!toast) return;
  toast.textContent = message;
  toast.className = `app-toast app-toast-${tone} app-toast-visible`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("app-toast-visible"), 5000);
}

function formatDelay(minutes) {
  const totalMinutes = Math.max(0, Math.round(Number(minutes) || 0));
  if (totalMinutes < 60) return `${totalMinutes} min`;
  const hours = Math.floor(totalMinutes / 60);
  const remainder = totalMinutes % 60;
  return `${String(hours).padStart(2, "0")}:${String(remainder).padStart(2, "0")} hrs`;
}

function alertsEnabled() {
  const val = localStorage.getItem("railsarthi-alerts") || localStorage.getItem("railpulse-alerts");
  return val === "enabled" && Notification.permission === "granted";
}

function refreshAlertButton() {
  if (!alertButton) return;
  const enabled = alertsEnabled();
  alertButton.textContent = enabled ? "Alerts enabled" : "Enable alerts";
  alertButton.setAttribute("aria-pressed", String(enabled));
}

async function enableAlerts() {
  if (!("Notification" in window)) {
    showToast("Browser notifications are not supported here.", "warning");
    return;
  }
  const permission = await Notification.requestPermission();
  if (permission === "granted") {
    localStorage.setItem("railsarthi-alerts", "enabled");
    localStorage.removeItem("railpulse-alerts");
    showToast("Delay alerts enabled for this browser.", "success");
  } else {
    localStorage.removeItem("railsarthi-alerts");
    localStorage.removeItem("railpulse-alerts");
    showToast("Notifications were not enabled. You can change this in browser settings.", "warning");
  }
  refreshAlertButton();
}

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  initClock();
  initTabs();
  initMap();
  refreshAlertButton();
  alertButton?.addEventListener("click", enableAlerts);
  await loadTrainCatalog();
  await checkHealth();
  await executePrediction(12919); // Default train
  setupAutoRefresh();
  setupJourneySimulation();
});

// Real-time IST Clock
function initClock() {
  function update() {
    const now = new Date();
    const istString = now.toLocaleTimeString("en-IN", {
      timeZone: "Asia/Kolkata",
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    if (liveIstClock) liveIstClock.textContent = `${istString} IST`;
  }
  update();
  setInterval(update, 1000);
}

// Navigation Tabs
function initTabs() {
  const tabs = {
    "tab-live": "view-live",
    "tab-journey": "view-journey",
    "tab-model": "view-model",
  };

  Object.entries(tabs).forEach(([tabId, viewId]) => {
    const tabBtn = document.getElementById(tabId);
    if (!tabBtn) return;
    tabBtn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((btn) => {
        btn.classList.remove("text-white", "bg-cyber-surface", "border", "border-white/10", "shadow-sm");
        btn.classList.add("text-slate-400");
      });
      tabBtn.classList.remove("text-slate-400");
      tabBtn.classList.add("text-white", "bg-cyber-surface", "border", "border-white/10", "shadow-sm");

      document.querySelectorAll(".tab-view").forEach((view) => view.classList.add("hidden"));
      const targetView = document.getElementById(viewId);
      if (targetView) targetView.classList.remove("hidden");

      if (tabId === "tab-live" && map) {
        setTimeout(() => map.invalidateSize(), 150);
      }
    });
  });
}

// Leaflet Map Initialization with 100% Free, Keyless Dark Tiles
function initMap() {
  const mapElement = document.getElementById("map");
  if (!mapElement) return;

  // Center on Central India
  map = L.map("map", {
    center: [23.5937, 78.9629],
    zoom: 5,
    zoomControl: true,
    preferCanvas: true,
    zoomSnap: 0.5,
    zoomDelta: 1,
    wheelPxPerZoomLevel: 45,
    inertia: true,
    inertiaDeceleration: 2500,
    tap: true,
    touchZoom: true,
  });

  // A page scroll should not unexpectedly zoom the map. Users can scroll the
  // map when it is focused, while touch gestures remain enabled on mobile.
  map.scrollWheelZoom.disable();
  mapElement.addEventListener("mouseenter", () => map.scrollWheelZoom.enable());
  mapElement.addEventListener("mouseleave", () => map.scrollWheelZoom.disable());
  mapElement.addEventListener("focusin", () => map.scrollWheelZoom.enable());
  mapElement.addEventListener("focusout", () => map.scrollWheelZoom.disable());
  map.on("zoomend", updateTrainLabelVisibility);

  // Free Esri Dark Gray Base (No API key required, zero watermark)
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
    attribution: '&copy; Esri &bull; Indian Railways Network',
    maxZoom: 16,
  }).addTo(map);

  // Free Esri Reference Labels Layer
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}", {
    attribution: '',
    maxZoom: 16,
    opacity: 0.8,
  }).addTo(map);

  stationMarkersLayer = L.layerGroup().addTo(map);

  document.getElementById("btn-fit-route")?.addEventListener("click", () => {
    if (routePolyline) map.fitBounds(routePolyline.getBounds(), { padding: [40, 40] });
  });

  document.getElementById("btn-focus-train")?.addEventListener("click", () => {
    if (trainMarker) {
      map.setView(trainMarker.getLatLng(), 8, { animate: true });
    }
  });
}

// Load Train Catalog
async function loadTrainCatalog() {
  try {
    const res = await fetch("/trains");
    if (!res.ok) return;
    const data = await res.json();
    availableCatalog = data.catalog || [];
    renderQuickPills(availableCatalog);
  } catch (e) {
    console.warn("Could not fetch train catalog:", e);
  }
}

function renderQuickPills(catalog) {
  if (!quickTrainPills) return;
  quickTrainPills.innerHTML = "";

  catalog.forEach((item) => {
    const pill = document.createElement("button");
    pill.className =
      "px-2.5 py-1 rounded-lg bg-cyber-surface hover:bg-cyber-card border border-cyber-border text-xs text-slate-300 hover:text-cyber-teal whitespace-nowrap transition-all flex items-center gap-1.5 active:scale-95";
    pill.innerHTML = `<span class="font-mono font-bold text-cyber-teal">${item.train_number}</span> <span>${item.train_name.split(" ")[0]}</span>`;
    pill.addEventListener("click", () => {
      if (trainInput) trainInput.value = item.train_number;
      executePrediction(item.train_number);
    });
    quickTrainPills.appendChild(pill);
  });
}

// Health Check
async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      if (providerModeText) {
        providerModeText.textContent = data.provider_mode === "SIMULATED" ? "SIMULATION TELEMETRY" : "LIVE RAIL TELEMETRY";
      }
    }
  } catch (e) {
    if (providerModeText) providerModeText.textContent = "LIVE RAIL TELEMETRY";
  }
}

// Form Search Listener
trainSearchForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  const trainNo = parseInt(trainInput.value.trim(), 10);
  if (trainNo >= 1 && trainNo <= 99999) {
    executePrediction(trainNo);
  }
});

btnRefresh?.addEventListener("click", () => {
  if (currentTrainData) {
    executePrediction(currentTrainData.train);
  }
});

// Auto-Refresh
let resetAutoRefreshTimer = null;
function setupAutoRefresh() {
  function startTimer() {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = setInterval(() => {
      if (isAutoRefreshActive && currentTrainData) {
        executePrediction(currentTrainData.train, true);
      }
    }, 15000);
  }

  resetAutoRefreshTimer = startTimer;

  toggleAutoRefresh?.addEventListener("click", () => {
    isAutoRefreshActive = !isAutoRefreshActive;
    if (isAutoRefreshActive) {
      toggleAutoRefresh.innerHTML = '<span class="w-2 h-2 rounded-full bg-cyber-emerald animate-ping"></span> 15s LIVE';
      toggleAutoRefresh.classList.replace("text-slate-500", "text-cyber-emerald");
      startTimer();
    } else {
      toggleAutoRefresh.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-500"></span> PAUSED';
      toggleAutoRefresh.classList.replace("text-cyber-emerald", "text-slate-500");
      clearInterval(autoRefreshTimer);
    }
  });

  startTimer();
}

let lastDataSyncTimestamp = null;
let freshnessTicker = null;

function updateFreshnessTicker() {
  if (!lastDataSyncTimestamp) return;
  const secAgo = Math.max(0, Math.floor((Date.now() - lastDataSyncTimestamp) / 1000));
  const timeText = secAgo < 3 ? "just now" : `${secAgo}s ago`;

  const lastUpdatedEl = document.getElementById("last-updated-text");
  if (lastUpdatedEl) {
    lastUpdatedEl.textContent = `Updated ${timeText}`;
  }

  const freshnessEl = document.getElementById("data-freshness");
  if (freshnessEl && currentTrainData) {
    const isLive = currentTrainData.data_freshness?.provider_mode === "LIVE";
    freshnessEl.textContent = `${isLive ? "● Live rail telemetry" : "● Latest reported telemetry"} • Updated ${timeText}`;
  }
}

// Execute Live Prediction API Call
async function executePrediction(trainNumber, isBackground = false) {
  const btnPredict = document.getElementById("btn-predict");
  const searchStatus = document.getElementById("search-status-text");
  const currentSeq = ++activePredictionSeq;

  if (!isBackground && resetAutoRefreshTimer) {
    resetAutoRefreshTimer();
  }
  if (!isBackground && btnPredict) {
    btnPredict.disabled = true;
    btnPredict.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i> Fetching live telemetry...';
    if (window.lucide) window.lucide.createIcons();
  }
  if (!isBackground && searchStatus) {
    searchStatus.textContent = `Fetching live telemetry for Train #${trainNumber}...`;
  }

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ train: trainNumber }),
    });

    if (currentSeq !== activePredictionSeq) return;

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      let msg = err.detail;
      if (res.status === 401) {
        msg = "Session expired or unauthorized. Please reload the page.";
      } else if (res.status === 429) {
        msg = "Rate limit exceeded (120 req/min). Please wait a moment.";
      } else if (res.status === 400 || res.status === 404) {
        msg = typeof err.detail === "string" ? err.detail : "Train not found or not running today.";
      } else if (res.status === 502) {
        msg = typeof err.detail === "string" ? err.detail : "External railway telemetry provider is temporarily unreachable.";
      } else if (res.status === 503) {
        msg = typeof err.detail === "string" ? err.detail : "Prediction service temporarily unavailable.";
      }
      showToast(`Prediction: ${msg || "Unable to fetch train telemetry"}`, "error");
      if (searchStatus) {
        searchStatus.textContent = `Unable to load Train #${trainNumber}: ${msg || "Error"}`;
      }
      return;
    }

    const data = await res.json();
    if (currentSeq !== activePredictionSeq) return;
    currentTrainData = data;
    lastDataSyncTimestamp = Date.now();
    renderPredictionResults(data);

    if (searchStatus) {
      searchStatus.textContent = `Live telemetry synced for Train #${data.train} (${data.train_name || "Express"}).`;
    }
    if (!freshnessTicker) {
      freshnessTicker = setInterval(updateFreshnessTicker, 1000);
    }
    updateFreshnessTicker();
  } catch (error) {
    if (currentSeq !== activePredictionSeq) return;
    console.error("API error:", error);
    if (!isBackground) showToast("Unable to reach the prediction service. Please try again.", "error");
    if (searchStatus) {
      searchStatus.textContent = "Network error while reaching prediction service.";
    }
  } finally {
    if (!isBackground && btnPredict && currentSeq === activePredictionSeq) {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i> Check Live ETA';
      if (window.lucide) window.lucide.createIcons();
    }
  }
}

// Render Results on UI
function renderPredictionResults(data) {
  renderPredictionExplanation(data);
  notifyOnDelayChange(data);

  // Train Identity in Hero
  document.getElementById("hero-train-number").textContent = `#${data.train}`;
  document.getElementById("hero-train-name").textContent = data.train_name || `Express #${data.train}`;

  // Current Station & Next Station Flow
  const currStationText = data.current_station_name ? `${data.current_station_name} (${data.current_station})` : (data.current_station || "Origin");
  const nextStationText = data.next_station_name ? `${data.next_station_name} (${data.next_station})` : (data.next_station || (data.upcoming_stations?.length === 0 ? "Terminus (Arrived)" : "Destination"));

  const currStationEl = document.getElementById("hero-curr-station");
  if (currStationEl) {
    currStationEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-cyber-teal"></span> ${escapeHtml(currStationText)}`;
  }
  const nextStationEl = document.getElementById("hero-next-station");
  if (nextStationEl) {
    nextStationEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-cyber-amber"></span> ${escapeHtml(nextStationText)}`;
  }

  // Current Delay Value & Subtext
  const delayMinutes = data.current_delay_minutes || 0;
  const currDelayValEl = document.getElementById("hero-current-delay-val");
  const currDelaySubEl = document.getElementById("hero-delay-sub");
  const statusBadge = document.getElementById("hero-delay-status");

  if (currDelayValEl) {
    if (delayMinutes <= 0) {
      currDelayValEl.textContent = delayMinutes === 0 ? "0 min" : `${delayMinutes} min`;
      currDelayValEl.className = "text-2xl font-extrabold font-mono text-emerald-400";
    } else {
      currDelayValEl.textContent = `+${Math.round(delayMinutes)} min`;
      currDelayValEl.className = `text-2xl font-extrabold font-mono ${delayMinutes > 60 ? "text-rose-400" : delayMinutes > 15 ? "text-amber-400" : "text-yellow-400"}`;
    }
  }

  if (currDelaySubEl) {
    currDelaySubEl.textContent = delayMinutes <= 0
      ? (delayMinutes < 0 ? "Running ahead of schedule" : "Operating on schedule")
      : (delayMinutes > 60 ? "Severe cascading delay" : "Current reported delay");
  }

  // Clear visual status badge distinction
  if (statusBadge) {
    if (delayMinutes <= 0) {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5";
      statusBadge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> On-Time / Early';
    } else if (delayMinutes <= 15) {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1.5";
      statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Minor Delay (+${Math.round(delayMinutes)}m)`;
    } else if (delayMinutes <= 60) {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-amber-500/15 text-amber-300 border border-amber-500/40 flex items-center gap-1.5";
      statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Delayed (+${Math.round(delayMinutes)}m)`;
    } else {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1.5";
      statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-rose-400"></span> Severely Delayed (+${Math.round(delayMinutes)}m)`;
    }
  }

  // Next Station ETA Countdown
  const nextCountdownEl = document.getElementById("hero-next-eta-countdown");
  const nextRelEl = document.getElementById("hero-next-eta-rel");
  if (data.next_station) {
    let clockStr = "--:--";
    if (data.next_station_eta) {
      clockStr = new Date(data.next_station_eta).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true });
    }
    const nextEtaMin = Math.round(data.next_station_eta_minutes ?? 0);
    if (nextCountdownEl) nextCountdownEl.textContent = clockStr;
    if (nextRelEl) nextRelEl.textContent = nextEtaMin > 0 ? `+${nextEtaMin}m from now` : "Approaching now";
  } else {
    if (nextCountdownEl) nextCountdownEl.textContent = "ARRIVED";
    if (nextRelEl) nextRelEl.textContent = "Terminus reached";
  }

  // Segment Progress Bar
  const progressPct = Math.min(100, Math.max(0, Math.round((data.segment_progress || 0) * 100)));
  const progCurr = document.getElementById("progress-curr-station");
  const progNext = document.getElementById("progress-next-station");
  const progLbl = document.getElementById("progress-pct-label");
  const progBar = document.getElementById("segment-progress-bar");

  if (progCurr) progCurr.textContent = data.current_station_name || data.current_station || "Origin";
  if (progNext) progNext.textContent = data.next_station_name || data.next_station || (progressPct >= 100 ? "Terminus (Arrived)" : "Terminus");
  if (progLbl) progLbl.textContent = `${progressPct}% Segment Complete`;
  if (progBar) progBar.style.width = `${progressPct}%`;

  // 4-Card HUD
  // Card 1: AI Predicted Delay
  const predDelay = data.predicted_delay_minutes ?? 0;
  const addDelay = data.additional_predicted_delay_minutes ?? 0;
  const hudPredDelay = document.getElementById("hud-predicted-delay");
  const hudDelayDelta = document.getElementById("hud-delay-delta");
  if (hudPredDelay) hudPredDelay.textContent = formatDelay(predDelay);
  if (hudDelayDelta) {
    hudDelayDelta.textContent = addDelay > 0.5 ? `+${formatDelay(addDelay)} vs current delay` : "No additional delay predicted";
  }

  // Card 2: Historical Segment Evidence (Replaced fake confidence!)
  const histStats = data.historical_statistics || {};
  const segmentEvidenceEl = document.getElementById("hud-segment-evidence");
  if (segmentEvidenceEl) {
    segmentEvidenceEl.innerHTML = `${(histStats.count || 0).toLocaleString()} <span class="text-sm font-normal text-slate-400">samples</span>`;
  }
  const statsScopeEl = document.getElementById("hud-stats-scope");
  if (statsScopeEl) {
    const scope = data.historical_lookup_scope || "EXACT";
    const seg = data.historical_segment || `${data.current_station} \u2192 ${data.next_station}`;
    statsScopeEl.textContent = `Scope: ${scope} (${seg})`;
  }

  // Card 3: Track Weather (Context only)
  const weather = data.weather || {};
  const weatherTempEl = document.getElementById("hud-weather-temp");
  const weatherCondEl = document.getElementById("hud-weather-condition");
  if (weather.available && weather.temperature_c != null) {
    if (weatherTempEl) weatherTempEl.innerHTML = `${weather.temperature_c}&deg;C`;
    if (weatherCondEl) weatherCondEl.textContent = `Humidity: ${weather.humidity_percent ?? "--"}% | Wind: ${weather.wind_speed_kmh ?? "--"}km/h`;
  } else {
    if (weatherTempEl) weatherTempEl.innerHTML = "--&deg;C";
    if (weatherCondEl) weatherCondEl.textContent = "Weather feed unavailable";
  }

  // Card 4: Historical Segment Average
  const histMedianEl = document.getElementById("hud-hist-median");
  const histMeanEl = document.getElementById("hud-hist-mean");
  if (histMedianEl) histMedianEl.innerHTML = `${histStats.median ?? 0} <span class="text-sm font-normal text-slate-400">min</span>`;
  if (histMeanEl) histMeanEl.textContent = `Mean: ${histStats.mean ?? 0}m (\u00B1${histStats.std ?? 0}m)`;

  // Render Map & Timeline
  renderMapRoute(data);
  renderStationsTimeline(data);

  if (window.lucide) window.lucide.createIcons();
}

function renderPredictionExplanation(data) {
  const explanation = data.prediction_explanation || {};
  const freshness = data.data_freshness || {};
  const summary = document.getElementById("prediction-summary");
  const freshnessElement = document.getElementById("data-freshness");
  const factors = document.getElementById("prediction-factors");
  const weatherNote = document.getElementById("prediction-weather-note");

  if (summary) {
    summary.textContent = explanation.summary || "Explanation unavailable for this prediction.";
  }

  if (freshnessElement) {
    const live = freshness.provider_mode === "LIVE";
    freshnessElement.textContent = live ? "● Live rail telemetry • Updated just now" : "● Latest reported telemetry • Updated just now";
    freshnessElement.className = `text-xs font-mono px-2 py-1 rounded border ${live ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" : "bg-amber-500/10 border-amber-500/30 text-amber-400"}`;
  }

  if (factors) {
    factors.replaceChildren();
    (explanation.factors || []).forEach((factor) => {
      const item = document.createElement("div");
      item.className = "rounded-lg bg-cyber-bg/70 border border-white/5 p-2";
      const label = document.createElement("div");
      label.className = "text-[10px] text-slate-500 uppercase";
      label.textContent = factor.name;
      const value = document.createElement("div");
      value.className = "text-sm font-mono text-slate-200";
      value.textContent = `${factor.value} ${factor.unit || ""}`;
      item.append(label, value);
      factors.appendChild(item);
    });
  }

  if (weatherNote) {
    weatherNote.textContent = explanation.weather_note || "Weather is displayed for context and is not currently a live-model feature.";
  }
}

function notifyOnDelayChange(data) {
  const delay = Number(data.predicted_delay_minutes || 0);
  const significantChange = lastAlertedDelay !== null && Math.abs(delay - lastAlertedDelay) >= 10;
  const newlySevere = lastAlertedDelay !== null && lastAlertedDelay < 30 && delay >= 30;
  if (alertsEnabled() && (significantChange || newlySevere)) {
    new Notification(`Train ${data.train}: delay update`, {
      body: `Predicted delay is now ${delay.toFixed(0)} minutes. Next: ${data.next_station_name || data.next_station || "destination"}.`,
    });
  }
  lastAlertedDelay = delay;
}

// Render Map Elements with Stations
function renderMapRoute(data) {
  if (!map) return;

  stationMarkersLayer.clearLayers();

  const lat = data.position?.latitude;
  const lon = data.position?.longitude;

  // Route geometry & clean overlay
  const noRouteMsg = document.getElementById("map-no-route-msg");
  const routeCoordinates = data.route_geometry?.geometry?.coordinates;
  const routeSignature = routeCoordinates ? `${data.train}:${routeCoordinates.length}:${JSON.stringify(routeCoordinates[0])}:${JSON.stringify(routeCoordinates.at(-1))}` : null;

  if (routeCoordinates && routeCoordinates.length > 0) {
    if (noRouteMsg) noRouteMsg.classList.add("hidden");
    if (routeSignature !== renderedRouteSignature) {
      if (routePolyline) map.removeLayer(routePolyline);

      routePolyline = L.geoJSON(data.route_geometry, {
        style: {
          color: "#00f2fe",
          weight: 4.5,
          opacity: 0.9,
        },
      }).addTo(map);
      map.fitBounds(routePolyline.getBounds(), { padding: [40, 40] });
      renderedRouteSignature = routeSignature;
    }
  } else {
    if (noRouteMsg) noRouteMsg.classList.remove("hidden");
    if (routePolyline) {
      map.removeLayer(routePolyline);
      routePolyline = null;
      renderedRouteSignature = null;
    }
  }

  // Train Marker
  if (lat && lon) {
    const isLive = data.data_freshness?.provider_mode === "LIVE";
    const modePrefix = isLive ? "LIVE TELEMETRY" : "REPORTED POSITION";
    const stationLabel = data.current_station_name || data.current_station || "Train location";
    const label = `${modePrefix} • ${stationLabel} • ${Math.round(data.current_delay_minutes || 0)} min delay`;
    const trainIcon = L.divIcon({
      className: "train-pulse-icon",
      html: `<div class="train-location-label">${escapeHtml(label)}</div><div class="train-pulse-inner"></div><div class="train-shape" aria-label="Train location marker"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h12a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3l2 2v1h-3l-2-2h-6l-2 2H4v-1l2-2a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3Zm0 3v6h12V6H6Zm2 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm8 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z" /></svg></div>`,
      iconSize: [250, 62],
      iconAnchor: [125, 31],
    });

    if (trainMarker) {
      trainMarker.setLatLng([lat, lon]);
      trainMarker.setIcon(trainIcon);
    } else {
      trainMarker = L.marker([lat, lon], { icon: trainIcon }).addTo(map);
    }

    const posSource = data.position?.source || (isLive ? "Reported Provider Telemetry" : "Scheduled Station Coordinates");
    const popup = `
      <div class="p-2.5 font-sans">
        <div class="font-bold text-cyber-teal text-sm">🚆 #${escapeHtml(data.train)} ${escapeHtml(data.train_name || "")}</div>
        <div class="text-xs text-slate-200 mt-1">Current Halt: <b>${escapeHtml(stationLabel)}</b></div>
        <div class="text-xs text-slate-200 mt-0.5">Current Delay: <b class="text-amber-400">${Math.round(data.current_delay_minutes || 0)} mins</b></div>
        <div class="text-xs text-slate-300 mt-0.5">Segment Progress: <b>${Math.round((data.segment_progress || 0) * 100)}%</b></div>
        <div class="text-[11px] text-slate-400 mt-1">Position Source: ${escapeHtml(posSource)}</div>
      </div>
    `;
    if (trainMarker.getPopup()) trainMarker.setPopupContent(popup);
    else trainMarker.bindPopup(popup);
    updateTrainLabelVisibility();
  } else if (trainMarker) {
    map.removeLayer(trainMarker);
    trainMarker = null;
  }
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = String(value);
  return element.innerHTML;
}

function updateTrainLabelVisibility() {
  if (!map || !trainMarker) return;
  const markerElement = trainMarker.getElement();
  if (markerElement) {
    const closeZoom = map.getZoom() >= 8;
    markerElement.classList.toggle("train-label-visible", closeZoom);
    markerElement.classList.toggle("train-shape-visible", closeZoom);
  }
}

// Render Railway-Style Vertical Timeline
function renderStationsTimeline(data) {
  const upcomingStations = data.upcoming_stations || [];
  const container = document.getElementById("stations-timeline-container");
  const countBadge = document.getElementById("timeline-stop-count");
  if (!container) return;

  container.innerHTML = "";
  if (countBadge) countBadge.textContent = `${upcomingStations.length} halts`;

  // 1. Current Station as Starting Node
  if (data.current_station) {
    const currentItem = document.createElement("div");
    currentItem.className = "timeline-item";
    const currDelay = Math.round(data.current_delay_minutes || 0);
    const delayBadge = currDelay <= 0
      ? '<span class="text-xs font-mono font-bold text-emerald-400">On Time</span>'
      : `<span class="text-xs font-mono font-bold text-amber-400">+${currDelay} min delay</span>`;

    currentItem.innerHTML = `
      <div class="timeline-dot timeline-dot-current">
        <div class="timeline-dot-inner"></div>
      </div>
      <div class="p-3 rounded-xl bg-cyber-teal/10 border border-cyber-teal/40">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="font-bold text-sm text-white">${escapeHtml(data.current_station_name || data.current_station)}</span>
            <span class="text-xs font-mono text-cyber-teal font-bold px-1.5 py-0.5 rounded bg-cyber-teal/20">${escapeHtml(data.current_station)}</span>
            <span class="text-[10px] uppercase font-bold text-cyber-teal bg-cyber-teal/20 px-2 py-0.5 rounded-full border border-cyber-teal/30">Current Halt</span>
          </div>
          <div class="font-mono text-xs">
            ${delayBadge}
          </div>
        </div>
      </div>
    `;
    container.appendChild(currentItem);
  }

  // 2. Upcoming Stations or Terminus Reached
  if (upcomingStations.length === 0) {
    const emptyItem = document.createElement("div");
    emptyItem.className = "timeline-item";
    emptyItem.innerHTML = `
      <div class="timeline-dot">
        <div class="timeline-dot-inner"></div>
      </div>
      <div class="p-4 text-center rounded-xl bg-cyber-bg/60 border border-white/5">
        <div class="font-bold text-sm text-slate-300">ARRIVED &bull; Terminus Reached</div>
        <p class="text-xs text-slate-500 mt-1">This train has completed all scheduled stops.</p>
      </div>
    `;
    container.appendChild(emptyItem);
    return;
  }

  upcomingStations.forEach((stop, index) => {
    const isNext = index === 0;
    const isTerminus = index === upcomingStations.length - 1;
    const item = document.createElement("div");
    item.className = "timeline-item";

    const schArr = stop.scheduled_arrival
      ? new Date(stop.scheduled_arrival).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true })
      : "--:--";
    const predArr = stop.predicted_arrival
      ? new Date(stop.predicted_arrival).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true })
      : "--:--";

    const etaMin = Math.round(stop.eta_minutes_from_now);
    const etaText = etaMin > 0 ? `+${etaMin}m from now` : "Approaching now";

    item.innerHTML = `
      <div class="timeline-dot ${isNext ? "timeline-dot-next" : ""}">
        <div class="timeline-dot-inner"></div>
      </div>
      <div class="p-3.5 rounded-xl border transition-all ${
        isNext
          ? "bg-cyber-surface/90 border-amber-500/40 shadow-lg shadow-amber-500/5"
          : "bg-cyber-bg/70 border-white/5 hover:border-white/20"
      }">
        <div class="flex items-start justify-between gap-3">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-sm text-white">${escapeHtml(stop.station_name || stop.station_code)}</span>
              <span class="text-xs font-mono text-slate-400 px-1.5 py-0.2 rounded bg-white/5">${escapeHtml(stop.station_code)}</span>
              ${isNext ? '<span class="text-[10px] uppercase font-bold text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded border border-amber-400/20">Next Stop</span>' : ""}
              ${isTerminus ? '<span class="text-[10px] uppercase font-mono text-slate-400 bg-slate-700/50 px-1.5 py-0.5 rounded">Terminus</span>' : ""}
            </div>
            <div class="text-xs text-slate-400 mt-1 flex items-center gap-3 font-mono">
              <span>Dist: ${stop.distance_km != null ? stop.distance_km + " km" : "--"}</span>
              ${stop.platform ? `<span>Platform: ${escapeHtml(stop.platform)}</span>` : `<span>${stop.is_halt ? "Scheduled Halt" : "Through"}</span>`}
            </div>
          </div>

          <div class="text-right font-mono">
            <div class="text-sm font-bold ${isNext ? "text-amber-400" : "text-cyber-teal"}">ETA ${predArr}</div>
            <div class="text-[11px] text-slate-400 line-through">Sch: ${schArr}</div>
          </div>
        </div>

        <div class="mt-2.5 pt-2 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Arrival: <b>${etaText}</b></span>
          <span class="${stop.is_independent_ml_prediction ? "text-cyber-teal font-semibold" : "text-slate-400"}">
            ${stop.is_independent_ml_prediction ? `⚡ Autoregressive Hop ${stop.cascade_hop || index + 1}` : "Downstream Cascaded"}
          </span>
        </div>
      </div>
    `;

    container.appendChild(item);
  });
}

// Setup Journey Level Simulator Tab
function setupJourneySimulation() {
  const sliderDist = document.getElementById("slider-distance");
  const sliderFog = document.getElementById("slider-fog");
  const sliderMaint = document.getElementById("slider-maintenance");

  const updateDistLabels = (val) => {
    const dist = parseFloat(val);
    document.getElementById("val-distance").textContent = `${dist} km`;
    const estStopsEl = document.getElementById("sim-est-stops");
    const estHoursEl = document.getElementById("sim-est-hours");
    if (estStopsEl) estStopsEl.textContent = `${Math.max(1, Math.round(dist / 65))} halts`;
    if (estHoursEl) estHoursEl.textContent = `${(dist / 65.0).toFixed(1)} hrs`;
  };

  sliderDist?.addEventListener("input", (e) => updateDistLabels(e.target.value));
  sliderFog?.addEventListener("input", (e) => {
    document.getElementById("val-fog").textContent = e.target.value;
  });
  sliderMaint?.addEventListener("input", (e) => {
    document.getElementById("val-maintenance").textContent = `${e.target.value} / 10`;
  });

  // Scenario Presets
  const presets = {
    monsoon: { trainType: "Superfast Express", season: "Monsoon", fog: 0.2, maint: 6.0, dist: 1400, zone: "Northern Railway (NR)", monsoon: true, festival: false, doubled: true, hdn: false, lhb: true, shared: false, late: false, traction: "Electric (25kV AC)" },
    winter_fog: { trainType: "Mail/Express", season: "Winter/Fog", fog: 0.95, maint: 5.5, dist: 1200, zone: "Northern Railway (NR)", monsoon: false, festival: false, doubled: true, hdn: true, lhb: true, shared: false, late: true, traction: "Electric (25kV AC)" },
    hdn_congestion: { trainType: "Passenger Train", season: "Summer", fog: 0.1, maint: 4.5, dist: 800, zone: "North Central Railway (NCR)", monsoon: false, festival: true, doubled: true, hdn: true, lhb: false, shared: true, late: true, traction: "Electric (25kV AC)" },
    vande_bharat: { trainType: "Vande Bharat Express", season: "Summer", fog: 0.05, maint: 9.5, dist: 750, zone: "Northern Railway (NR)", monsoon: false, festival: false, doubled: true, hdn: false, lhb: true, shared: false, late: false, traction: "Electric (25kV AC)" },
  };

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = presets[btn.dataset.preset];
      if (!p) return;
      if (p.trainType) document.getElementById("journey-train-type").value = p.trainType;
      if (p.season) document.getElementById("journey-season").value = p.season;
      if (p.zone) document.getElementById("journey-zone").value = p.zone;
      if (p.traction) document.getElementById("journey-traction").value = p.traction;

      sliderFog.value = p.fog;
      document.getElementById("val-fog").textContent = p.fog;
      sliderMaint.value = p.maint;
      document.getElementById("val-maintenance").textContent = `${p.maint} / 10`;
      sliderDist.value = p.dist;
      updateDistLabels(p.dist);

      const checkHdn = document.getElementById("check-hdn");
      const checkDoubled = document.getElementById("check-doubled");
      const checkLhb = document.getElementById("check-lhb");
      const checkShared = document.getElementById("check-shared-rake");
      const checkLate = document.getElementById("check-incoming-late");
      const checkMonsoon = document.getElementById("check-monsoon");
      const checkFestival = document.getElementById("check-festival");

      if (checkHdn) checkHdn.checked = !!p.hdn;
      if (checkDoubled) checkDoubled.checked = !!p.doubled;
      if (checkLhb) checkLhb.checked = !!p.lhb;
      if (checkShared) checkShared.checked = !!p.shared;
      if (checkLate) checkLate.checked = !!p.late;
      if (checkMonsoon) checkMonsoon.checked = !!p.monsoon;
      if (checkFestival) checkFestival.checked = !!p.festival;

      runJourneySimulation();
    });
  });

  document.getElementById("btn-simulate-journey")?.addEventListener("click", runJourneySimulation);
  runJourneySimulation();
}

async function runJourneySimulation() {
  const dist = parseFloat(document.getElementById("slider-distance").value);
  const fog = parseFloat(document.getElementById("slider-fog").value);
  const maint = parseFloat(document.getElementById("slider-maintenance").value);
  const trainType = document.getElementById("journey-train-type").value;
  const zone = document.getElementById("journey-zone").value;
  const season = document.getElementById("journey-season").value;
  const traction = document.getElementById("journey-traction").value;

  const zoneAbbrMatch = zone.match(/\(([A-Z]+)\)/);
  const zone_abbr = zoneAbbrMatch ? zoneAbbrMatch[1] : "NR";

  const features = {
    train_number: "12919",
    train_type: trainType,
    year: 2024,
    month: season === "Winter/Fog" ? 1 : (season === "Monsoon" ? 7 : 5),
    day_of_week: 2,
    departure_hour: 14,
    is_weekend: 0,
    is_night_departure: 0,
    is_peak_hour: 1,
    is_festival_season: document.getElementById("check-festival")?.checked ? 1 : 0,
    season: season,
    zone: zone,
    zone_abbr: zone_abbr,
    source_station_category: "A1",
    destination_station_category: "A1",
    distance_km: dist,
    num_scheduled_stops: Math.max(1, Math.round(dist / 65)),
    scheduled_travel_hours: Math.max(0.5, dist / 65.0),
    track_doubled: document.getElementById("check-doubled")?.checked ? 1 : 0,
    is_hdn_route: document.getElementById("check-hdn")?.checked ? 1 : 0,
    traction_type: traction,
    is_electrified: traction.includes("Electric") ? 1 : 0,
    psr_count: 5,
    is_circular_route: 0,
    is_monsoon_season: document.getElementById("check-monsoon")?.checked ? 1 : (season === "Monsoon" ? 1 : 0),
    is_fog_risk: fog > 0.6 ? 1 : 0,
    fog_risk_score: fog,
    zone_fog_index: fog * 0.8,
    zone_congestion_index: 0.65,
    season_severity_score: season === "Monsoon" ? 0.9 : (season === "Winter/Fog" ? 0.85 : 0.4),
    loco_age_years: 5.5,
    coach_age_years: 4.2,
    has_lhb_coaches: document.getElementById("check-lhb")?.checked ? 1 : 0,
    is_rake_shared: document.getElementById("check-shared-rake")?.checked ? 1 : 0,
    maintenance_score: maint,
    seat_utilisation_pct: 95.0,
    is_overloaded: 0,
    late_incoming_rake: document.getElementById("check-incoming-late")?.checked ? 1 : 0,
    is_special_train: 0,
    route_historical_ontime_pct: 82.0,
  };

  const simBtn = document.getElementById("btn-simulate-journey");
  if (simBtn) {
    simBtn.disabled = true;
    simBtn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Running journey model...';
    if (window.lucide) window.lucide.createIcons();
  }

  try {
    const res = await fetch("/predict-journey", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      let msg = typeof err.detail === "string" ? err.detail : "Invalid simulation input";
      if (res.status === 401) {
        msg = "Session expired or unauthorized. Please reload the page.";
      } else if (res.status === 429) {
        msg = "Rate limit exceeded. Please wait a moment.";
      } else if (res.status === 503) {
        msg = "Journey model artifacts are not currently installed on the server.";
      }
      showToast(`Journey simulator: ${msg}`, "error");
      return;
    }

    const data = await res.json();
    const delayMin = data.predicted_destination_delay_minutes;
    document.getElementById("sim-predicted-minutes").innerHTML = `${delayMin} <span class="text-xl font-normal text-slate-400">mins</span>`;

    const badge = document.getElementById("sim-risk-badge");
    if (data.is_predicted_delayed) {
      badge.className = "px-3.5 py-1.5 rounded-full text-xs font-semibold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/30 mt-2";
      badge.textContent = `⚠️ Likely Delayed (>15m Late Arrival Threshold Exceeded)`;
    } else {
      badge.className = "px-3.5 py-1.5 rounded-full text-xs font-semibold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 mt-2";
      badge.textContent = `✅ Likely On-Time (Within 15m Threshold)`;
    }

    const valMetrics = data.validation_metrics || data.validation;
    if (valMetrics && valMetrics.model) {
      const maeEl = document.getElementById("sim-val-mae");
      const recallEl = document.getElementById("sim-val-recall");
      if (maeEl && valMetrics.model.mae_minutes != null) {
        maeEl.textContent = `${valMetrics.model.mae_minutes} min`;
      }
      if (recallEl && valMetrics.model.late_arrival_recall != null) {
        recallEl.textContent = `${(valMetrics.model.late_arrival_recall * 100).toFixed(2)}%`;
      }
    }
  } catch (e) {
    console.error("Simulation error:", e);
    showToast("Unable to calculate journey delay simulation.", "error");
  } finally {
    if (simBtn) {
      simBtn.disabled = false;
      simBtn.innerHTML = '<i data-lucide="play" class="w-5 h-5"></i> Run Route Delay Simulation';
      if (window.lucide) window.lucide.createIcons();
    }
  }
}
