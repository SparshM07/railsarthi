/**
 * RailPulse AI - Command Center Web Application
 * Handles live telemetry polling, Leaflet map rendering, cascading station timelines,
 * and offline journey-level ML simulations.
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

function alertsEnabled() {
  return localStorage.getItem("railpulse-alerts") === "enabled" && Notification.permission === "granted";
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
    localStorage.setItem("railpulse-alerts", "enabled");
    showToast("Delay alerts enabled for this browser.", "success");
  } else {
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
        providerModeText.textContent = data.provider_mode === "LIVE_READY" ? "🟢 LIVE PROVIDER READY" : "🟡 SIMULATION MODE";
      }
    }
  } catch (e) {
    if (providerModeText) providerModeText.textContent = "SERVICE ACTIVE";
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
function setupAutoRefresh() {
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

  function startTimer() {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = setInterval(() => {
      if (isAutoRefreshActive && currentTrainData) {
        executePrediction(currentTrainData.train, true);
      }
    }, 15000);
  }
  startTimer();
}

// Execute Live Prediction API Call
async function executePrediction(trainNumber, isBackground = false) {
  const btnPredict = document.getElementById("btn-predict");
  if (!isBackground && btnPredict) {
    btnPredict.disabled = true;
    btnPredict.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Processing...';
    if (window.lucide) window.lucide.createIcons();
  }

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ train: trainNumber }),
    });

    if (!res.ok) {
      const err = await res.json();
      showToast(`Prediction error: ${err.detail || "Unable to fetch train prediction"}`, "error");
      return;
    }

    const data = await res.json();
    currentTrainData = data;
    renderPredictionResults(data);
  } catch (error) {
    console.error("API error:", error);
    if (!isBackground) showToast("Unable to reach the prediction service. Please try again.", "error");
  } finally {
    if (!isBackground && btnPredict) {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '<i data-lucide="zap" class="w-5 h-5"></i> Run AI Predict';
      if (window.lucide) window.lucide.createIcons();
    }
  }
}

// Render Results on UI
function renderPredictionResults(data) {
  renderPredictionExplanation(data);
  notifyOnDelayChange(data);
  // Hero Section
  document.getElementById("hero-train-number").textContent = `#${data.train}`;
  document.getElementById("hero-train-name").textContent = data.train_name || `Express #${data.train}`;

  const delayMinutes = data.current_delay_minutes || 0;
  const statusBadge = document.getElementById("hero-delay-status");
  if (delayMinutes <= 5) {
    statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5";
    statusBadge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> On-Time / Normal';
  } else {
    statusBadge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1.5";
    statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Delayed by ${delayMinutes} min`;
  }

  document.getElementById("hero-route-summary").innerHTML = `
    <i data-lucide="map-pin" class="w-4 h-4 text-cyber-teal"></i> ${data.current_station_name || data.current_station} &rarr; ${data.next_station_name || data.next_station || "Destination"}
  `;

  // Countdown & Next Station
  const nextEtaMin = data.next_station_eta_minutes ?? 0;
  const hours = Math.floor(nextEtaMin / 60);
  const mins = Math.round(nextEtaMin % 60);
  document.getElementById("hero-next-eta-countdown").textContent = `${String(hours).padStart(2, "0")}h ${String(mins).padStart(2, "0")}m`;
  document.getElementById("hero-next-station-label").textContent = data.next_station_name ? `${data.next_station_name} (${data.next_station})` : "Final Destination";

  // Progress Bar
  const progressPct = Math.round((data.segment_progress || 0) * 100);
  document.getElementById("progress-curr-station").textContent = data.current_station_name || data.current_station;
  document.getElementById("progress-next-station").textContent = data.next_station_name || data.next_station || "Terminus";
  document.getElementById("progress-pct-label").textContent = `${progressPct}% Segment Complete`;
  document.getElementById("segment-progress-bar").style.width = `${progressPct}%`;

  // 4-Card HUD
  const predDelay = data.predicted_delay_minutes ?? 0;
  const addDelay = data.additional_predicted_delay_minutes ?? 0;
  document.getElementById("hud-predicted-delay").innerHTML = `${predDelay} <span class="text-sm font-normal text-slate-400">min</span>`;
  document.getElementById("hud-delay-delta").textContent = addDelay > 0 ? `+${addDelay}m vs current delay` : "No additional delay predicted";

  const confidence = data.eta_confidence || "MEDIUM";
  const confElem = document.getElementById("hud-confidence");
  confElem.textContent = confidence;
  confElem.className = `text-2xl font-bold font-mono ${confidence === "HIGH" ? "text-emerald-400" : confidence === "MEDIUM" ? "text-cyan-400" : "text-amber-400"}`;
  document.getElementById("hud-stats-scope").textContent = `${data.historical_statistics?.count || 0} Historical Samples (${data.historical_lookup_scope || "EXACT"})`;

  // Weather HUD
  const weather = data.weather || {};
  document.getElementById("hud-weather-temp").innerHTML = weather.temperature_c !== undefined ? `${weather.temperature_c}&deg;C` : "--&deg;C";
  document.getElementById("hud-weather-condition").textContent = `Humidity: ${weather.humidity_percent || "--"}% | Wind: ${weather.wind_speed_kmh || "--"}km/h`;

  // Historical HUD
  const histStats = data.historical_statistics || {};
  document.getElementById("hud-hist-median").innerHTML = `${histStats.median ?? 0} <span class="text-sm font-normal text-slate-400">min</span>`;
  document.getElementById("hud-hist-mean").textContent = `Mean: ${histStats.mean ?? 0}m (\u00B1${histStats.std ?? 0}m)`;

  // Render Map & Timeline
  renderMapRoute(data);
  renderStationsTimeline(data.upcoming_stations || []);

  if (window.lucide) window.lucide.createIcons();
}

function renderPredictionExplanation(data) {
  const explanation = data.prediction_explanation || {};
  const freshness = data.data_freshness || {};
  const summary = document.getElementById("prediction-summary");
  const freshnessElement = document.getElementById("data-freshness");
  const factors = document.getElementById("prediction-factors");
  const weatherNote = document.getElementById("prediction-weather-note");
  if (summary) summary.textContent = explanation.summary || "Explanation unavailable for this prediction.";
  if (freshnessElement) {
    const live = freshness.provider_mode === "LIVE";
    freshnessElement.textContent = live ? "● Live provider data" : "● Simulated fallback data";
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
  if (weatherNote) weatherNote.textContent = explanation.weather_note || "";
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

  // Draw GeoJSON Route or Polyline
  const routeCoordinates = data.route_geometry?.geometry?.coordinates;
  const routeSignature = routeCoordinates ? `${data.train}:${routeCoordinates.length}:${JSON.stringify(routeCoordinates[0])}:${JSON.stringify(routeCoordinates.at(-1))}` : null;
  if (routeCoordinates && routeSignature !== renderedRouteSignature) {
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

  // Draw Station Markers along the upcoming list
  (data.upcoming_stations || []).forEach((stop, index) => {
    const isFirst = index === 0;
    // Station coordinates from stop if available
    // Otherwise fallback if stop coordinates exist
  });

  // Train Marker
  if (lat && lon) {
    const label = `LIVE • ${data.current_station_name || data.current_station || "Train location"} • ${Math.round(data.current_delay_minutes || 0)} min delay`;
    const trainIcon = L.divIcon({
      className: "train-pulse-icon",
      html: `<div class="train-location-label">${escapeHtml(label)}</div><div class="train-pulse-inner"></div>`,
      // The pulse dot is flex-centred in this icon. Anchor on that centre so
      // the visual train sits exactly on the route coordinate.
      iconSize: [250, 62],
      iconAnchor: [125, 31],
    });

    if (trainMarker) {
      trainMarker.setLatLng([lat, lon]);
      trainMarker.setIcon(trainIcon);
    } else {
      trainMarker = L.marker([lat, lon], { icon: trainIcon }).addTo(map);
    }

    const popup = `
      <div class="p-2.5 font-sans">
        <div class="font-bold text-cyber-teal text-sm">🚆 #${data.train} ${data.train_name || ""}</div>
        <div class="text-xs text-slate-200 mt-1">Current Delay: <b class="text-amber-400">${data.current_delay_minutes} mins</b></div>
        <div class="text-xs text-slate-300 mt-0.5">Segment Progress: <b>${Math.round((data.segment_progress || 0) * 100)}%</b></div>
        <div class="text-[11px] text-slate-400 mt-1">Position Source: ${data.position?.source || "GPS Interpolated"}</div>
      </div>
    `;
    if (trainMarker.getPopup()) trainMarker.setPopupContent(popup);
    else trainMarker.bindPopup(popup);
    updateTrainLabelVisibility();
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
  if (markerElement) markerElement.classList.toggle("train-label-visible", map.getZoom() >= 8);
}

// Render Timeline of Stations
function renderStationsTimeline(upcomingStations) {
  const container = document.getElementById("stations-timeline-container");
  const countBadge = document.getElementById("timeline-stop-count");
  if (!container) return;

  container.innerHTML = "";
  if (countBadge) countBadge.textContent = `${upcomingStations.length} stops`;

  if (upcomingStations.length === 0) {
    container.innerHTML = '<div class="p-6 text-center text-slate-400 text-sm">No upcoming stops recorded for this route.</div>';
    return;
  }

  upcomingStations.forEach((stop, index) => {
    const isFirst = index === 0;
    const item = document.createElement("div");
    item.className = `p-3.5 rounded-xl border transition-all ${
      isFirst
        ? "bg-cyber-surface/90 border-cyber-teal/40 shadow-lg shadow-cyber-teal/5"
        : "bg-cyber-bg/70 border-white/5 hover:border-white/20"
    }`;

    const schArr = stop.scheduled_arrival ? new Date(stop.scheduled_arrival).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false }) : "--:--";
    const predArr = stop.predicted_arrival ? new Date(stop.predicted_arrival).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false }) : "--:--";

    item.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div class="flex items-start gap-2.5">
          <div class="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-mono font-bold mt-0.5 ${
            isFirst ? "bg-cyber-teal text-slate-950" : "bg-cyber-surface text-slate-400 border border-white/10"
          }">
            ${stop.sequence || index + 1}
          </div>
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-sm text-white">${stop.station_name || stop.station_code}</span>
              <span class="text-xs font-mono text-cyber-teal px-1.5 py-0.2 rounded bg-cyber-teal/10">${stop.station_code}</span>
              ${isFirst ? '<span class="text-[10px] uppercase font-bold text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded border border-amber-400/20">Immediate Next</span>' : ""}
            </div>
            <div class="text-xs text-slate-400 mt-1 flex items-center gap-3 font-mono">
              <span>Dist: ${stop.distance_km ?? "--"} km</span>
              <span>Platform: ${stop.platform || "1"}</span>
            </div>
          </div>
        </div>

        <div class="text-right font-mono">
          <div class="text-sm font-bold ${isFirst ? "text-cyber-teal" : "text-white"}">${predArr}</div>
          <div class="text-[11px] text-slate-400 line-through">Sch: ${schArr}</div>
        </div>
      </div>

      <div class="mt-2.5 pt-2 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span>ETA: <b>+${Math.round(stop.eta_minutes_from_now)}m</b> from now</span>
        <span class="${stop.is_independent_ml_prediction ? "text-cyber-teal" : "text-slate-400"}">
          ${stop.is_independent_ml_prediction ? "⚡ LightGBM Direct Predict" : "Cascaded Schedule"}
        </span>
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

  sliderDist?.addEventListener("input", (e) => {
    document.getElementById("val-distance").textContent = `${e.target.value} km`;
  });
  sliderFog?.addEventListener("input", (e) => {
    document.getElementById("val-fog").textContent = e.target.value;
  });
  sliderMaint?.addEventListener("input", (e) => {
    document.getElementById("val-maintenance").textContent = e.target.value;
  });

  // Presets
  const presets = {
    monsoon: { trainType: "Superfast", season: "Monsoon", fog: 0.2, maint: 6.0, dist: 1400 },
    winter_fog: { trainType: "Mail/Express", season: "Winter/Fog", fog: 0.95, maint: 5.5, dist: 1200 },
    hdn_congestion: { trainType: "Passenger", season: "Summer", fog: 0.1, maint: 4.5, dist: 800 },
    vande_bharat: { trainType: "Vande Bharat", season: "Summer", fog: 0.05, maint: 9.5, dist: 750 },
  };

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = presets[btn.dataset.preset];
      if (!p) return;
      document.getElementById("journey-train-type").value = p.trainType;
      document.getElementById("journey-season").value = p.season;
      sliderFog.value = p.fog;
      document.getElementById("val-fog").textContent = p.fog;
      sliderMaint.value = p.maint;
      document.getElementById("val-maintenance").textContent = p.maint;
      sliderDist.value = p.dist;
      document.getElementById("val-distance").textContent = `${p.dist} km`;
      runJourneySimulation();
    });
  });

  document.getElementById("btn-simulate-journey")?.addEventListener("click", runJourneySimulation);
}

async function runJourneySimulation() {
  const dist = parseFloat(document.getElementById("slider-distance").value);
  const fog = parseFloat(document.getElementById("slider-fog").value);
  const maint = parseFloat(document.getElementById("slider-maintenance").value);
  const trainType = document.getElementById("journey-train-type").value;
  const zone = document.getElementById("journey-zone").value;
  const season = document.getElementById("journey-season").value;
  const traction = document.getElementById("journey-traction").value;

  const features = {
    train_number: "12919",
    train_type: trainType,
    year: 2024,
    month: 1,
    day_of_week: 2,
    departure_hour: 14,
    is_weekend: 0,
    is_night_departure: 0,
    is_peak_hour: 1,
    is_festival_season: 0,
    season: season,
    zone: zone,
    zone_abbr: "NR",
    source_station_category: "A1",
    destination_station_category: "A1",
    distance_km: dist,
    num_scheduled_stops: Math.round(dist / 65),
    scheduled_travel_hours: dist / 65.0,
    track_doubled: 1,
    is_hdn_route: document.getElementById("check-hdn")?.checked ? 1 : 0,
    traction_type: traction,
    is_electrified: 1,
    psr_count: 5,
    is_circular_route: 0,
    is_monsoon_season: season === "Monsoon" ? 1 : 0,
    is_fog_risk: fog > 0.6 ? 1 : 0,
    fog_risk_score: fog,
    zone_fog_index: fog * 0.8,
    zone_congestion_index: 0.65,
    season_severity_score: season === "Monsoon" ? 0.9 : 0.4,
    loco_age_years: 5.5,
    coach_age_years: 4.2,
    has_lhb_coaches: document.getElementById("check-lhb")?.checked ? 1 : 0,
    is_rake_shared: document.getElementById("check-shared-rake")?.checked ? 1 : 0,
    maintenance_score: maint,
    seat_utilisation_pct: 95.0,
    is_overloaded: 0,
    late_incoming_rake: 0,
    is_special_train: 0,
    route_historical_ontime_pct: 82.0,
  };

  try {
    const res = await fetch("/predict-journey", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    if (!res.ok) {
      const err = await res.json();
      showToast(`Journey model: ${typeof err.detail === "string" ? err.detail : "invalid input"}`, "error");
      return;
    }

    const data = await res.json();
    const delayMin = data.predicted_destination_delay_minutes;
    document.getElementById("sim-predicted-minutes").innerHTML = `${delayMin} <span class="text-xl font-normal text-slate-400">mins</span>`;

    const badge = document.getElementById("sim-risk-badge");
    if (data.is_predicted_delayed) {
      badge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/30 mt-2";
      badge.textContent = `⚠️ High Delay Risk (>15m Late Arrival Threshold Exceeded)`;
    } else {
      badge.className = "px-3 py-1 rounded-full text-xs font-semibold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 mt-2";
      badge.textContent = `✅ On-Time Arrival Expected (Within 15m Threshold)`;
    }
  } catch (e) {
    console.error("Simulation error:", e);
  }
}
