import os
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

NEPAL_BOUNDS = {
    "minlatitude": 26.34,
    "maxlatitude": 30.42,
    "minlongitude": 80.05,
    "maxlongitude": 88.20
}

MOCK_NEPAL_DATA = [
    {
        "type": "Feature",
        "properties": {
            "category": "earthquake",
            "severity": "Extreme",
            "title": "Gorkha Epicenter (Simulation)",
            "desc": "Simulated tectonic rupture near Barpak. High structural damage radius.",
            "metric": "Magnitude: 7.8",
            "time": "2026-06-27 14:30:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [84.708, 28.147]}
    },
    {
        "type": "Feature",
        "properties": {
            "category": "earthquake",
            "severity": "Moderate",
            "title": "Jajarkot Tremor (Simulation)",
            "desc": "Shallow intra-plate seismic activity recorded across Western Nepal hills.",
            "metric": "Magnitude: 5.6",
            "time": "2026-06-27 15:12:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [82.19, 28.85]}
    },
    {
        "type": "Feature",
        "properties": {
            "category": "wildfire",
            "severity": "High",
            "title": "Chitwan Buffer Zone Fire (Simulation)",
            "desc": "Active canopy forest fire detected via satellite thermal anomalies. Rapid spread north.",
            "metric": "FRP: 85.2 MW",
            "time": "2026-06-27 13:45:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [84.45, 27.53]}
    },
    {
        "type": "Feature",
        "properties": {
            "category": "flood",
            "severity": "Extreme",
            "title": "Koshi River Inundation Zone (Simulation)",
            "desc": "Water levels crossed the critical safety thresholds at Chatara gauging station.",
            "metric": "Gauge: +6.4m",
            "time": "2026-06-27 12:20:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [87.15, 26.85]}
    }
]

@app.get("/api/disasters")
def get_disasters(demo: bool = False):
    usgs_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "minlatitude": NEPAL_BOUNDS["minlatitude"],
        "maxlatitude": NEPAL_BOUNDS["maxlatitude"],
        "minlongitude": NEPAL_BOUNDS["minlongitude"],
        "maxlongitude": NEPAL_BOUNDS["maxlongitude"],
        "minmagnitude": 2.0,
        "limit": 30
    }

    live_features = []
    try:
        r = requests.get(usgs_url, params=params, timeout=4)
        usgs_data = r.json().get("features", [])
        for eq in usgs_data:
            timestamp_ms = eq["properties"].get("time")
            if timestamp_ms:
                dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                time_str = "Unknown"

            live_features.append({
                "type": "Feature",
                "properties": {
                    "category": "earthquake",
                    "severity": "High" if eq["properties"]["mag"] >= 5.0 else "Moderate",
                    "title": eq["properties"]["place"],
                    "desc": "Real-time event detected by global networks.",
                    "metric": f"Magnitude: {eq['properties']['mag']}",
                    "time": time_str
                },
                "geometry": eq["geometry"]
            })
    except Exception:
        pass

    if demo:
        return {"type": "FeatureCollection", "features": live_features + MOCK_NEPAL_DATA}
    return {"type": "FeatureCollection", "features": live_features}

@app.post("/api/get-advice")
async def get_advice(request: Request):
    if not client:
        return {"advice": "AI Service not configured. Please check .env file."}

    data = await request.json()
    hazard_type = data.get("category", "incident")
    title = data.get("title", "the area")
    lang = data.get("language", "ne")

    lang_instruction = "Respond strictly in Nepali language using Devanagari script." if lang == "ne" else "Respond in English."

    prompt = f"""
    You are an expert disaster response AI for Nepal. 
    A {hazard_type} event is occurring at {title}. 
    Provide 3 bullet points of actionable safety advice for local residents in Nepal. 
    Be concise, practical, and prioritize immediate life-saving actions.
    {lang_instruction}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful disaster response assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        raw_advice = chat_completion.choices[0].message.content
        return {"advice": raw_advice}
    except Exception:
        fallback = "सुरक्षा प्रोटोकल: शान्त रहनुहोस्, बाढी प्रभावित क्षेत्र भए माथिल्लो स्थानमा जानुहोस्, वा भूकम्प प्रभावित क्षेत्र भए खुला स्थान खोज्नुहोस्।" if lang == "ne" else "Safety protocol: Remain calm, move to higher ground if flood-prone, or seek open spaces if earthquake-prone."
        return {"advice": fallback}

@app.post("/api/chat")
async def chat(request: Request):
    if not client:
        return {"reply": "AI Service not configured."}

    data = await request.json()
    user_message = data.get("message", "")
    cat = data.get("category", "incident")
    title = data.get("title", "the area")
    lang = data.get("language", "ne")

    lang_instruction = "Respond strictly in Nepali language using Devanagari script." if lang == "ne" else "Respond in English."

    prompt = f"""
    You are an expert disaster response AI for Nepal. 
    Context: A {cat} event is occurring at {title}.
    The user is asking a follow-up question: "{user_message}"
    Provide a concise, practical, and life-saving answer. Keep it under 3 sentences.
    {lang_instruction}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful disaster response assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        reply = chat_completion.choices[0].message.content
        return {"reply": reply}
    except Exception:
        return {"reply": "Failed to generate response."}

@app.get("/video")
def get_video():
    return FileResponse("earthquake.mp4", media_type="video/mp4")

@app.get("/")
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Smart Disaster Assistant</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.css" />

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.js"></script>

<style>
body { margin:0; font-family:sans-serif; background: #111827; color: #f3f4f6; }
#map { height:100vh; position: relative; z-index: 1; }

#panel {
  position:absolute; top:20px; left:20px; z-index:999;
  background:#1f2937; padding:20px; border-radius:12px; width:290px;
  box-shadow:0 10px 30px rgba(0,0,0,0.5); border: 1px solid #374151;
}
.panel-title { font-size:18px; font-weight:bold; margin-bottom:4px; color:#fff; }
.mode-container { margin: 15px 0; padding-bottom: 12px; border-bottom: 1px solid #374151; }
.switch-label { display: flex; align-items: center; justify-content: space-between; font-size: 13px; cursor: pointer; }
.status-indicator { font-size: 11px; margin-top: 6px; color: #9ca3af; font-weight: bold; }

.filter-section { margin-top: 15px; }
.filter-title { font-size: 12px; text-transform: uppercase; color: #9ca3af; letter-spacing: 0.05em; margin-bottom: 8px; font-weight: bold; }
.filter-btn { 
  background: #2d3748; border: 1px solid #4a5568; color: #fff; padding: 6px 10px; 
  font-size: 12px; border-radius: 6px; cursor: pointer; margin-right: 4px; margin-bottom: 6px; display: inline-block;
}
.filter-btn.active { background: #3b82f6; border-color: #60a5fa; }

.slider-container { margin-top: 15px; padding-top: 12px; border-top: 1px solid #374151; }

.pulse-earthquake { background:#ef4444; border-radius:50%; width:12px; height:12px; animation: p-eq 1.5s infinite; }
.pulse-wildfire { background:#f97316; border-radius:50%; width:12px; height:12px; animation: p-wf 1.5s infinite; }
.pulse-flood { background:#3b82f6; border-radius:50%; width:12px; height:12px; animation: p-fl 1.5s infinite; }
.pulse-landslide { background:#a855f7; border-radius:50%; width:12px; height:12px; animation: p-ls 1.5s infinite; }

@keyframes p-eq { 0% { box-shadow:0 0 0 0 rgba(239,68,68,0.7); } 70% { box-shadow:0 0 0 15px rgba(239,68,68,0); } 100% { box-shadow:0 0 0 0 rgba(239,68,68,0); } }
@keyframes p-wf { 0% { box-shadow:0 0 0 0 rgba(249,115,22,0.7); } 70% { box-shadow:0 0 0 15px rgba(249,115,22,0); } 100% { box-shadow:0 0 0 0 rgba(249,115,22,0); } }
@keyframes p-fl { 0% { box-shadow:0 0 0 0 rgba(59,130,246,0.7); } 70% { box-shadow:0 0 0 15px rgba(59,130,246,0); } 100% { box-shadow:0 0 0 0 rgba(59,130,246,0); } }
@keyframes p-ls { 0% { box-shadow:0 0 0 0 rgba(168,85,247,0.7); } 70% { box-shadow:0 0 0 15px rgba(168,85,247,0); } 100% { box-shadow:0 0 0 0 rgba(168,85,247,0); } }

.ai-result p, .chat-log p { margin: 6px 0; line-height: 1.4; }
.ai-result ul, .chat-log ul { margin: 6px 0; padding-left: 20px; }
.ai-result li, .chat-log li { margin-bottom: 4px; line-height: 1.4; }
.ai-result strong, .chat-log strong { color: #111827; }

.talk-btn { transition: all 0.2s; }
.talk-btn:hover { filter: brightness(0.9); }
.evac-btn:hover { background: #dc2626 !important; }

/* Customizing the Leaflet Routing Machine Panel to match dark mode */
.leaflet-routing-container { background-color: #1f2937 !important; color: #f3f4f6 !important; border-radius: 8px !important; border: 1px solid #374151 !important; }
.leaflet-routing-alt { border-bottom: 1px solid #374151 !important; }
.leaflet-routing-alt h2 { color: #10b981 !important; }

#emergency-modal {
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.95);
    z-index: 99999;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}
#emergency-modal.active { display: flex; }
#video-container { width: 80%; max-width: 640px; aspect-ratio: 16/9; background: #000; }
#calm-text {
    font-size: 4.5rem; color: #10b981; text-align: center; font-weight: 900;
    text-shadow: 0 0 40px rgba(16, 185, 129, 0.6); display: none; line-height: 1.3; letter-spacing: 2px;
}
#calm-text.active { display: block; animation: pulse-calm 2.5s infinite ease-in-out; }
@keyframes pulse-calm { 0% { opacity: 0.6; transform: scale(0.98); } 50% { opacity: 1; transform: scale(1.02); } 100% { opacity: 0.6; transform: scale(0.98); } }
</style>
</head>

<body>

<div id="panel">
  <div class="panel-title">Smart Disaster Assistant</div>
  <div style="font-size:12px; color:#9ca3af;">Nepal Multi-Hazard Early Warning Dashboard</div>

  <div class="mode-container">
    <label class="switch-label">
      <span><b>Simulation Environment</b></span>
      <input id="demoToggle" type="checkbox">
    </label>
    <div id="statusText" class="status-indicator">Connecting to Real-Time USGS Feed...</div>
  </div>

  <div class="mode-container">
    <label class="switch-label">
      <span><b>Auto-Voice Responses</b></span>
      <input id="autoVoiceToggle" type="checkbox" checked>
    </label>
  </div>

  <div class="mode-container" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
    <div style="font-size: 12px; color: #9ca3af; margin-bottom: 6px; font-weight: bold;">AI Response Language</div>
    <div style="display:flex; gap:6px;">
        <button id="langNe" class="filter-btn active" style="flex:1; text-align:center; margin:0;" onclick="setLang('ne', this)">नेपाली</button>
        <button id="langEn" class="filter-btn" style="flex:1; text-align:center; margin:0;" onclick="setLang('en', this)">English</button>
    </div>
  </div>

  <div class="filter-section">
    <div class="filter-title">Hazard Filter</div>
    <div id="filters">
      <span class="filter-btn active" onclick="setFilter('all', this)">All</span>
      <span class="filter-btn" onclick="setFilter('earthquake', this)">Seismic</span>
      <span class="filter-btn" onclick="setFilter('flood', this)">Floods</span>
      <span class="filter-btn" onclick="setFilter('wildfire', this)">Fires</span>
      <span class="filter-btn" onclick="setFilter('landslide', this)">Landslide</span>
    </div>
  </div>

  <div class="slider-container">
    <input id="slider" type="range" min="1" max="50" value="50" style="width: 100%; cursor:pointer;">
    <div style="font-size:11px; margin-top:4px; color:#9ca3af; display:flex; justify-content:space-between;">
      <span>Oldest Incidents</span>
      <span>Most Recent</span>
    </div>
  </div>
</div>

<div id="map"></div>

<div id="emergency-modal">
    <div id="video-container"></div>
    <div id="calm-text">DEEP BREATH<br>STAY CALM</div>
</div>

<script>
const map = L.map('map', {attributionControl: false}).setView([28.25, 84.3], 7);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  maxZoom: 18
}).addTo(map);

const nepalBounds = L.latLngBounds([26.34, 80.05], [30.42, 88.20]);
map.setMaxBounds(nepalBounds);

let allEvents = [];
let currentFilter = 'all';
let dataLayers = L.layerGroup().addTo(map);
let heatLayer;
let autoVoiceEnabled = true;
let isListening = false;
let simulationTimer = null;
let currentLang = 'ne';
let routingControl = null; // Store active route

const HAZARD_CONFIG = {
    earthquake: { color: '#ef4444', pulse: 'pulse-earthquake', baseRadius: 25 },
    wildfire:   { color: '#f97316', pulse: 'pulse-wildfire',   baseRadius: 15 },
    flood:      { color: '#3b82f6', pulse: 'pulse-flood',      baseRadius: 20 },
    landslide:  { color: '#a855f7', pulse: 'pulse-landslide',  baseRadius: 10 }
};

let recognition = null;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false; 
    recognition.interimResults = false;
    recognition.lang = 'ne-NP'; 
}

function setLang(lang, btn) {
    currentLang = lang;
    document.querySelectorAll('#langNe, #langEn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (recognition) {
        recognition.lang = lang === 'ne' ? 'ne-NP' : 'en-US';
    }
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); 
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = marked.parse(text); 
        const cleanText = tempDiv.textContent || tempDiv.innerText || "";

        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = currentLang === 'ne' ? 'ne-NP' : 'en-US';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

function showEarthquakeDrill() {
    const modal = document.getElementById('emergency-modal');
    const videoContainer = document.getElementById('video-container');
    const calmText = document.getElementById('calm-text');

    if (modal.classList.contains('active')) return;

    modal.classList.add('active');
    videoContainer.style.display = 'block';
    calmText.classList.remove('active');

    videoContainer.innerHTML = `
        <video id="local-vid" style="width:100%; height:100%; object-fit:contain;" autoplay playsinline>
            <source src="/video" type="video/mp4">
        </video>
    `;

    const vid = document.getElementById('local-vid');
    vid.onended = () => {
        videoContainer.style.display = 'none';
        calmText.classList.add('active');
        setTimeout(() => {
            modal.classList.remove('active');
            calmText.classList.remove('active');
        }, 5000);
    };
}

// Distance Calculator Helper (Haversine Formula)
function getDistanceKM(lat1, lon1, lat2, lon2) {
    const R = 6371; 
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Evacuation Routing Feature
async function findNearestHospital(startLat, startLng, btnElement) {
    const originalText = btnElement.innerText;
    btnElement.innerText = "⏳ Searching for safety...";
    btnElement.disabled = true;

    // Overpass API query: Find hospitals or clinics within 15km
    const radius = 15000;
    const query = `
        [out:json];
        (
          node["amenity"="hospital"](around:${radius},${startLat},${startLng});
          way["amenity"="hospital"](around:${radius},${startLat},${startLng});
          node["amenity"="clinic"](around:${radius},${startLat},${startLng});
        );
        out center;
    `;
    const overpassUrl = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;

    try {
        const response = await fetch(overpassUrl);
        const data = await response.json();

        if (data.elements && data.elements.length > 0) {
            // Find the absolute closest facility
            let nearest = null;
            let minDistance = Infinity;

            data.elements.forEach(el => {
                let hLat = el.lat || el.center.lat;
                let hLon = el.lon || el.center.lon;
                let dist = getDistanceKM(startLat, startLng, hLat, hLon);
                if (dist < minDistance) {
                    minDistance = dist;
                    nearest = el;
                }
            });

            let hLat = nearest.lat || nearest.center.lat;
            let hLon = nearest.lon || nearest.center.lon;
            let hName = nearest.tags.name || "Emergency Medical Facility";

            if (routingControl) {
                map.removeControl(routingControl);
            }

            // Draw the route using OSRM
            routingControl = L.Routing.control({
                waypoints: [
                    L.latLng(startLat, startLng), 
                    L.latLng(hLat, hLon)
                ],
                routeWhileDragging: false,
                addWaypoints: false,
                fitSelectedRoutes: true,
                showAlternatives: false,
                lineOptions: {
                    styles: [{color: '#10b981', weight: 6, opacity: 0.9}]
                }
            }).addTo(map);

            alert(`✅ Found: ${hName} (${minDistance.toFixed(1)} km away).\nDrawing evacuation route on map.`);
        } else {
            alert("⚠️ No hospitals or clinics found within a 15km radius.");
        }
    } catch (err) {
        console.error(err);
        alert("⚠️ Failed to connect to OpenStreetMap servers.");
    } finally {
        btnElement.innerText = originalText;
        btnElement.disabled = false;
        map.closePopup(); // Close popup so user can see the route
    }
}


map.on('popupopen', function(e) {
    const container = e.popup.getElement();
    if (!container) return;
    
    const aiWrapper = container.querySelector('.ai-wrapper');
    if (!aiWrapper || aiWrapper.dataset.fetched === 'true') return;
    
    aiWrapper.dataset.fetched = 'true';
    const cat = aiWrapper.getAttribute('data-cat');
    const title = aiWrapper.getAttribute('data-title');
    const resultDiv = aiWrapper.querySelector('.ai-result');
    const chatContainer = container.querySelector('.chat-container');

    if (cat === 'earthquake') {
        showEarthquakeDrill();
    }

    fetch('/api/get-advice', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ category: cat, title: title, language: currentLang })
    })
    .then(res => res.json())
    .then(data => {
        resultDiv.innerHTML = marked.parse(data.advice);
        if (chatContainer) chatContainer.style.display = 'block';
        if (autoVoiceEnabled) {
            speakText(data.advice);
        }
    })
    .catch(err => {
        resultDiv.innerHTML = "Failed to connect to AI.";
    });
});

function injectCollegeEarthquake() {
    const newEvent = {
        "type": "Feature",
        "properties": {
            "category": "earthquake",
            "severity": "Extreme",
            "title": "Cosmos College Epicenter (Simulation)",
            "desc": "CRITICAL: Severe tectonic rupture detected directly beneath your location. Immediate structural failure imminent.",
            "metric": "Magnitude: 7.2",
            "time": new Date().toISOString().replace('T', ' ').substring(0, 19) + " UTC"
        },
        "geometry": {"type": "Point", "coordinates": [85.2843652, 27.7079551]}
    };
    
    allEvents.push(newEvent);
    applyFilterAndRender();
    
    map.setView([27.7079551, 85.2843652], 18);
    
    setTimeout(() => {
        dataLayers.eachLayer((layer) => {
            if (layer instanceof L.Marker) {
                const latLng = layer.getLatLng();
                if (Math.abs(latLng.lat - 27.7079551) < 0.0001 && Math.abs(latLng.lng - 85.2843652) < 0.0001) {
                    layer.openPopup();
                }
            }
        });
    }, 500);
}

async function sendChatMessage(container) {
    const input = container.querySelector('.chat-input');
    const log = container.querySelector('.chat-log');
    const message = input.value.trim();
    if (!message) return;

    const cat = container.getAttribute('data-cat');
    const title = container.getAttribute('data-title');

    const userDiv = document.createElement('div');
    userDiv.style.marginBottom = '8px';
    userDiv.innerHTML = `<b>You:</b> ${message}`;
    log.appendChild(userDiv);
    input.value = '';
    log.scrollTop = log.scrollHeight;

    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.style.marginBottom = '8px';
    typingDiv.style.color = '#6b7280';
    typingDiv.innerHTML = `<b>AI:</b> <i>Thinking...</i>`;
    log.appendChild(typingDiv);
    log.scrollTop = log.scrollHeight;

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: message, category: cat, title: title, language: currentLang })
        });
        const data = await res.json();
        
        const tEl = document.getElementById(typingId);
        if(tEl) tEl.remove();

        const replyHtml = marked.parse(data.reply);
        const replyDiv = document.createElement('div');
        replyDiv.style.marginBottom = '8px';
        replyDiv.style.color = '#059669';
        replyDiv.innerHTML = `<b>AI:</b> ${replyHtml}`;
        log.appendChild(replyDiv);
        log.scrollTop = log.scrollHeight;

        if (autoVoiceEnabled) {
            speakText(data.reply);
        }
    } catch (err) {
        const tEl = document.getElementById(typingId);
        if(tEl) tEl.remove();
        const errDiv = document.createElement('div');
        errDiv.style.marginBottom = '8px';
        errDiv.style.color = '#dc2626';
        errDiv.innerHTML = `<b>AI:</b> Failed to connect.`;
        log.appendChild(errDiv);
    }
}

function loadData(isDemoMode) {
    const statusText = document.getElementById('statusText');
    
    // Clear any existing route when switching modes
    if (routingControl) { map.removeControl(routingControl); routingControl = null; }

    if (isDemoMode) {
        statusText.innerHTML = "SIMULATION ACTIVE (Multi-Hazard Injected)";
        statusText.style.color = "#a855f7";
        
        if (simulationTimer) clearTimeout(simulationTimer);
        simulationTimer = setTimeout(() => {
            injectCollegeEarthquake();
        }, 40000);
    } else {
        statusText.innerHTML = "LIVE TELEMETRY (Awaiting USGS Events)";
        statusText.style.color = "#10b981";
        if (simulationTimer) {
            clearTimeout(simulationTimer);
            simulationTimer = null;
        }
    }

    fetch(`/api/disasters?demo=${isDemoMode}`)
    .then(r => r.json())
    .then(data => {
        allEvents = data.features || [];
        applyFilterAndRender();
    });
}

function setFilter(type, clickedBtn) {
    currentFilter = type;
    const buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    clickedBtn.classList.add('active');
    applyFilterAndRender();
}

function applyFilterAndRender() {
    let filtered = allEvents;
    if (currentFilter !== 'all') {
        filtered = allEvents.filter(e => e.properties.category === currentFilter);
    }

    const slider = document.getElementById('slider');
    slider.max = filtered.length > 0 ? filtered.length : 1;
    const count = (currentFilter === 'all') ? filtered.length : (parseInt(slider.value) || 1);
    render(filtered.slice(0, count));
}

function render(events) {
    dataLayers.clearLayers();
    let heatPoints = [];

    events.forEach(e => {
        const [lng, lat] = e.geometry.coordinates;
        const props = e.properties;
        const cat = props.category;
        const config = HAZARD_CONFIG[cat] || { color: '#ffffff', pulse: 'pulse-earthquake', baseRadius: 10 };

        L.circleMarker([lat, lng], {
            radius: config.baseRadius, color: config.color, fillColor: config.color, fillOpacity: 0.15, weight: 1.5
        }).addTo(dataLayers);

        const marker = L.marker([lat, lng], {
            icon: L.divIcon({ className: config.pulse, iconSize: [12, 12] })
        }).addTo(dataLayers);

        const safeTitle = props.title.replace(/"/g, '&quot;');
        const safeTime = props.time || "Unknown";

        marker.bindPopup(`
            <div style="color:#111827; font-family:sans-serif; width:100%;">
                <b style="font-size:16px; text-transform:uppercase; color:${config.color}">${cat} Detected</b><br>
                <b style="font-size:14px; display:block; margin-top:4px;">${props.title}</b>
                <span style="font-size:12px; color:#6b7280; display:block; margin-top:2px;">Time: ${safeTime}</span>
                <span style="font-size:14px; color:#4b5563; display:block; margin-top:6px;">${props.desc}</span>
                <hr style="margin:10px 0; border:0; border-top:1px solid #e5e7eb;">
                <span style="font-size:14px; font-weight:bold; color:#1f2937;">${props.metric}</span>
                
                <button class="evac-btn" onclick="findNearestHospital(${lat}, ${lng}, this)" style="width:100%; padding:10px; background:#ef4444; color:white; font-size:14px; font-weight:bold; border:none; border-radius:8px; margin-top:12px; cursor:pointer; transition: 0.2s;">
                    🏥 Find Nearest Hospital & Route
                </button>

                <div class="ai-wrapper" data-cat="${cat}" data-title="${safeTitle}" style="margin-top:12px;">
                    <div class="ai-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <b style="font-size:13px; color:#059669;">AI Safety Advice</b>
                        <button class="tts-btn" style="background:transparent; border:none; font-size:18px; cursor:pointer; color:#4b5563;" title="Listen again">🔊</button>
                    </div>
                    <div class="ai-result" style="font-size:14px; color:#333; max-height:250px; overflow-y:auto; line-height:1.5;">Fetching safety protocol...</div>
                </div>

                <div class="chat-container" data-cat="${cat}" data-title="${safeTitle}" style="margin-top:12px; display:none; border-top:1px dashed #d1d5db; padding-top:10px;">
                    <div class="chat-log" style="max-height:250px; overflow-y:auto; font-size:14px; margin-bottom:8px; background:#f3f4f6; padding:10px; border-radius:6px; color:#374151;"></div>
                    <div style="display:flex; gap:8px; margin-top:8px;">
                        <input type="text" class="chat-input" placeholder="Or type here..." style="flex:1; padding:10px; font-size:14px; border:1px solid #d1d5db; border-radius:8px; outline:none;">
                        <button class="chat-send" style="padding:10px 14px; font-size:14px; background:#3b82f6; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold;">Send</button>
                        <button class="talk-btn" style="padding:10px 16px; font-size:16px; background:#10b981; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold; display:flex; align-items:center; gap:6px;">
                            <span class="talk-icon">🎤</span> <span class="talk-text">Talk</span>
                        </button>
                    </div>
                </div>
            </div>
        `, { maxWidth: 500, minWidth: 400 });

        heatPoints.push([lat, lng, 0.5]);
    });

    if (heatLayer) map.removeLayer(heatLayer);
    heatLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15, maxZoom: 10 }).addTo(map);
}

document.addEventListener('click', async function(e) {
    if (e.target && e.target.classList.contains('chat-send')) {
        const container = e.target.closest('.chat-container');
        sendChatMessage(container);
    }

    if (e.target && (e.target.classList.contains('talk-btn') || e.target.closest('.talk-btn'))) {
        const talkBtn = e.target.closest('.talk-btn');
        const container = talkBtn.closest('.chat-container');
        const input = container.querySelector('.chat-input');
        const icon = talkBtn.querySelector('.talk-icon');
        const textSpan = talkBtn.querySelector('.talk-text');

        if (!recognition) {
            alert("Voice input is not supported. Please allow microphone permissions in Brave settings.");
            return;
        }

        if (isListening) {
            try { recognition.stop(); } catch(err) {}
            return;
        }

        isListening = true;
        icon.innerText = "⏺️";
        textSpan.innerText = "Listening...";
        talkBtn.style.background = "#dc2626";

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            input.value = transcript;
            
            setTimeout(() => {
                sendChatMessage(container);
            }, 600);
        };

        recognition.onend = () => {
            isListening = false;
            icon.innerText = "🎤";
            textSpan.innerText = "Talk";
            talkBtn.style.background = "#10b981";
        };

        recognition.onerror = (event) => {
            isListening = false;
            icon.innerText = "🎤";
            textSpan.innerText = "Talk";
            talkBtn.style.background = "#10b981";
            if(event.error === 'not-allowed') {
                alert("Microphone access blocked. Click the 🔒 icon next to the URL in your browser to allow it.");
            }
        };

        try {
            recognition.start();
        } catch (err) {
            console.error("Recognition start error", err);
            isListening = false;
            icon.innerText = "🎤";
            textSpan.innerText = "Talk";
            talkBtn.style.background = "#10b981";
        }
    }

    if (e.target && e.target.classList.contains('tts-btn')) {
        const popup = e.target.closest('.leaflet-popup-content');
        const resultDiv = popup.querySelector('.ai-result');
        speakText(resultDiv.innerText);
    }
});

document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && e.target.classList.contains('chat-input')) {
        const container = e.target.closest('.chat-container');
        sendChatMessage(container);
    }
});

document.getElementById('demoToggle').addEventListener('change', (e) => {
    loadData(e.target.checked);
});

document.getElementById('autoVoiceToggle').addEventListener('change', (e) => {
    autoVoiceEnabled = e.target.checked;
    if (!autoVoiceEnabled) {
        window.speechSynthesis.cancel(); 
    }
});

document.getElementById('slider').addEventListener('input', applyFilterAndRender);

loadData(False);
</script>

</body>
</html>
""")
