import os
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
            "category": "wildfire",
            "severity": "Low",
            "title": "Bardiya National Park Bushfire (Simulation)",
            "desc": "Undergrowth ground fire detected. Low immediate risk to heavy timber.",
            "metric": "FRP: 12.0 MW",
            "time": "2026-06-27 16:05:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [81.35, 28.45]}
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
    },
    {
        "type": "Feature",
        "properties": {
            "category": "flood",
            "severity": "High",
            "title": "Narayani River High Swell (Simulation)",
            "desc": "Flash flood warning active for low-lying agricultural plains of Chitwan/Nawalparasi.",
            "metric": "Gauge: +4.1m",
            "time": "2026-06-27 14:55:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [84.41, 27.68]}
    },
    {
        "type": "Feature",
        "properties": {
            "category": "landslide",
            "severity": "High",
            "title": "Mugling Highway Mass Wasting (Simulation)",
            "desc": "Slope structural failure blocking major freight artery. Road clearance underway.",
            "metric": "Volume: Massive",
            "time": "2026-06-27 17:10:00 UTC"
        },
        "geometry": {"type": "Point", "coordinates": [84.56, 27.85]}
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

    prompt = f"""
    You are an expert disaster response AI for Nepal. 
    A {hazard_type} event is occurring at {title}. 
    Provide 3 bullet points of actionable safety advice for local residents in Nepal. 
    Be concise, practical, and prioritize immediate life-saving actions.
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
        return {"advice": raw_advice.replace('\n', '<br>')}
    except Exception:
        return {"advice": "Safety protocol: Remain calm, move to higher ground if flood-prone, or seek open spaces if earthquake-prone. Stay updated via local radio."}

@app.post("/api/chat")
async def chat(request: Request):
    if not client:
        return {"reply": "AI Service not configured."}

    data = await request.json()
    user_message = data.get("message", "")
    cat = data.get("category", "incident")
    title = data.get("title", "the area")

    prompt = f"""
    You are an expert disaster response AI for Nepal. 
    Context: A {cat} event is occurring at {title}.
    The user is asking a follow-up question: "{user_message}"
    Provide a concise, practical, and life-saving answer. Keep it under 3 sentences.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful disaster response assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        reply = chat_completion.choices[0].message.content.replace('\n', '<br>')
        return {"reply": reply}
    except Exception:
        return {"reply": "Failed to generate response."}

@app.get("/")
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Smart Disaster Assistant</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>

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

const HAZARD_CONFIG = {
    earthquake: { color: '#ef4444', pulse: 'pulse-earthquake', baseRadius: 15000 },
    wildfire:   { color: '#f97316', pulse: 'pulse-wildfire',   baseRadius: 8000 },
    flood:      { color: '#3b82f6', pulse: 'pulse-flood',      baseRadius: 12000 },
    landslide:  { color: '#a855f7', pulse: 'pulse-landslide',  baseRadius: 5000 }
};

function loadData(isDemoMode) {
    const statusText = document.getElementById('statusText');
    if (isDemoMode) {
        statusText.innerHTML = "SIMULATION ACTIVE (Multi-Hazard Injected)";
        statusText.style.color = "#a855f7";
    } else {
        statusText.innerHTML = "LIVE TELEMETRY (Awaiting USGS Events)";
        statusText.style.color = "#10b981";
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

        const config = HAZARD_CONFIG[cat] || { color: '#ffffff', pulse: 'pulse-earthquake', baseRadius: 5000 };

        L.circle([lat, lng], {
            radius: config.baseRadius,
            color: config.color,
            fillColor: config.color,
            fillOpacity: 0.15,
            weight: 1.5
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

                <button class="ai-btn" data-cat="${cat}" data-title="${safeTitle}" style="margin-top:12px; padding:10px; background:${config.color}; color:white; border:none; border-radius:6px; cursor:pointer; width:100%; font-size:14px; font-weight:bold;">
                    Get AI Advice
                </button>

                <div class="ai-result" style="margin-top:10px; font-size:14px; color:#333; display:none; max-height:250px; overflow-y:auto; line-height:1.5;"></div>

                <div class="chat-container" data-cat="${cat}" data-title="${safeTitle}" style="margin-top:12px; display:none; border-top:1px dashed #d1d5db; padding-top:10px;">
                    <div class="chat-log" style="max-height:250px; overflow-y:auto; font-size:14px; margin-bottom:8px; background:#f3f4f6; padding:10px; border-radius:6px; color:#374151;"></div>
                    <div style="display:flex; gap:6px;">
                        <input type="text" class="chat-input" placeholder="Ask follow-up..." style="flex:1; padding:8px; font-size:14px; border:1px solid #d1d5db; border-radius:6px; outline:none;">
                        <button class="chat-send" style="padding:8px 12px; font-size:14px; background:#3b82f6; color:white; border:none; border-radius:6px; cursor:pointer; font-weight:bold;">Send</button>
                    </div>
                </div>
            </div>
        `, {
            maxWidth: 500,
            minWidth: 400
        });

        heatPoints.push([lat, lng, 0.5]);
    });

    if (heatLayer) map.removeLayer(heatLayer);

    heatLayer = L.heatLayer(heatPoints, {
        radius: 25,
        blur: 15,
        maxZoom: 10
    }).addTo(map);
}

document.addEventListener('click', async function(e) {
    if (e.target && e.target.classList.contains('ai-btn')) {
        const btn = e.target;
        const cat = btn.getAttribute('data-cat');
        const title = btn.getAttribute('data-title');
        const resultDiv = btn.nextElementSibling;
        const chatContainer = resultDiv.nextElementSibling;

        btn.innerText = "Thinking...";
        btn.disabled = true;
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = "Fetching safety protocol...";

        try {
            const res = await fetch('/api/get-advice', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ category: cat, title: title })
            });
            const data = await res.json();
            resultDiv.innerHTML = data.advice;
            btn.innerText = "Advice Received";
            btn.style.background = "#10b981";
            chatContainer.style.display = 'block';
        } catch (err) {
            resultDiv.innerHTML = "Failed to connect to AI.";
            btn.innerText = "Retry";
            btn.disabled = false;
        }
    }

    if (e.target && e.target.classList.contains('chat-send')) {
        const btn = e.target;
        const container = btn.closest('.chat-container');
        const input = container.querySelector('.chat-input');
        const log = container.querySelector('.chat-log');
        const message = input.value.trim();

        if (!message) return;

        const cat = container.getAttribute('data-cat');
        const title = container.getAttribute('data-title');

        log.innerHTML += `<div style="margin-bottom:8px;"><b>You:</b> ${message}</div>`;
        input.value = '';
        btn.disabled = true;
        btn.innerText = "...";

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: message, category: cat, title: title })
            });
            const data = await res.json();
            log.innerHTML += `<div style="margin-bottom:8px; color:#059669;"><b>AI:</b> ${data.reply}</div>`;
            log.scrollTop = log.scrollHeight;
        } catch (err) {
            log.innerHTML += `<div style="margin-bottom:8px; color:#dc2626;"><b>AI:</b> Failed to connect.</div>`;
        }

        btn.disabled = false;
        btn.innerText = "Send";
    }
});

document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && e.target.classList.contains('chat-input')) {
        const container = e.target.closest('.chat-container');
        const sendBtn = container.querySelector('.chat-send');
        sendBtn.click();
    }
});

document.getElementById('demoToggle').addEventListener('change', (e) => {
    loadData(e.target.checked);
});

document.getElementById('slider').addEventListener('input', applyFilterAndRender);

loadData(false);
</script>

</body>
</html>
""")