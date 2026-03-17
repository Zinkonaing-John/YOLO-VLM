/* ── Edge Defect Inspector — Frontend Application ────────────────────── */

const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/live`;

let ws = null;
let reconnectTimer = null;
const recentResults = [];
const MAX_RECENT = 50;

/* ── WebSocket ───────────────────────────────────────────────────────── */

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        setBadge(true);
        if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleInspectionResult(data);
        } catch (e) {
            console.error("WS parse error:", e);
        }
    };

    ws.onclose = () => {
        setBadge(false);
        scheduleReconnect();
    };

    ws.onerror = () => {
        ws.close();
    };

    // Keepalive
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, 15000);
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setInterval(() => {
        connectWebSocket();
    }, 3000);
}

function setBadge(connected) {
    const badge = document.getElementById("connection-badge");
    badge.textContent = connected ? "Connected" : "Disconnected";
    badge.className = "badge " + (connected ? "badge-connected" : "badge-disconnected");
}

/* ── Handle live inspection result ───────────────────────────────────── */

function handleInspectionResult(data) {
    // Update live image
    const img = document.getElementById("live-image");
    const placeholder = document.getElementById("live-placeholder");
    if (data.image_url) {
        img.src = data.image_url;
        img.style.display = "block";
        placeholder.style.display = "none";
    }

    // Draw bounding boxes
    drawDetections("live-canvas", "live-image", data.detections || []);

    // Verdict badge
    const verdictBadge = document.getElementById("live-verdict");
    verdictBadge.textContent = data.verdict;
    verdictBadge.className = "verdict-badge verdict-" + data.verdict.toLowerCase();

    // Metrics
    setText("metric-fps", "--");
    setText("metric-pipeline", data.pipeline_ms ? data.pipeline_ms.toFixed(1) : "--");
    setText("metric-defects", data.defect_count ?? "--");
    const mv = document.getElementById("metric-verdict");
    mv.textContent = data.verdict;
    mv.style.color = data.verdict === "NG" ? "var(--ng)" : "var(--ok)";

    // Meta
    setText("live-meta",
        `ID: ${(data.inspection_id || "").slice(0, 8)} | ` +
        `Detections: ${data.total_detections || 0} | ` +
        `Pipeline: ${(data.pipeline_ms || 0).toFixed(1)}ms`
    );

    // Detection list
    renderDetectionList(data.detections || []);

    // Add to recent feed
    recentResults.unshift(data);
    if (recentResults.length > MAX_RECENT) recentResults.pop();
    renderResultsFeed();
}

/* ── Draw bounding boxes on canvas ───────────────────────────────────── */

function drawDetections(canvasId, imgId, detections) {
    const canvas = document.getElementById(canvasId);
    const img = document.getElementById(imgId);
    if (!canvas || !img) return;

    // Wait for image to load to get correct dimensions
    const draw = () => {
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.height = rect.height + "px";

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const det of detections) {
            const norm = det.bbox_norm || {};
            const x1 = norm.x1 * canvas.width;
            const y1 = norm.y1 * canvas.height;
            const x2 = norm.x2 * canvas.width;
            const y2 = norm.y2 * canvas.height;
            const w = x2 - x1;
            const h = y2 - y1;

            const isNG = det.roi_verdict === "NG";
            const color = isNG ? "#ef4444" : "#22c55e";

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, w, h);

            // Label
            const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}% ${det.roi_verdict || ""}`;
            ctx.font = "12px sans-serif";
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - 18, tw + 8, 18);
            ctx.fillStyle = "#fff";
            ctx.fillText(label, x1 + 4, y1 - 5);
        }
    };

    if (img.complete) { draw(); } else { img.onload = draw; }
}

/* ── Detection list panel ────────────────────────────────────────────── */

function renderDetectionList(detections) {
    const el = document.getElementById("detection-list");
    if (!detections.length) {
        el.innerHTML = '<p class="placeholder-text">No detections</p>';
        return;
    }
    el.innerHTML = detections.map(d => {
        const v = d.roi_verdict || "NG";
        return `<div class="detection-item ${v.toLowerCase()}">
            <span>${d.class_name} (${(d.confidence * 100).toFixed(1)}%)</span>
            <span>${v} ${d.roi_confidence ? (d.roi_confidence * 100).toFixed(0) + "%" : ""}</span>
        </div>`;
    }).join("");
}

/* ── Recent results feed ─────────────────────────────────────────────── */

function renderResultsFeed() {
    const el = document.getElementById("results-feed");
    el.innerHTML = recentResults.slice(0, 20).map(r => {
        const v = r.verdict;
        const time = new Date(r.timestamp * 1000).toLocaleTimeString();
        return `<div class="result-entry" onclick="showModal(${JSON.stringify(r).replace(/"/g, '&quot;')})">
            ${r.image_url ? `<img class="result-thumb" src="${r.image_url}">` : ""}
            <span class="result-verdict ${v.toLowerCase()}">${v}</span>
            <span>${r.defect_count || 0} defects</span>
            <span style="color:var(--text-muted);margin-left:auto">${time}</span>
        </div>`;
    }).join("");
}

/* ── Tab switching ───────────────────────────────────────────────────── */

function switchTab(name) {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.toggle("active", t.id === "tab-" + name));

    if (name === "stats") loadStatistics();
    if (name === "gallery") loadGallery();
}

/* ── Statistics ──────────────────────────────────────────────────────── */

async function loadStatistics() {
    try {
        const res = await fetch(`${API_BASE}/api/statistics`);
        const data = await res.json();

        setText("stat-total", data.total);
        setText("stat-ok", data.ok);
        setText("stat-ok-rate", data.ok_rate + "%");
        setText("stat-ng", data.ng);
        setText("stat-ng-rate", data.ng_rate + "%");
        setText("stat-pipeline", data.avg_pipeline_ms);

        renderDefectChart(data.defect_classes || []);
        renderTrendChart(data.hourly_trend || []);
    } catch (e) {
        console.error("Failed to load stats:", e);
    }
}

function renderDefectChart(classes) {
    const el = document.getElementById("defect-chart");
    if (!classes.length) {
        el.innerHTML = '<p class="placeholder-text">No defects recorded</p>';
        return;
    }
    const max = Math.max(...classes.map(c => c.count));
    el.innerHTML = classes.map(c => {
        const pct = (c.count / max * 100).toFixed(0);
        return `<div class="bar-row">
            <span class="bar-label">${c.class_name}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
            <span class="bar-value">${c.count}</span>
        </div>`;
    }).join("");
}

function renderTrendChart(data) {
    const canvas = document.getElementById("trend-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 20, right: 20, bottom: 40, left: 50 };

    ctx.clearRect(0, 0, W, H);

    if (!data.length) {
        ctx.fillStyle = "#8b8fa3";
        ctx.font = "14px sans-serif";
        ctx.fillText("No data yet", W / 2 - 30, H / 2);
        return;
    }

    // Aggregate by hour
    const hours = {};
    data.forEach(d => {
        const key = d.hour_ts;
        if (!hours[key]) hours[key] = { ok: 0, ng: 0 };
        if (d.verdict === "OK") hours[key].ok += d.count;
        else hours[key].ng += d.count;
    });

    const keys = Object.keys(hours).sort((a, b) => a - b);
    const maxVal = Math.max(...keys.map(k => hours[k].ok + hours[k].ng), 1);

    const barW = Math.max(4, (W - pad.left - pad.right) / keys.length - 4);
    const plotH = H - pad.top - pad.bottom;

    // Axes
    ctx.strokeStyle = "#2a2e3a";
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, H - pad.bottom);
    ctx.lineTo(W - pad.right, H - pad.bottom);
    ctx.stroke();

    keys.forEach((key, i) => {
        const x = pad.left + i * ((W - pad.left - pad.right) / keys.length) + 2;
        const h = hours[key];
        const totalH = (h.ok + h.ng) / maxVal * plotH;
        const ngH = h.ng / maxVal * plotH;

        // OK portion
        ctx.fillStyle = "#22c55e";
        ctx.fillRect(x, H - pad.bottom - totalH, barW, totalH - ngH);

        // NG portion
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x, H - pad.bottom - ngH, barW, ngH);

        // Hour label
        const date = new Date(key * 1000);
        ctx.fillStyle = "#8b8fa3";
        ctx.font = "10px sans-serif";
        ctx.save();
        ctx.translate(x + barW / 2, H - pad.bottom + 12);
        ctx.fillText(date.getHours() + ":00", 0, 0);
        ctx.restore();
    });

    // Y-axis label
    ctx.fillStyle = "#8b8fa3";
    ctx.font = "10px sans-serif";
    ctx.fillText(maxVal.toString(), pad.left - 30, pad.top + 10);
    ctx.fillText("0", pad.left - 15, H - pad.bottom);
}

/* ── Gallery ─────────────────────────────────────────────────────────── */

async function loadGallery() {
    const verdict = document.getElementById("gallery-filter").value;
    const url = `${API_BASE}/api/inspections?limit=100${verdict ? "&verdict=" + verdict : ""}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        const grid = document.getElementById("gallery-grid");
        if (!data.length) {
            grid.innerHTML = '<p class="placeholder-text">No inspections found</p>';
            return;
        }

        grid.innerHTML = data.map(item => {
            const v = item.verdict;
            const imgUrl = item.image_path ? `/uploads/${item.image_path.split("/").pop()}` : "";
            const time = new Date(item.timestamp * 1000).toLocaleString();
            return `<div class="gallery-item" onclick='showModalFromInspection(${JSON.stringify(item).replace(/'/g, "\\'")})'>
                ${imgUrl ? `<img src="${imgUrl}" loading="lazy">` : '<div style="aspect-ratio:4/3;background:#111;"></div>'}
                <div class="gallery-meta">
                    <span>${time}</span>
                    <span class="gallery-verdict ${v.toLowerCase()}">${v}</span>
                </div>
            </div>`;
        }).join("");
    } catch (e) {
        console.error("Failed to load gallery:", e);
    }
}

/* ── Modal ────────────────────────────────────────────────────────────── */

function showModal(data) {
    const modal = document.getElementById("modal");
    const img = document.getElementById("modal-image");
    const info = document.getElementById("modal-info");

    if (data.image_url) img.src = data.image_url;
    else if (data.image_path) img.src = `/uploads/${data.image_path.split("/").pop()}`;

    // Draw detections on modal canvas after image loads
    img.onload = () => {
        drawDetections("modal-canvas", "modal-image", data.detections || []);
    };

    const time = data.timestamp ? new Date(data.timestamp * 1000).toLocaleString() : "--";
    info.innerHTML = `<table>
        <tr><td>Inspection ID</td><td>${(data.inspection_id || "").slice(0, 12)}</td></tr>
        <tr><td>Timestamp</td><td>${time}</td></tr>
        <tr><td>Verdict</td><td style="font-weight:700;color:${data.verdict === "NG" ? "var(--ng)" : "var(--ok)"}">${data.verdict}</td></tr>
        <tr><td>Defects</td><td>${data.defect_count || 0}</td></tr>
        <tr><td>Total Detections</td><td>${data.total_detections || 0}</td></tr>
        <tr><td>Pipeline</td><td>${(data.pipeline_ms || 0).toFixed(1)} ms</td></tr>
    </table>
    ${(data.detections || []).length ? "<h4 style='margin:12px 0 6px;font-size:13px;color:var(--text-muted)'>Detections</h4>" + (data.detections || []).map(d =>
        `<div class="detection-item ${(d.roi_verdict || "ng").toLowerCase()}" style="margin-bottom:4px">
            <span>${d.class_name} (${(d.confidence * 100).toFixed(1)}%)</span>
            <span>${d.roi_verdict || "NG"} ${d.roi_confidence ? (d.roi_confidence * 100).toFixed(0) + "%" : ""}</span>
        </div>`
    ).join("") : ""}`;

    modal.classList.add("active");
}

function showModalFromInspection(item) {
    showModal({
        ...item,
        image_url: item.image_path ? `/uploads/${item.image_path.split("/").pop()}` : null,
    });
}

function closeModal(event) {
    if (event.target === document.getElementById("modal")) {
        document.getElementById("modal").classList.remove("active");
    }
}

/* ── Camera controls ─────────────────────────────────────────────────── */

async function controlCamera(action) {
    try {
        const res = await fetch(`${API_BASE}/api/camera/${action}`, { method: "POST" });
        const data = await res.json();
        console.log(`Camera ${action}:`, data.status);
    } catch (e) {
        console.error(`Camera ${action} failed:`, e);
    }
}

/* ── File upload ─────────────────────────────────────────────────────── */

document.getElementById("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/api/inspect`, { method: "POST", body: form });
        const data = await res.json();
        handleInspectionResult(data);
    } catch (err) {
        console.error("Upload failed:", err);
    }

    e.target.value = "";
});

/* ── Utilities ───────────────────────────────────────────────────────── */

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

/* ── Init ────────────────────────────────────────────────────────────── */

connectWebSocket();
