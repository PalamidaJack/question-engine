"""A2A Peer Registry endpoints extracted from app.py."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/a2a", tags=["A2A"])




@router.get("/peers")
async def list_peers(request: Request):
    """List all registered A2A peers."""
    if request.app.state.peer_registry is None:
        return {"total_peers": 0, "healthy_peers": 0, "peers": []}
    return request.app.state.peer_registry.status()


@router.post("/peers")
async def register_peer(request: Request):
    """Register a new peer agent by URL (auto-discovers capabilities)."""
    if request.app.state.peer_registry is None:
        return JSONResponse(
            {"error": "Peer registry not initialized"}, status_code=503,
        )
    body = await request.json()
    url = body.get("url", "")
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    peer = await request.app.state.peer_registry.discover_and_register(url)
    if peer is None:
        return JSONResponse(
            {"error": f"Failed to discover agent at {url}"},
            status_code=502,
        )
    return peer.model_dump()


@router.delete("/peers/{peer_id}")
async def remove_peer(request: Request, peer_id: str):
    """Remove a registered peer."""
    if request.app.state.peer_registry is None:
        return JSONResponse(
            {"error": "Peer registry not initialized"}, status_code=503,
        )
    removed = request.app.state.peer_registry.unregister(peer_id)
    if not removed:
        return JSONResponse(
            {"error": f"Peer not found: {peer_id}"}, status_code=404,
        )
    return {"removed": peer_id}


@router.get("/peers/{peer_id}/health")
async def check_peer_health(request: Request, peer_id: str):
    """Check health of a registered peer."""
    if request.app.state.peer_registry is None:
        return JSONResponse(
            {"error": "Peer registry not initialized"}, status_code=503,
        )
    peer = request.app.state.peer_registry.get(peer_id)
    if peer is None:
        return JSONResponse(
            {"error": f"Peer not found: {peer_id}"}, status_code=404,
        )
    healthy = await request.app.state.peer_registry.check_health(peer_id)
    return {"peer_id": peer_id, "healthy": healthy, "url": peer.url}


# ── Interactive API Playground ──────────────────────────────────────


_PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QE API Playground</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
    .header { background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2a2d37;
              display: flex; align-items: center; gap: 16px; }
    .header h1 { margin: 0; font-size: 20px; color: #7c8aff; }
    .header .badge { background: #2a3a2a; color: #4ade80; padding: 2px 8px;
                     border-radius: 4px; font-size: 12px; }
    .container { display: flex; height: calc(100vh - 57px); }
    .sidebar { width: 280px; background: #14161e; border-right: 1px solid #2a2d37;
               overflow-y: auto; padding: 12px; }
    .main { flex: 1; padding: 24px; overflow-y: auto; }
    .group-title { color: #7c8aff; font-size: 13px; font-weight: 600;
                   text-transform: uppercase; margin: 16px 0 8px; letter-spacing: 0.5px; }
    .endpoint { padding: 6px 10px; border-radius: 6px; cursor: pointer;
                font-size: 13px; margin: 2px 0; display: flex; align-items: center; gap: 8px; }
    .endpoint:hover { background: #1e2130; }
    .endpoint.active { background: #1e2844; }
    .method { font-size: 11px; font-weight: 700; padding: 1px 6px; border-radius: 3px;
              min-width: 36px; text-align: center; }
    .method.get { background: #1a3a2a; color: #4ade80; }
    .method.post { background: #3a2a1a; color: #fbbf24; }
    .method.delete { background: #3a1a1a; color: #f87171; }
    .method.put { background: #1a2a3a; color: #60a5fa; }
    .method.ws { background: #2a1a3a; color: #c084fc; }
    .path { color: #a0a0b0; font-family: monospace; font-size: 12px; }
    .panel { background: #1a1d27; border-radius: 8px; border: 1px solid #2a2d37;
             padding: 20px; margin-bottom: 16px; }
    .panel h2 { margin: 0 0 12px; font-size: 16px; color: #e0e0e0; }
    .input-group { margin-bottom: 12px; }
    .input-group label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; }
    .input-group input, .input-group textarea, .input-group select {
      width: 100%; padding: 8px 12px; background: #0f1117; border: 1px solid #2a2d37;
      border-radius: 6px; color: #e0e0e0; font-family: monospace; font-size: 13px;
      box-sizing: border-box; }
    .input-group textarea { min-height: 100px; resize: vertical; }
    .btn { padding: 8px 20px; border-radius: 6px; border: none; cursor: pointer;
           font-weight: 600; font-size: 13px; }
    .btn-primary { background: #7c8aff; color: #fff; }
    .btn-primary:hover { background: #6b79ee; }
    .response-box { background: #0a0c14; border: 1px solid #2a2d37; border-radius: 6px;
                    padding: 16px; font-family: monospace; font-size: 13px;
                    white-space: pre-wrap; overflow-x: auto; min-height: 60px;
                    max-height: 500px; overflow-y: auto; }
    .status { display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 12px; font-weight: 600; margin-right: 8px; }
    .status.s2xx { background: #1a3a2a; color: #4ade80; }
    .status.s4xx { background: #3a2a1a; color: #fbbf24; }
    .status.s5xx { background: #3a1a1a; color: #f87171; }
    .timing { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>QE API Playground</h1>
    <span class="badge">v0.1.0</span>
    <span style="color:#666;font-size:13px">Interactive API explorer</span>
    <a href="/docs" style="margin-left:auto;color:#7c8aff;font-size:13px">
      OpenAPI Docs
    </a>
  </div>
  <div class="container">
    <div class="sidebar" id="sidebar"></div>
    <div class="main" id="main">
      <div class="panel">
        <h2>Welcome to the QE API Playground</h2>
        <p style="color:#888;font-size:14px">
          Select an endpoint from the sidebar to get started.
          You can send requests and see live responses.
        </p>
      </div>
    </div>
  </div>
<script>
const ENDPOINTS = [
  {group:"Health",endpoints:[
    {m:"GET",p:"/api/health",d:"Basic health check"},
    {m:"GET",p:"/api/health/ready",d:"Readiness probe"},
    {m:"GET",p:"/api/health/live",d:"Live health from Doctor"},
    {m:"GET",p:"/api/status",d:"Full engine status"},
  ]},
  {group:"Goals",endpoints:[
    {m:"POST",p:"/api/goals",d:"Submit a new goal",body:'{"description":""}'},
    {m:"GET",p:"/api/goals",d:"List all goals"},
    {m:"GET",p:"/api/goals/{goal_id}",d:"Get goal details",params:["goal_id"]},
    {m:"GET",p:"/api/goals/{goal_id}/progress",d:"Goal progress DAG",params:["goal_id"]},
    {m:"GET",p:"/api/goals/{goal_id}/result",d:"Goal result",params:["goal_id"]},
  ]},
  {group:"Memory",endpoints:[
    {m:"POST",p:"/api/memory/search",d:"Cross-tier memory search",
     body:'{"query":"","tier":"all","top_k":10}'},
    {m:"GET",p:"/api/memory/tiers",d:"Memory tier status"},
    {m:"GET",p:"/api/memory/procedural",d:"Procedural memory templates"},
    {m:"GET",p:"/api/memory/export",d:"Export all memory"},
  ]},
  {group:"Chat",endpoints:[
    {m:"POST",p:"/api/chat",d:"Send chat message",
     body:'{"message":"","session_id":null}'},
    {m:"GET",p:"/api/chat/stream",d:"SSE chat stream",params:["message"]},
  ]},
  {group:"A2A",endpoints:[
    {m:"GET",p:"/.well-known/agent.json",d:"Agent discovery card"},
    {m:"GET",p:"/api/a2a/peers",d:"List registered peers"},
    {m:"POST",p:"/api/a2a/peers",d:"Register a peer",body:'{"url":""}'},
    {m:"POST",p:"/api/a2a/tasks",d:"Create A2A task",
     body:'{"description":""}'},
  ]},
  {group:"Guardrails",endpoints:[
    {m:"GET",p:"/api/guardrails/status",d:"Guardrails status"},
    {m:"POST",p:"/api/guardrails/configure",d:"Update guardrails",
     body:'{"content_filter_enabled":true}'},
  ]},
  {group:"Claims",endpoints:[
    {m:"GET",p:"/api/claims",d:"List claims"},
  ]},
  {group:"Features",endpoints:[
    {m:"GET",p:"/api/flags",d:"List feature flags"},
    {m:"POST",p:"/api/flags/{name}/enable",d:"Enable flag",params:["name"]},
    {m:"POST",p:"/api/flags/{name}/disable",d:"Disable flag",params:["name"]},
  ]},
  {group:"Monitoring",endpoints:[
    {m:"GET",p:"/api/metrics",d:"Engine metrics"},
    {m:"GET",p:"/api/bus/stats",d:"Bus statistics"},
    {m:"GET",p:"/api/arena/status",d:"Arena rankings"},
    {m:"GET",p:"/api/knowledge/status",d:"Knowledge loop status"},
    {m:"GET",p:"/api/bridge/status",d:"Inquiry bridge status"},
    {m:"GET",p:"/api/scout/status",d:"Scout status"},
  ]},
];

const sidebar = document.getElementById("sidebar");
const main = document.getElementById("main");

ENDPOINTS.forEach(g => {
  const title = document.createElement("div");
  title.className = "group-title";
  title.textContent = g.group;
  sidebar.appendChild(title);
  g.endpoints.forEach(ep => {
    const div = document.createElement("div");
    div.className = "endpoint";
    div.innerHTML = `<span class="method ${ep.m.toLowerCase()}">${ep.m}</span>`
      + `<span class="path">${ep.p}</span>`;
    div.onclick = () => showEndpoint(ep, div);
    sidebar.appendChild(div);
  });
});

function showEndpoint(ep, el) {
  document.querySelectorAll(".endpoint").forEach(e => e.classList.remove("active"));
  el.classList.add("active");
  let paramsHtml = "";
  if (ep.params) {
    paramsHtml = ep.params.map(p =>
      `<div class="input-group"><label>${p}</label>`
      + `<input id="param-${p}" placeholder="${p}" /></div>`
    ).join("");
  }
  let bodyHtml = "";
  if (ep.body) {
    bodyHtml = `<div class="input-group"><label>Request Body (JSON)</label>`
      + `<textarea id="req-body">${ep.body}</textarea></div>`;
  }
  main.innerHTML = `
    <div class="panel">
      <h2><span class="method ${ep.m.toLowerCase()}">${ep.m}</span> ${ep.p}</h2>
      <p style="color:#888;font-size:13px">${ep.d}</p>
      ${paramsHtml}${bodyHtml}
      <button class="btn btn-primary" onclick="sendRequest()">Send Request</button>
    </div>
    <div class="panel" id="response-panel" style="display:none">
      <h2>Response <span id="resp-status"></span><span class="timing" id="resp-time"></span></h2>
      <div class="response-box" id="resp-body"></div>
    </div>`;
  main._ep = ep;
}

async function sendRequest() {
  const ep = main._ep;
  if (!ep) return;
  let path = ep.p;
  if (ep.params) {
    ep.params.forEach(p => {
      const v = document.getElementById("param-"+p)?.value || "";
      path = path.replace("{"+p+"}", encodeURIComponent(v));
    });
  }
  const opts = {method: ep.m, headers: {"Content-Type":"application/json"}};
  if (ep.body) {
    const bodyEl = document.getElementById("req-body");
    if (bodyEl) opts.body = bodyEl.value;
  }
  const panel = document.getElementById("response-panel");
  panel.style.display = "block";
  document.getElementById("resp-body").textContent = "Loading...";
  const t0 = performance.now();
  try {
    const resp = await fetch(path, opts);
    const elapsed = Math.round(performance.now() - t0);
    const statusEl = document.getElementById("resp-status");
    const sc = resp.status < 300 ? "s2xx" : resp.status < 500 ? "s4xx" : "s5xx";
    statusEl.innerHTML = `<span class="status ${sc}">${resp.status}</span>`;
    document.getElementById("resp-time").textContent = `${elapsed}ms`;
    const text = await resp.text();
    try { // noqa: E501
      const j = JSON.stringify(JSON.parse(text),null,2);
      document.getElementById("resp-body").textContent = j;
    } catch { document.getElementById("resp-body").textContent = text; }
  } catch(e) {
    document.getElementById("resp-body").textContent = "Error: " + e.message;
  }
}
</script>
</body>
</html>"""


@router.get("/playground", include_in_schema=False)
async def playground(request: Request):
    """Interactive API playground for exploring QE endpoints."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_PLAYGROUND_HTML)
