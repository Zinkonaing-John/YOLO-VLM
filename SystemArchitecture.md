# System Architecture — VLM-YOLO Industrial AI Vision Inspector

## Overview

A full-stack industrial defect detection platform combining multiple AI models (YOLO + CLIP + optional SimpleNet/VLM) with real-time WebSocket streaming, a FastAPI backend, and a Next.js frontend.

---

## High-Level Data Flow

```
┌─────────────┐    POST /inspect     ┌──────────────────────────────────┐
│  Frontend    │ ──────────────────►  │  FastAPI Backend                 │
│  (Next.js)   │                      │                                  │
│              │ ◄── WebSocket ─────  │  1. Save image to disk           │
│  - Upload    │     broadcast        │  2. CLIP full-image classify     │
│  - LiveFeed  │                      │  3. YOLO detect bounding boxes   │
│  - Dashboard │ ◄── GET /statistics  │  4. CLIP per-ROI classification  │
│  - Gallery   │ ◄── GET /inspections │  5. Determine verdict (OK/NG)    │
│  - Reports   │                      │  6. Persist to PostgreSQL        │
└─────────────┘                      │  7. Broadcast via WebSocket      │
                                     └──────────┬───────────────────────┘
                                                │
                                     ┌──────────▼───────────┐
                                     │  PostgreSQL           │
                                     │  - inspections table  │
                                     │  - defects table      │
                                     └───────────────────────┘
```

---

## AI/ML Pipeline (Multi-Model Ensemble)

| Stage | Model | Purpose |
|-------|-------|---------|
| 1. Global Classification | CLIP (ViT-B/32) | Zero-shot OK/NG verdict on full image using semantic labels |
| 2. Object Detection | YOLOv8 | Localize defect regions with bounding boxes |
| 3. ROI Classification | CLIP (per-crop) | Classify each detected ROI as defective or normal |
| 4. Anomaly Detection | SimpleNet | ResNet-18 backbone + discriminator for pixel-level anomaly heatmaps (training-only, not yet in pipeline) |
| 5. Explanation | VLM (LLaVA via Ollama) | Optional natural-language defect explanations |

**Verdict logic:** NG if CLIP full-image NG score ≥ 0.5 threshold, otherwise OK.

---

## AI Pipeline Architecture

### Inference Flow

```
Input Image
    │
    ├──────────────────────────┐
    ▼                          ▼
┌────────────────┐   ┌─────────────────────┐
│  CLIP (ViT-B/32)│   │  YOLOv8m Detection   │
│  Full-Image     │   │  Object Localization │
│  Classification  │   │  conf=0.25, iou=0.45│
└───────┬────────┘   └─────────┬───────────┘
        │                      │
        │               ┌──────▼──────┐
        │               │ Crop each   │
        │               │ bounding box│
        │               └──────┬──────┘
        │                      │
        │               ┌──────▼──────────┐
        │               │ CLIP per-ROI    │
        │               │ Classification  │
        │               └──────┬──────────┘
        │                      │
        ▼                      ▼
┌──────────────────────────────────────┐
│         Verdict Decision             │
│  NG if CLIP full-image NG ≥ 0.5     │
│  Each ROI tagged defect/normal       │
└──────────────────┬───────────────────┘
                   │
          ┌────────▼────────┐
          │  Optional: VLM  │
          │  (LLaVA/Ollama) │
          │  Natural language│
          │  explanation     │
          └─────────────────┘
```

### Stage 1 — CLIP Full-Image Classification

**Model:** OpenAI CLIP ViT-B/32 (zero-shot, no fine-tuning)

**How it works:** Compares the image embedding against text embeddings of semantic labels, then softmax over all similarities.

| OK Labels | NG Labels |
|-----------|-----------|
| "smooth clean surface" | "scratched surface" |
| "flawless product" | "dirty stained surface" |
| | "cracked damaged surface" |

**Decision:** Aggregates NG label probabilities. If NG total ≥ **0.5** threshold → image is defective. This determines the **overall verdict**.

### Stage 2 — YOLO Object Detection

**Model:** Ultralytics YOLOv8 with custom-trained weights (`weights/best.pt`)

**Purpose:** Localize defect regions — outputs bounding boxes with class labels and confidence scores.

**Config:** confidence threshold 0.25, IoU threshold 0.45

**Output per detection:**
- `defect_class` — label from trained class map
- `confidence` — detection score
- `bbox_x1, y1, x2, y2` — normalized coordinates

### Stage 3 — CLIP Per-ROI Classification

For **each** YOLO bounding box:

1. Crop the region from the original image
2. Run CLIP classification on the cropped patch
3. Assign `clip_label`, `clip_score`, and `is_defect` flag

This enriches every detection with semantic meaning — YOLO says *where*, CLIP says *what kind*.

### Stage 4 — SimpleNet Anomaly Detection *(training-only, not yet in pipeline)*

**Architecture:**

```
Input Image
    ▼
┌──────────────────┐
│ ResNet-18 Backbone│  ← frozen (pretrained ImageNet features)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Feature Projector │  ← trainable (Linear + BatchNorm + LeakyReLU)
└────────┬─────────┘
         ├────────────────────┐
         ▼                    ▼
┌────────────────┐  ┌─────────────────┐
│ Normal Features│  │ Anomaly Generator│  ← produces pseudo-anomalies
└────────┬───────┘  └────────┬────────┘
         │                   │
         ▼                   ▼
┌────────────────────────────────────┐
│ Binary Discriminator                │  ← normal=0, anomaly=1
│ (BCE loss, cosine annealing LR)    │
└────────────────────────────────────┘
         ▼
   Anomaly Heatmap + Score
```

**Training:** Uses only normal images (`data/train/good/`). The generator learns to create realistic pseudo-anomalies, the discriminator learns to distinguish them. Evaluated via AUROC on labeled test set.

### Stage 5 — VLM Explanation *(optional)*

**Model:** LLaVA via local Ollama server

**Capabilities:**
- `explain_defect()` — natural-language description of a detected defect
- `detect_defects()` — alternative PASS/FAIL classification with reasoning
- `ask()` — free-form visual Q&A about the image

Triggered by the user toggling "VLM" in the upload UI. Sends base64-encoded image + prompt to Ollama HTTP API.

### AI Pipeline Summary

| Stage | Model | Input | Output | Status |
|-------|-------|-------|--------|--------|
| Global classify | CLIP ViT-B/32 | Full image | OK/NG verdict + score | **Active** |
| Detect regions | YOLOv8 | Full image | Bounding boxes + classes | **Active** |
| ROI classify | CLIP ViT-B/32 | Cropped ROIs | Per-defect label + score | **Active** |
| Anomaly detect | SimpleNet | Full image | Pixel heatmap + score | **Training only** |
| Explain | LLaVA (Ollama) | Image + prompt | Natural language text | **Optional** |

---

## Backend (FastAPI)

```
backend/app/
├── main.py                 # App init, lifespan, WebSocket manager
├── core/
│   ├── config.py           # Pydantic settings (DB, YOLO, CLIP, VLM, auth)
│   └── database.py         # Async SQLAlchemy (asyncpg, pool=20)
├── models/
│   ├── db_models.py        # ORM: Inspection + Defect tables
│   ├── ai_models.py        # YOLODetector + CLIPClassifier wrappers
│   └── simplenet.py        # SimpleNet anomaly detection model
├── services/
│   ├── inspection_service.py  # Core pipeline orchestration
│   └── vlm_service.py        # Optional Ollama/LLaVA integration
└── routers/
    ├── inspection.py       # POST /inspect, GET /inspections, DELETE
    ├── statistics.py       # GET /statistics, GET /statistics/daily
    └── auth.py             # API key verification
```

**Key endpoints:**
- `POST /inspect` — upload image, run full pipeline, return verdict + defects
- `GET /inspections` — paginated history with verdict/class filtering
- `GET /statistics` — aggregates (OK/NG rates, defect distribution, avg latency)
- `GET /statistics/daily` — daily trend data for charts
- `WS /ws/inspection` — real-time result broadcasting

---

## Frontend (Next.js + TypeScript + Tailwind)

```
frontend/src/
├── app/
│   ├── page.tsx            # Home: stats, camera feed, upload, live feed
│   ├── dashboard/page.tsx  # Charts: daily bar chart + defect donut
│   ├── gallery/page.tsx    # Card grid with filters + detail modal
│   └── reports/page.tsx    # Table view with sorting + pagination
├── components/
│   ├── HeaderNav.tsx       # Navigation + connection status
│   ├── InspectUpload.tsx   # Drag-drop upload + VLM toggle
│   ├── DefectCard.tsx      # Result card with bbox overlays
│   ├── LiveFeed.tsx        # Real-time WebSocket result stream
│   ├── CameraFeed.tsx      # WebRTC live camera input
│   ├── AnomalyHeatmap.tsx  # SimpleNet heatmap visualization
│   ├── StatsChart.tsx      # Bar + donut chart components
│   └── Modal.tsx           # Reusable modal dialog
└── hooks/
    └── useInspection.ts    # WebSocket hook with auto-reconnect
```

---

## Database Schema

Two tables with UUID primary keys and cascade deletes:

- **inspections** — id, tenant_id, timestamp, image_path, verdict (OK/NG), total_defects, processing_ms
- **defects** — id, inspection_id (FK), defect_class, confidence, bbox coordinates, clip_label, clip_score, is_defect

---

## Infrastructure

| Service | Purpose |
|---------|---------|
| PostgreSQL | Primary database |
| Redis | Caching (configured, lightly used) |
| Ollama | Optional local VLM server (LLaVA) |
| Docker Compose | Orchestrates all services |

---

## Key Design Decisions

1. **Async-first** — all DB/IO operations use async/await
2. **Zero-shot CLIP** — no fine-tuning needed; classifies via semantic text labels
3. **Graceful degradation** — works without YOLO weights, CLIP, or VLM
4. **Real-time streaming** — WebSocket broadcasts results to all connected clients
5. **Multi-tenant ready** — optional `tenant_id` on inspections
