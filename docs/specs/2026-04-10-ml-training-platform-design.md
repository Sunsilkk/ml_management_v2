# ML Training Platform — Design Spec

**Date:** 2026-04-10
**Status:** Draft (Revision 2 — post Codex review)
**Scope:** Phase 1 — Training pipeline + Dashboard (Anomaly Detection / SALAD)

---

## 1. Overview

Hệ thống quản lý training pipeline và inference tự động cho ML models. User upload dataset, chọn task/pipeline, config hyperparameters, hệ thống tự chạy training trên GPU server và trả kết quả.

**Phase 1 focus:** Anomaly Detection sử dụng SALAD pipeline.

### Goals

- Upload dataset qua web UI, tự validate format
- Dataset versioning (split config → tạo version mới)
- Config training pipeline qua UI (mỗi step có form riêng)
- Tự động chạy training trên remote GPU server qua Docker containers
- Live logs, real-time metrics, charts
- Model registry — lưu trữ, so sánh, export models
- Worker management — khai báo GPU workers, auto job scheduling

### Target Users

- Ban đầu: team nội bộ
- Sau: mở rộng cho khách hàng (multi-tenant)

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python 3.11) |
| Frontend | Next.js + shadcn/ui |
| Database | PostgreSQL |
| Job Queue | Celery + Redis |
| Object Storage | MinIO |
| Training Env | Docker containers (nvidia/cuda) |
| Package Manager | uv |

---

## 2. Architecture

### Distributed System — 2 machines

```
┌─── App Server (MacBook / Cloud) ────────────────────────────────┐
│                                                                  │
│  Next.js UI ──▶ FastAPI ──▶ PostgreSQL                           │
│                    │           Redis (Celery broker)              │
│                    │           MinIO (object storage)             │
│                    │                                             │
└────────────────────┼─────────────────────────────────────────────┘
                     │  Celery task dispatch (qua Redis network)
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│  GPU Server                                                       │
│                                                                   │
│  Worker Daemon (nhận config từ API, spawn/manage Celery workers)  │
│  ├── Worker A (GPU 2) — 1 job tại 1 thời điểm                   │
│  ├── Worker B (GPU 3) — 1 job tại 1 thời điểm                   │
│  └── ...                                                          │
│                                                                   │
│  Mỗi training job chạy trong Docker container riêng:              │
│  - Pull pretrained models từ MinIO                                │
│  - Pull dataset từ MinIO                                          │
│  - Chạy pipeline                                                  │
│  - Upload results lên MinIO                                       │
│  - Cleanup (GPU server luôn stateless)                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Modular Monolith

FastAPI app tổ chức theo modules, mỗi module có domain riêng:

```
FastAPI App
├── modules/
│   ├── auth/           # Authentication + authorization
│   ├── datasets/       # Upload, versioning, split config
│   ├── pipelines/      # Pipeline registry, config schemas
│   ├── training/       # Job management, orchestration, live logs
│   ├── model_registry/ # Model storage, export, comparison
│   └── admin/          # Worker management, pretrained models
├── workers/
│   ├── celery_app.py   # Celery configuration
│   ├── tasks.py        # Task definitions
│   └── runner.py       # Docker container orchestration
└── core/               # DB, storage, auth, events
```

---

## 3. GPU Worker System

### Worker Registration & Management

Admin khai báo workers qua dashboard — hệ thống tự quản lý:

1. Admin vào Dashboard → khai báo "worker-a, GPU 2" và "worker-b, GPU 3"
2. API gửi config xuống Worker Daemon trên GPU server
3. Daemon tự spawn Celery workers với đúng `CUDA_VISIBLE_DEVICES`
4. Admin không cần SSH hay chạy lệnh tay

### Job Scheduling

**Dispatch Contract:**

- **Queue routing:** Mỗi worker có dedicated queue (`celery -Q worker-a`). Job dispatch chỉ định queue cụ thể, không dùng shared queue.
- **Atomic job lease:** Trước khi dispatch, API set `training_jobs.status = 'queued'` + `worker_name = 'worker-a'` trong 1 transaction. Nếu worker đã busy → reject, chọn worker khác.
- **Idempotent start:** Mỗi job có `idempotency_token` (UUID). Worker kiểm tra token trước khi bắt đầu — nếu đã có container chạy cùng token → skip (tránh duplicate từ Celery redelivery).
- **Late acknowledgement:** Task sử dụng `acks_late=True` + `reject_on_worker_lost=True`. Celery chỉ ack khi job hoàn tất/fail, không ack khi nhận.
- **Heartbeat:** Worker gửi heartbeat mỗi 30s (`workers.last_heartbeat`). API coi worker offline nếu miss 3 heartbeats liên tiếp (90s). Job trên offline worker sẽ được re-queue.
- **Visibility timeout:** Celery `visibility_timeout = 86400` (24h, vì training jobs chạy lâu). Ngắn hơn sẽ gây phantom redelivery.
- Mỗi worker có `concurrency=1` (1 job tại 1 thời điểm)
- Hết worker → job nằm trong queue đợi
- Worker chỉ thấy GPU được assign → không xâm phạm GPU của service khác

### Per-Job Docker Container Lifecycle

```
QUEUED → PULLING_DATA → PREPROCESSING → TRAINING → UPLOADING → CLEANUP → DONE

Container flow:
1. Spawn container với image pipeline cụ thể (vd: salad-training:latest)
2. Mount /workspace (tmpdir cho job)
3. Pull pretrained models từ MinIO → /workspace/models/
4. Pull dataset từ MinIO → /workspace/data/
5. Chạy pipeline steps tuần tự
6. Upload models + metrics + logs → MinIO
7. Report kết quả về API
8. Xóa container + /workspace/ (sạch sẽ)
```

### Cleanup Policy

| Trường hợp | Hành động |
|------------|-----------|
| Job thành công | Upload final models + metrics → MinIO, xóa container + data |
| Job thất bại | Upload error logs + partial artifacts → MinIO, xóa container + data |
| Job timeout | Kill container, upload logs, xóa sạch |
| Job cancelled | SIGTERM → container graceful stop (30s), fallback SIGKILL, upload logs |

GPU server luôn sạch sau mỗi job — không tích tụ data rác.

### Failure, Retry, Resume & Cancellation

**Per-step tracking:** Mỗi step có record riêng trong `job_steps` table (xem Schema bên dưới) với status, timing, retry count, artifact paths.

**Retry policy:**
- Mỗi step tối đa `max_retries = 2` (configurable per pipeline)
- Transient failures (OOM, container crash) → auto-retry cùng step, không reset steps trước
- Persistent failures (3 fails liên tiếp) → job status = 'failed', partial artifacts giữ nguyên trên MinIO
- Mỗi retry attempt ghi vào `job_steps.attempts` (JSONB array)

**Resume semantics:**
- API endpoint `POST /api/v1/training/jobs/{id}/resume` cho phép resume từ step cuối thành công
- Resume tạo new attempt cho failed step, giữ nguyên output các steps trước
- Chỉ resume được jobs ở status 'failed' (không resume 'cancelled')

**Cancellation:**
- `POST /api/v1/training/jobs/{id}/cancel` → API set status = 'cancelling'
- Worker nhận signal → SIGTERM container → graceful shutdown (30s timeout)
- Container trap SIGTERM → save checkpoint nếu có thể → exit
- Sau cancel: upload partial logs/artifacts, status = 'cancelled'
- **Cancel propagation:** API gửi Celery revoke + kill signal. Worker daemon verify container stopped.

**Artifact validity:**
- Mỗi step output có `status.json` trong MinIO: `{status: "completed"|"partial"|"failed", ...}`
- Steps sau chỉ chạy nếu step trước có `status: "completed"`
- Partial artifacts được tag — không dùng làm input cho steps sau

---

## 4. SALAD Pipeline Detail

### Pipeline Steps (tuần tự)

#### Step 1: Foreground Mask Generation (SAM)

- **Script:** `create_fg_masks.py`
- **Input:** Raw images từ dataset
- **Output:** Binary foreground masks
- **Prerequisites:** `pretrained-models/salad/sam_vit_h_4b8939.pth` (~2.5GB)
- **Configurable:** Không (auto)

#### Step 2: Pseudo Label Generation (DINO + SAM-HQ)

- **Script:** `create_pseudo_labels.py`
- **Input:** Raw images + FG masks từ Step 1
- **Output:** Noisy composition maps (6-class one-hot)
- **Prerequisites:**
  - `pretrained-models/salad/sam_hq_vit_h.pth` (~2.5GB)
  - DINO-ViT-B/8 — **phải mirror sẵn trong Docker image** (script dùng `torch.hub` để load, cần bake weights vào image hoặc cache trong MinIO tại `pretrained-models/salad/dino_vitbase8/`)

| Parameter | Default | UI Control | Notes |
|-----------|---------|------------|-------|
| n_clusters | 5 | Number input | Duy nhất param script expose qua CLI |

> **Note:** `iou_threshold` và `min_mask_area` hiện **hard-coded** trong script. Nếu cần expose → wrapper script sẽ patch values trước khi gọi. Phase 1: giữ defaults.

#### Step 3: Composition Segmentation Training

- **Script:** `train_composition_segmentation_model.py`
- **Input:** Raw images + noisy composition maps từ Step 2
- **Output:** Clean composition maps cho train/validation/test

| Parameter | Default | UI Control | Notes |
|-----------|---------|------------|-------|
| epochs | 10 | Number input | Duy nhất param script expose qua CLI |

> **Note:** `learning_rate` (1e-4) và `batch_size` (8) hiện **hard-coded** trong script. Phase 1: giữ defaults. Wrapper có thể patch nếu cần expose sau.

#### Step 4: SALAD Main Training

- **Script:** `train_salad.py`
- **Input:** Raw images + clean composition maps từ Step 3
- **Output:** 5 trained models (.pth)
- **Prerequisites:** `pretrained-models/salad/teacher_medium.pth` (~32MB)

**Training Parameters:**

| Category | Parameter | Default | UI Control | Notes |
|----------|-----------|---------|------------|-------|
| Training | train_steps | 70000 | Number input | CLI: `--train_steps` |
| Training | seed | 42 | Number input | CLI: `--seed` |
| Data | imagenet_penalty | OFF | Toggle | CLI: `--imagenet_train_path` (set 'none' or MinIO path) |

> **Note:** Optimizer params (lr, weight_decay, scheduler, loss weights) đều **hard-coded** trong `train_salad.py`. Spec liệt kê chúng ở bảng dưới làm reference, KHÔNG hiện trên UI Phase 1. Wrapper script có thể inject/patch nếu cần expose sau.

**Hard-coded Reference (KHÔNG configurable qua UI):**

| Parameter | Hard-coded Value | Location |
|-----------|-----------------|----------|
| lr_appearance (Student + AE) | 1e-4 | train_salad.py L200-205 |
| lr_composition (Comp AE + UNet) | 1e-5 | train_salad.py L200-205 |
| weight_decay | 1e-5 | train_salad.py L210 |
| lr_decay_gamma | 0.1 | train_salad.py L215 |
| lr_decay_at_percent | 95% | train_salad.py L215 |
| focal_loss_gamma | 2 | train_salad.py L218 |
| image_size | 256 | train_salad.py L120 |
| batch_size | 1 | train_salad.py L145 |
| num_workers | 4 | train_salad.py L147 |
| checkpoint_interval | 10000 | train_salad.py L310 |

**ImageNet Penalty:** Optional. Khi ON → pull ImageNet data từ MinIO (`pretrained-models/imagenet/`). Mặc định OFF — đủ tốt cho custom datasets.

### Training Output (5 models)

| Model | Actual Filename | Vai trò |
|-------|----------------|---------|
| Teacher | teacher_final.pth | Frozen reference network |
| Student | student_final.pth | Appearance anomaly detection |
| AutoEncoder | autoencoder_final.pth | Appearance reconstruction |
| Comp AE | comp_autoencoder_final.pth | Composition reconstruction |
| Comp UNet | comp_unet_final.pth | Composition anomaly mask |

> **Important:** Filenames PHẢI match exactly với `train_salad.py` output. Spec dùng `comp_autoencoder_final.pth` (không phải `comp_ae_final.pth`).

### Evaluation Metrics (mỗi checkpoint)

- AUC Combined (0.5 * logical + 0.5 * structural)
- AUC Logical (logical anomalies vs good — all branches fused)
- AUC Structural (structural anomalies vs good — all branches fused)
- AUC Appearance (combined) + Appearance Logical + Appearance Structural
- AUC Mahalanobis (combined) + Mahalanobis Logical + Mahalanobis Structural
- AUC Composition (combined) + Composition Logical + Composition Structural

> Total 15 AUC metrics per checkpoint. Per-branch logical/structural tính từ per-image branch scores grouped by anomaly type.

---

## 5. MinIO Storage Structure

```
MinIO
├── pretrained-models/
│   ├── salad/
│   │   ├── teacher_medium.pth
│   │   ├── sam_vit_h_4b8939.pth
│   │   ├── sam_hq_vit_h.pth
│   │   └── dino_vitbase8/dino_vitb8.pth    ← mirror từ torch.hub
│   ├── imagenet/train/...              (optional, ~150GB)
│   └── {future-pipeline}/...
│
├── datasets/
│   └── {dataset_id}/
│       ├── meta.json
│       ├── raw/dataset.zip
│       └── versions/{version_id}/
│           ├── version.json
│           ├── {category}/                  ← Category root (SALAD expects this)
│           │   ├── train/good/...
│           │   ├── validation/
│           │   │   ├── good/...
│           │   │   ├── logical_anomalies/...   ← exact folder names
│           │   │   └── structural_anomalies/...
│           │   ├── test/
│           │   │   ├── good/...
│           │   │   ├── logical_anomalies/...
│           │   │   └── structural_anomalies/...
│           │   └── ground_truth/            ← Required for evaluation
│           │       ├── logical_anomalies/
│           │       │   └── {defect_id}/000.png   (binary masks)
│           │       └── structural_anomalies/
│           │           └── {defect_id}/000.png
│           └── composition_maps/            ← Generated by Steps 1-3
│               └── {category}/
│                   ├── train/good/...
│                   ├── validation/{good,logical,structural}/...
│                   └── test/{good,logical,structural}/...
│
├── training-jobs/
│   └── {job_id}/
│       ├── job.json
│       ├── pipeline-config.json
│       ├── steps/
│       │   ├── 01_fg_masks/{status.json, masks/...}
│       │   ├── 02_pseudo_labels/{status.json, composition_maps/...}
│       │   ├── 03_comp_seg_training/{status.json, composition_maps/...}
│       │   │   └── Note: Script outputs composition maps only, NOT a model artifact
│       │   └── 04_salad_training/
│       │       ├── status.json
│       │       ├── checkpoints/{iter_10000/,iter_20000/,...}
│       │       └── final/{teacher,student,autoencoder,comp_autoencoder,comp_unet}_final.pth
│       ├── logs/
│       │   ├── step_01/chunks/{offset}.jsonl   (immutable log chunks)
│       │   ├── step_01/index.json
│       │   ├── step_02/chunks/...
│       │   ├── step_03/chunks/...
│       │   └── step_04/chunks/...
│       └── metrics/
│           ├── loss.jsonl      (1 dòng = 1 log event, every 10 iters → ~7000 entries for 70k steps)
│           ├── eval.jsonl      (1 dòng = 1 checkpoint evaluation, every 10k iters → ~7 entries)
│           └── summary.json    (kết quả cuối cùng)
│
└── model-registry/
    └── {model_id}/
        ├── manifest.json       (see Inference Manifest Schema below)
        ├── weights/
        │   ├── teacher_final.pth
        │   ├── student_final.pth
        │   ├── autoencoder_final.pth
        │   ├── comp_autoencoder_final.pth   ← exact SALAD filename
        │   └── comp_unet_final.pth
        ├── calibration/            ← REQUIRED for inference (see schema below)
        │   └── calibration.pt      (single torch dict with all normalization values)
        └── export/{category}_salad_{version}.zip  (weights + calibration + manifest)
```

**Adapter Layer:** SALAD scripts không natively produce `loss.jsonl`, `eval.jsonl`, hay calibration bundles. Container wrapper sẽ **instrument SALAD code** (not just parse stdout):
1. **Loss metrics:** Inject callback vào training loop (every 10 iters, matching existing ClearML log cadence) → write structured `loss.jsonl` với 6 fields: `loss_total, loss_st, loss_ae, loss_stae, loss_comp_recon, loss_comp_mask`
2. **Eval metrics:** Inject callback vào eval loop (every 10k iters) → write `eval.jsonl` với 15 fields: `auc_combined, auc_logical, auc_structural, auc_appearance, auc_appearance_logical, auc_appearance_structural, auc_mahalanobis, auc_mahalanobis_logical, auc_mahalanobis_structural, auc_composition, auc_composition_logical, auc_composition_structural`. Per-branch logical/structural computed từ per-image branch scores + folder labels.
3. **Calibration:** After final training + test: collect all normalization stats → save `calibration.pt`
4. Copy model artifacts với exact filenames từ SALAD output

> **Instrumentation approach:** Wrapper imports SALAD modules, monkey-patches logging hooks at known iteration cadence. This is more reliable than stdout parsing (which is fragile and incomplete — e.g., logical/structural breakdown is only available in the return values, not printed to stdout).

> **Metrics cadence note:** SALAD chỉ log losses mỗi 10 iterations, không phải mỗi iteration. `loss.jsonl` sẽ có ~7,000 entries cho 70k steps (không phải 70k entries). Eval metrics logged mỗi 10k iterations → ~7 entries.

**Inference Manifest Schema (`manifest.json`):**
```json
{
  "version": "1.0",
  "pipeline": "salad",
  "category": "chip_pcb",
  "job_id": "uuid",
  "created_at": "2026-04-10T12:00:00Z",
  "weights": {
    "teacher": "weights/teacher_final.pth",
    "student": "weights/student_final.pth",
    "autoencoder": "weights/autoencoder_final.pth",
    "comp_autoencoder": "weights/comp_autoencoder_final.pth",
    "comp_unet": "weights/comp_unet_final.pth"
  },
  "calibration": "calibration/calibration.pt",
  "metrics": {
    "auc_combined": 0.9629,
    "auc_appearance": 0.95,
    "auc_mahalanobis": 0.91,
    "auc_composition": 0.98
  }
}
```

**Calibration Artifact Schema (`calibration.pt` — single torch.save dict):**

Schema derived directly from `test_salad.py` `test()` function signature + helper outputs.

```python
{
    # ── Teacher output normalization (from teacher_normalization()) ──
    "teacher_mean": torch.Tensor,              # shape [1, 384, 1, 1], channelwise mean
    "teacher_std": torch.Tensor,               # shape [1, 384, 1, 1], channelwise std

    # ── Appearance map quantiles (from map_normalization()) ──
    "q_st_start": torch.Tensor,                # scalar — 90th percentile of student anomaly maps
    "q_st_end": torch.Tensor,                  # scalar — 99.5th percentile
    "q_ae_start": torch.Tensor,                # scalar — 90th percentile of AE anomaly maps
    "q_ae_end": torch.Tensor,                  # scalar — 99.5th percentile

    # ── Score normalization (from score_normalization()) ──
    "q_eff_start": float,                      # mean of effective/combined scores on validation
    "q_eff_end": float,                        # std of effective/combined scores
    "q_seg_start": float,                      # mean of composition scores on validation
    "q_seg_end": float,                        # std of composition scores

    # ── Global Mahalanobis (from extract_features_mahalanobis()) ──
    "feature_vectors_mean": np.ndarray,        # shape [384], global teacher feature mean
    "feature_vectors_covinv": np.ndarray,      # shape [384, 384], LedoitWolf inverse covariance

    # ── Per-class segmentation Mahalanobis ──
    "feature_vectors_mean_seg": Dict[int, np.ndarray],      # {class_k: [384]} per composition class
    "feature_vectors_covinv_seg": Dict[int, np.ndarray],    # {class_k: [384, 384]}

    # ── Per-class area-weighted Mahalanobis ──
    "feature_vectors_mean_seg_area": Dict[int, np.ndarray],     # {class_k: [1]}
    "feature_vectors_covinv_seg_area": Dict[int, np.ndarray],   # {class_k: [1, 1]}

    # ── Mahalanobis score normalization (from map_normalization_mahalanobis()) ──
    "q_start_mah": Dict[Union[str, int], float],  # {"full": mean, 0: mean, 1: mean, ...K-1: mean}
    "q_end_mah": Dict[Union[str, int], float],    # {"full": std, 0: std, 1: std, ...K-1: std}

    # ── Metadata ──
    "image_size": int,                         # 256
    "num_composition_classes": int,            # typically 6
}
```

> **Type notes:**
> - `teacher_mean/std` are `torch.Tensor` (GPU → moved to CPU for serialization)
> - `q_st_*`, `q_ae_*` are `torch.Tensor` scalars
> - `q_eff_*`, `q_seg_*` are Python `float`
> - Mahalanobis arrays are `np.ndarray` (float64, from LedoitWolf + pinv)
> - Dict keys in `q_start_mah`/`q_end_mah` are mixed: `"full"` (str) + `0..K-1` (int)
> - All values serialized via `torch.save()` which handles mixed types via pickle

**Nguyên tắc:**

- `pretrained-models/` — Tổ chức theo pipeline. **Tất cả runtime dependencies (DINO weights, SAM, focal-loss) phải được mirror ở đây hoặc bake vào Docker image.** Không cho phép training containers fetch từ internet.
- `datasets/` — Tổ chức theo dataset_id → versions. **Category root + exact folder names** (`train/good`, `test/logical_anomalies`, `ground_truth/`) phải match SALAD filesystem contract.
- `training-jobs/` — Toàn bộ output 1 job: intermediate files, logs, metrics, checkpoints. Có thể cleanup cũ.
- `model-registry/` — **Full inference bundle**: weights + calibration artifacts + manifest. Model zip exportable và self-contained cho inference.

---

## 6. Database Schema

### users

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| email | VARCHAR UNIQUE | |
| name | VARCHAR | |
| role | ENUM('admin','user') | |
| hashed_password | VARCHAR | |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### datasets

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| name | VARCHAR | "chip_pcb" |
| category | VARCHAR | User tự đặt |
| description | TEXT | |
| owner_id | UUID FK → users | |
| minio_raw_path | VARCHAR | "datasets/{id}/raw/" |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### dataset_versions

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| dataset_id | UUID FK → datasets | |
| version_number | INT | 1, 2, 3... |
| split_config | JSONB | {"train":80,"val":10,"test":10} |
| split_seed | INT | 42 |
| stats | JSONB | {"train":96,"val":12,"test":67,...} |
| minio_path | VARCHAR | "datasets/{ds_id}/versions/v1/" |
| status | ENUM('processing','ready','failed') | |
| created_at | TIMESTAMP | |
| notes | TEXT | |

### pipelines

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| name | VARCHAR UNIQUE | "salad" |
| display_name | VARCHAR | "SALAD (Anomaly Detection)" |
| task_type | VARCHAR | "anomaly_detection" |
| description | TEXT | |
| steps | JSONB | Pipeline step definitions |
| default_config | JSONB | Default hyperparams per step |
| docker_image | VARCHAR | "salad-training:latest" |
| created_at | TIMESTAMP | |

### pipeline_prerequisites

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| pipeline_id | UUID FK → pipelines | |
| name | VARCHAR | "SAM-HQ-ViT-H" |
| minio_path | VARCHAR | "pretrained-models/salad/sam_hq_vit_h.pth" |
| file_size | BIGINT | Bytes |
| required | BOOLEAN | |
| description | TEXT | |

### training_jobs

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| pipeline_id | UUID FK → pipelines | |
| version_id | UUID FK → dataset_versions | Single source of truth (dataset derives from version) |
| owner_id | UUID FK → users | |
| category | VARCHAR | "chip_pcb" |
| config | JSONB | Full config user chọn trên UI |
| status | ENUM('queued','running','completed','failed','cancelled','cancelling') | Job-level status |
| current_step | INT | 1,2,3,4 |
| worker_id | UUID FK → workers | Nullable — assigned khi dispatch |
| idempotency_token | UUID UNIQUE | Prevent duplicate execution on redelivery |
| minio_path | VARCHAR | "training-jobs/{id}/" |
| error_message | TEXT | |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### job_steps

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | |
| step_number | INT | 1, 2, 3, 4 |
| step_name | VARCHAR | "fg_masks", "pseudo_labels", "comp_seg", "salad_training" |
| status | ENUM('pending','running','completed','failed','skipped') | |
| attempt_count | INT | Default 0, max 3 |
| attempts | JSONB | Array of {started_at, ended_at, exit_code, error} |
| artifact_path | VARCHAR | MinIO path cho step output |
| artifact_status | ENUM('none','partial','completed') | Validity marker |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |
| created_at | TIMESTAMP | |

Index: `(job_id, step_number)` UNIQUE

### training_metrics

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL PK | |
| job_id | UUID FK → training_jobs | |
| step | INT | Pipeline step (4 = SALAD) |
| iteration | INT | 0..70000 |
| metrics | JSONB | {"loss_total":0.023,"loss_st":0.008,...} |
| created_at | TIMESTAMP | |

Index: `(job_id, step, iteration)`

### training_evaluations

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | |
| iteration | INT | 10000, 20000, ... |
| auc_combined | FLOAT | 0.5 * (auc_logical + auc_structural) |
| auc_logical | FLOAT | AUROC on logical anomalies vs good (all branches fused) |
| auc_structural | FLOAT | AUROC on structural anomalies vs good (all branches fused) |
| auc_appearance | FLOAT | Appearance branch only (combined) |
| auc_appearance_logical | FLOAT | Appearance branch — logical anomalies vs good |
| auc_appearance_structural | FLOAT | Appearance branch — structural anomalies vs good |
| auc_mahalanobis | FLOAT | Mahalanobis branch only (combined) |
| auc_mahalanobis_logical | FLOAT | Mahalanobis branch — logical anomalies vs good |
| auc_mahalanobis_structural | FLOAT | Mahalanobis branch — structural anomalies vs good |
| auc_composition | FLOAT | Composition branch only (combined) |
| auc_composition_logical | FLOAT | Composition branch — logical anomalies vs good |
| auc_composition_structural | FLOAT | Composition branch — structural anomalies vs good |
| created_at | TIMESTAMP | |

Index: `(job_id, iteration)`

> **Per-branch logical/structural AUC:** Adapter tính được vì có per-image scores từ mỗi branch (appearance_score, mahalanobis_score, composition_score) + labels từ folder structure (good/logical_anomalies/structural_anomalies). Compute AUROC(branch_scores[logical_images], branch_scores[good_images]) cho mỗi branch x anomaly_type.

### models

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | Single lineage source (pipeline, dataset, version derive from job) |
| category | VARCHAR | Denormalized for quick query |
| auc_combined | FLOAT | |
| auc_appearance | FLOAT | |
| auc_mahalanobis | FLOAT | |
| auc_composition | FLOAT | |
| minio_path | VARCHAR | "model-registry/{id}/" |
| has_calibration | BOOLEAN | True if calibration artifacts exist |
| status | ENUM('active','archived','exported') | |
| created_at | TIMESTAMP | |
| notes | TEXT | |

### workers

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| name | VARCHAR UNIQUE | "worker-a" |
| queue_name | VARCHAR UNIQUE | Celery queue name (derived: "queue-{name}") |
| gpu_ids | INT[] | {2} |
| status | ENUM('idle','busy','offline','error') | |
| current_job_id | UUID FK → training_jobs | Nullable — derived from running job, NOT independently settable |
| gpu_server_host | VARCHAR | "192.168.1.100" |
| last_heartbeat | TIMESTAMP | |
| created_at | TIMESTAMP | |

> **Worker-Job consistency:** `workers.current_job_id` is set/cleared atomically with `training_jobs.worker_id` in a single transaction. The source of truth is `training_jobs.worker_id`; `workers.current_job_id` is a denormalized convenience field updated by the same transaction.

### Relationships

```
users ──1:N──▶ datasets ──1:N──▶ dataset_versions
pipelines ──1:N──▶ pipeline_prerequisites
training_jobs ──N:1──▶ pipelines
training_jobs ──N:1──▶ dataset_versions (single lineage source)
training_jobs ──N:1──▶ workers (via worker_id)
training_jobs ──1:N──▶ job_steps
training_jobs ──1:N──▶ training_metrics
training_jobs ──1:N──▶ training_evaluations
training_jobs ──1:1──▶ models (khi thành công, lineage derives from job)
```

---

## 7. API Design

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/auth/register | Register user |
| POST | /api/v1/auth/login | Login → JWT token |
| GET | /api/v1/auth/me | Current user info |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/datasets/ | List datasets (của user) |
| POST | /api/v1/datasets/ | Create dataset record (metadata only) |
| POST | /api/v1/datasets/{id}/upload-url | Get presigned MinIO upload URL for zip |
| POST | /api/v1/datasets/{id}/upload-complete | Notify upload done → trigger async validation |
| GET | /api/v1/datasets/{id} | Dataset detail + upload/validation status |
| DELETE | /api/v1/datasets/{id} | Delete dataset |
| GET | /api/v1/datasets/{id}/images | Browse images (paginated) |
| POST | /api/v1/datasets/{id}/versions | Apply split → create version |
| GET | /api/v1/datasets/{id}/versions | List versions |
| GET | /api/v1/datasets/{id}/versions/{ver_id} | Version detail + stats |

**Dataset Upload Flow (3-step):**
1. `POST /datasets/` → create record, status = 'pending_upload'
2. `POST /datasets/{id}/upload-url` → get presigned PUT URL (client uploads zip directly to MinIO)
3. `POST /datasets/{id}/upload-complete` → trigger Celery task for async extraction + validation
4. Client polls `GET /datasets/{id}` until status = 'ready' or 'failed'

> **Rationale:** Large zip uploads (1-10GB) bypass FastAPI completely — client streams directly to MinIO. Extraction + validation run as background Celery tasks with progress tracking.

### Pipelines

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/pipelines/ | List available pipelines |
| GET | /api/v1/pipelines/{id} | Pipeline detail + steps |
| GET | /api/v1/pipelines/{id}/prerequisites | Required pretrained models |
| GET | /api/v1/pipelines/{id}/config-schema | JSON Schema cho dynamic UI form |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/training/jobs | Create + start training job |
| GET | /api/v1/training/jobs | List jobs (filter by status) |
| GET | /api/v1/training/jobs/{id} | Job detail + status |
| POST | /api/v1/training/jobs/{id}/cancel | Cancel running job |
| POST | /api/v1/training/jobs/{id}/resume | Resume from last completed step (only for failed jobs) |
| GET | /api/v1/training/jobs/{id}/steps | Per-step status, timing, artifacts |
| GET | /api/v1/training/jobs/{id}/metrics | Loss data cho charts |
| GET | /api/v1/training/jobs/{id}/evaluations | AUC data cho charts |
| WS | /api/v1/training/jobs/{id}/logs | WebSocket live log streaming (see Log Streaming below) |
| POST | /api/v1/training/jobs/{id}/ws-ticket | Get short-lived WebSocket auth ticket |
| GET | /api/v1/training/jobs/{id}/logs/{step} | Historical logs (từ MinIO) |
| GET | /api/v1/training/jobs/{id}/logs/{step}?offset={n} | Paginated log lines from offset |

### Model Registry

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/models/ | List trained models |
| GET | /api/v1/models/{id} | Model detail + metrics |
| GET | /api/v1/models/{id}/export | Download model zip |
| POST | /api/v1/models/{id}/archive | Archive model |
| POST | /api/v1/models/compare | So sánh 2+ models |

### Inference (Phase 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/inference/{model_id}/predict | Upload ảnh → anomaly score |
| POST | /api/v1/inference/{model_id}/batch-predict | Upload nhiều ảnh |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/admin/workers | List workers + status |
| POST | /api/v1/admin/workers | Register worker |
| PUT | /api/v1/admin/workers/{id} | Update worker config |
| DELETE | /api/v1/admin/workers/{id} | Remove worker |
| GET | /api/v1/admin/workers/{id}/health | Heartbeat + GPU utilization |
| GET | /api/v1/admin/pretrained-models | List models in MinIO |
| POST | /api/v1/admin/pretrained-models/upload | Upload pretrained model |
| DELETE | /api/v1/admin/pretrained-models/{path} | Delete pretrained model |

### Key Request: Create Training Job

```json
POST /api/v1/training/jobs
{
  "pipeline_id": "salad",
  "version_id": "v1",
  "config": {
    "step_02_pseudo_labels": {
      "n_clusters": 5
    },
    "step_03_comp_seg": {
      "epochs": 10
    },
    "step_04_salad_training": {
      "train_steps": 70000,
      "seed": 42,
      "imagenet_penalty": false
    }
  }
}
```

### Dynamic UI Forms

`GET /api/v1/pipelines/{id}/config-schema` trả về JSON Schema → Next.js dùng để **tự động render form** cho mỗi pipeline step. Thêm pipeline mới chỉ cần đăng ký schema → UI tự generate form.

---

## 8. UI Screens

### 8.1 Dataset Upload

- Hiển thị folder template mẫu (good/, logical_anomalies/, structural_anomalies/)
- Drop zone cho ZIP file
- Sau upload: validation (check folders, image formats, count)
- Nhập dataset name + category

### 8.2 Dataset Browser + Split Config

- Grid view browse ảnh theo folder (good, logical, structural)
- Split configuration: slider cho train/val/test % (chỉ áp dụng cho good/ images)
- Anomaly images luôn vào test set
- Bấm Apply → tạo version mới
- Version history table

### 8.3 Training Configuration

- Chọn pipeline (SALAD), dataset, version
- Category auto-filled từ dataset
- 4 expandable cards cho 4 pipeline steps
- Mỗi card có form fields theo config schema
- Pre-check: system verify pretrained models có đủ trên MinIO
- Bấm Start Training → dispatch job

### 8.4 Training Progress Dashboard

- Pipeline progress bar (Step 1/4, 2/4, 3/4, 4/4)
- Worker info (tên, GPU)
- Tab navigation: Progress | Metrics | Logs | Charts

**Tab Logs:**
- Real-time streaming qua WebSocket
- Filter theo step
- Search trong logs
- Auto-scroll
- Download log file

**Tab Charts:**
- Loss curves (loss_total, loss_st, loss_ae, loss_stae, loss_comp_recon, loss_comp_mask) theo iteration — sampled every 10 iters
- AUC over checkpoints line chart (combined, logical, structural) — sampled every 10k iters
- Per-branch AUC bar chart: grouped bars cho mỗi branch (appearance, mahalanobis, composition), mỗi group có 2 bars (logical vs structural). Tổng 6 bars. Data từ `training_evaluations` columns `auc_{branch}_{type}`
- Compare mode: overlay charts của job khác

> **Chart data contract:** All chart data derives from `training_metrics.metrics` (JSONB) and `training_evaluations` columns. No chart requires data not persisted in these tables.

**Tab Metrics:**
- Table với metrics mỗi checkpoint

### 8.5 Training History

- Table: Job ID, Category, Dataset version, AUC, Status, Duration
- Click → vào detail page (logs, charts, model)
- Filter by status, category, date range

### 8.6 Model Registry

- List trained models với AUC scores
- Compare mode: chọn 2+ models → side-by-side metrics
- Export: download model zip
- Archive: ẩn model khỏi list

### 8.7 Admin: Worker Management

- Table: worker name, GPUs, status (idle/busy/offline), current job
- Add/remove workers
- Health check + GPU utilization

### 8.8 Admin: Pretrained Models

- Grouped by pipeline
- Status per model (có/thiếu trên MinIO)
- Upload / download from URL / delete
- Warning nếu pipeline chưa sẵn sàng (thiếu models)

---

## 9. Project Structure

```
ml_management_v2/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── core/
│   │   │   ├── database.py
│   │   │   ├── storage.py
│   │   │   ├── security.py
│   │   │   └── events.py
│   │   ├── modules/
│   │   │   ├── auth/
│   │   │   │   ├── router.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── service.py
│   │   │   │   └── models.py
│   │   │   ├── datasets/
│   │   │   │   ├── router.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── service.py
│   │   │   │   └── models.py
│   │   │   ├── pipelines/
│   │   │   │   ├── router.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── service.py
│   │   │   │   └── models.py
│   │   │   ├── training/
│   │   │   │   ├── router.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── service.py
│   │   │   │   ├── models.py
│   │   │   │   └── websocket.py
│   │   │   ├── model_registry/
│   │   │   │   ├── router.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── service.py
│   │   │   │   └── models.py
│   │   │   └── admin/
│   │   │       ├── router.py
│   │   │       ├── schemas.py
│   │   │       └── service.py
│   │   └── workers/
│   │       ├── celery_app.py
│   │       ├── tasks.py
│   │       └── runner.py
│   ├── alembic/
│   ├── pyproject.toml
│   └── Dockerfile
│
├── worker/
│   ├── daemon.py
│   ├── celery_worker.py
│   ├── docker_runner.py
│   └── Dockerfile
│
├── training-images/
│   └── salad/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── salad/
│
├── frontend/
│   ├── src/app/
│   │   ├── dashboard/
│   │   ├── datasets/
│   │   ├── training/
│   │   ├── models/
│   │   └── admin/
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml          (Dev: API + DB + Redis + MinIO)
├── docker-compose.gpu.yml      (GPU server: worker daemon)
└── Makefile
```

---

## 10. Phase Roadmap

### Phase 1 (Current)

- Dataset upload + versioning
- SALAD training pipeline (full 4 steps)
- Live logs + charts + metrics
- Model registry + export
- Worker management
- Admin: pretrained models

### Phase 2

- Inference API (predict single image + batch)
- Inference UI (upload ảnh → xem anomaly score + heatmap)
- Multi-tenant (user isolation)

### Phase 3

- Thêm pipelines mới (không chỉ SALAD)
- Auto-scaling workers
- Scheduled training (retrain khi có data mới)
- A/B testing models

---

## 11. Log Streaming Architecture

### Durable Log Pipeline

Training containers ghi logs qua stdout → worker daemon capture → Redis Pub/Sub + MinIO immutable chunks.

**Write path:**
1. Container stdout → worker daemon tee vào local buffer
2. Buffer flush mỗi 1s hoặc 100 lines (whichever first):
   - Publish batch tới Redis channel `job:{job_id}:step:{step}:logs`
   - Write immutable chunk tới MinIO: `training-jobs/{job_id}/logs/step_{N}/chunks/{offset}.jsonl`
   - Each chunk file: `[{offset, timestamp, line}, ...]`
3. Chunk index file updated: `training-jobs/{job_id}/logs/step_{N}/index.json` — list of chunk offsets + sizes

**Read path (WebSocket):**
1. Client connect `WS /api/v1/training/jobs/{id}/logs?step={N}&offset={last_offset}`
2. Server backfill: read chunks from MinIO index where offset >= `last_offset`
3. Server subscribe Redis Pub/Sub → stream new lines
4. Client nhận `{offset, step, lines[], eof?}` messages
5. Disconnect → client lưu last offset → reconnect resume từ đó

> **Rationale:** MinIO objects are immutable (no append). Each chunk is a small, versioned JSONL file (~1-100KB). Backfill reads chunk index then fetches needed chunks. No object mutations.

**Step-aware filtering:** Client có thể filter logs theo step. Backend chỉ subscribe/stream step được request.

**EOF marker:** Khi step hoàn tất, worker gửi `{eof: true, step: N}` message. Client biết step log đã xong.

**Authentication:** WebSocket dùng **short-lived ticket** thay vì JWT trực tiếp:
1. Client gọi `POST /api/v1/training/jobs/{id}/ws-ticket` (authenticated via HttpOnly cookie)
2. Server trả `{ticket: "random-token", expires_in: 30}` — single-use, 30s TTL, stored in Redis
3. Client connect `WS /api/v1/training/jobs/{id}/logs?ticket={ticket}`
4. Server validate ticket (exists in Redis + not expired + matches job_id) → delete ticket → upgrade connection
5. Ticket không xuất hiện trong logs vì single-use và expired ngay sau validate

> **Rationale:** HttpOnly cookies không đọc được từ JS, nên không thể dùng trực tiếp cho WebSocket query param. JWT trong query string dễ leak qua logs/history. Short-lived ticket giải quyết cả hai vấn đề.

---

## 12. Runtime Dependency Policy

### Air-gapped Training Containers

**Policy:** Training containers KHÔNG được phép fetch dependencies từ internet.

**Mirroring strategy:**
1. **Pretrained weights:** Tất cả `.pth` files phải có sẵn trong MinIO `pretrained-models/`
2. **DINO ViT-B/8:** Script `create_pseudo_labels.py` dùng `torch.hub.load('facebookresearch/dino:main')`. Container image phải bake `torch.hub` cache hoặc wrapper script load từ MinIO path.
3. **SAM / SAM-HQ:** Weights từ MinIO, không download runtime.
4. **Python packages:** Tất cả pip packages bake vào Docker image. Không `pip install` lúc runtime.
5. **Focal loss:** `kornia.losses.FocalLoss` hoặc custom implementation — bake vào image.

**Docker image build contract:**
```dockerfile
# salad-training:latest must include:
# - All pip packages (torch, torchvision, timm, segment-anything-hq, etc.)
# - DINO ViT-B/8 weights in /cache/torch/hub/
# - Python 3.11
# - CUDA 12.4 runtime
# NO network access during training
```

**Verification:** CI pipeline build image + run smoke test (dry-run 10 iterations) without network access.

---

## 13. Security

### Authentication & Authorization

| Scope | Rule |
|-------|------|
| JWT tokens | Access token (15m) + Refresh token (7d). HttpOnly cookies, SameSite=Strict. |
| Datasets | Owner-only: user chỉ thấy/sửa/xóa datasets mình tạo. Admin thấy all. |
| Training jobs | Owner-only: chỉ owner tạo/cancel/resume job. Admin can view all. |
| Models | Owner-only: derive ownership từ job.owner_id. Admin can view all. |
| Workers / Admin endpoints | Admin-only: `role = 'admin'` required. Return 403 otherwise. |
| WebSocket | Short-lived ticket via `POST /ws-ticket` endpoint (see Section 11). NOT raw JWT in query string. |

### Upload Hardening

| Threat | Defense |
|--------|---------|
| Zip bomb | Max decompressed size = 50GB. Abort extraction if exceeded. |
| Path traversal | Strip all `../` and absolute paths from zip entries. Validate against allowlist. |
| Malicious files | Only allow image extensions (.png, .jpg, .jpeg, .bmp, .tiff). Reject others silently. |
| File size | Max individual file = 100MB. Max zip = 10GB. |
| Upload rate | Rate limit: 5 uploads/hour per user. |

### Container Isolation

| Control | Implementation |
|---------|---------------|
| Non-root | Containers run as UID 1000, not root. |
| Read-only rootfs | `--read-only` except /workspace and /tmp. |
| No network | `--network=none` — containers cannot access internet or other services. MinIO data pre-pulled to /workspace. |
| GPU isolation | `CUDA_VISIBLE_DEVICES` strictly set. `--device` flag for specific GPUs only. |
| Resource limits | `--memory`, `--cpus`, `--shm-size` set per worker config. |
| Tmpdir cleanup | /workspace mounted as tmpfs or host tmpdir, cleaned after job. |

### Secret Management

- Database credentials, MinIO keys, JWT secret: **environment variables** loaded from `.env` (dev) or **Docker secrets / cloud KMS** (prod).
- No secrets in code, config files, or Docker images.
- Worker daemon authenticates to API with service account JWT (long-lived, admin-scoped, rotated monthly).
