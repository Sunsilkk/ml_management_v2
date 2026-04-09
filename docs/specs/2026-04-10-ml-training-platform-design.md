# ML Training Platform — Design Spec

**Date:** 2026-04-10
**Status:** Approved
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

- Mỗi worker có `concurrency=1` (1 job tại 1 thời điểm)
- Job vào Redis queue → worker trống tự pick
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
| Job thất bại | Upload error logs → MinIO, xóa container + data |
| Job timeout | Kill container, upload logs, xóa sạch |

GPU server luôn sạch sau mỗi job — không tích tụ data rác.

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
  - DINO-ViT-B/8 (auto-download từ Hugging Face)

| Parameter | Default | UI Control |
|-----------|---------|------------|
| n_clusters | 5 | Number input |
| iou_threshold | 0.8 | Number input |
| min_mask_area | 50 px | Number input |

#### Step 3: Composition Segmentation Training

- **Script:** `train_composition_segmentation_model.py`
- **Input:** Raw images + noisy composition maps từ Step 2
- **Output:** Clean composition maps cho train/validation/test

| Parameter | Default | UI Control |
|-----------|---------|------------|
| epochs | 10 | Number input |
| learning_rate | 1e-4 | Number input |
| batch_size | 8 | Number input |

#### Step 4: SALAD Main Training

- **Script:** `train_salad.py`
- **Input:** Raw images + clean composition maps từ Step 3
- **Output:** 5 trained models (.pth)
- **Prerequisites:** `pretrained-models/salad/teacher_medium.pth` (~32MB)

**Training Parameters:**

| Category | Parameter | Default | UI Control |
|----------|-----------|---------|------------|
| Training | train_steps | 70000 | Number input |
| Training | seed | 42 | Number input |
| Training | checkpoint_interval | 10000 | Number input |
| Optimizer | lr_appearance (Student + AE) | 1e-4 | Number input |
| Optimizer | lr_composition (Comp AE + UNet) | 1e-5 | Number input |
| Optimizer | weight_decay | 1e-5 | Number input |
| Optimizer | lr_decay_gamma | 0.1 | Number input |
| Optimizer | lr_decay_at_percent | 95% of steps | Number input |
| Loss | focal_loss_gamma | 2 | Number input |
| Loss | comp_mask_loss_weight | 5 | Number input |
| Loss | hard_sample_quantile | 0.999 | Number input |
| Data | image_size | 256x256 | Number input |
| Data | batch_size | 1 | Number input |
| Data | num_workers | 4 | Number input |
| Data | imagenet_penalty | OFF | Toggle |

**ImageNet Penalty:** Optional. Khi ON → pull ImageNet data từ MinIO (`pretrained-models/imagenet/`). Mặc định OFF — đủ tốt cho custom datasets.

### Training Output (5 models)

| Model | File | Vai trò |
|-------|------|---------|
| Teacher | teacher_final.pth | Frozen reference network |
| Student | student_final.pth | Appearance anomaly detection |
| AutoEncoder | autoencoder_final.pth | Appearance reconstruction |
| Comp AE | comp_ae_final.pth | Composition reconstruction |
| Comp UNet | comp_unet_final.pth | Composition anomaly mask |

### Evaluation Metrics (mỗi checkpoint)

- AUC Combined (0.5 * logical + 0.5 * structural)
- AUC Appearance
- AUC Mahalanobis
- AUC Composition

---

## 5. MinIO Storage Structure

```
MinIO
├── pretrained-models/
│   ├── salad/
│   │   ├── teacher_medium.pth
│   │   ├── sam_vit_h_4b8939.pth
│   │   ├── sam_hq_vit_h.pth
│   │   └── dino_vitbase8/dino_vitb8.pth
│   ├── imagenet/train/...              (optional, ~150GB)
│   └── {future-pipeline}/...
│
├── datasets/
│   └── {dataset_id}/
│       ├── meta.json
│       ├── raw/dataset.zip
│       └── versions/{version_id}/
│           ├── version.json
│           ├── train/good/...
│           ├── validation/{good,logical,structural}/...
│           └── test/{good,logical,structural}/...
│
├── training-jobs/
│   └── {job_id}/
│       ├── job.json
│       ├── pipeline-config.json
│       ├── steps/
│       │   ├── 01_fg_masks/{status.json, masks/...}
│       │   ├── 02_pseudo_labels/{status.json, composition_maps/...}
│       │   ├── 03_comp_seg_training/{status.json, model.pth, composition_maps/...}
│       │   └── 04_salad_training/
│       │       ├── status.json
│       │       ├── checkpoints/{iter_10000/,iter_20000/,...}
│       │       └── final/{teacher,student,autoencoder,comp_ae,comp_unet}_final.pth
│       ├── logs/{step_01.log, step_02.log, step_03.log, step_04.log}
│       └── metrics/
│           ├── loss.jsonl      (1 dòng = 1 iteration)
│           ├── eval.jsonl      (1 dòng = 1 checkpoint evaluation)
│           └── summary.json    (kết quả cuối cùng)
│
└── model-registry/
    └── {model_id}/
        ├── model.json          (metadata: pipeline, category, auc, job_id)
        ├── weights/{5 .pth files}
        └── export/{category}_salad_{version}.zip
```

**Nguyên tắc:**

- `pretrained-models/` — Tổ chức theo pipeline. Khi chọn pipeline → hệ thống biết cần pull những gì.
- `datasets/` — Tổ chức theo dataset_id → versions. Giữ raw zip gốc + mỗi version là 1 snapshot sau split.
- `training-jobs/` — Toàn bộ output 1 job: intermediate files, logs, metrics, checkpoints. Có thể cleanup cũ.
- `model-registry/` — Chỉ final models sẵn sàng dùng. Nhẹ, giữ lâu dài.

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
| dataset_id | UUID FK → datasets | |
| version_id | UUID FK → dataset_versions | |
| owner_id | UUID FK → users | |
| category | VARCHAR | "chip_pcb" |
| config | JSONB | Full config user chọn trên UI |
| status | ENUM('queued','pulling','preprocessing','training','uploading','completed','failed','cancelled') | |
| current_step | INT | 1,2,3,4 |
| worker_name | VARCHAR | "worker-a" |
| minio_path | VARCHAR | "training-jobs/{id}/" |
| error_message | TEXT | |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

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
| auc_combined | FLOAT | |
| auc_appearance | FLOAT | |
| auc_mahalanobis | FLOAT | |
| auc_composition | FLOAT | |
| created_at | TIMESTAMP | |

Index: `(job_id, iteration)`

### models

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | |
| pipeline_id | UUID FK → pipelines | |
| dataset_id | UUID FK → datasets | |
| version_id | UUID FK → dataset_versions | |
| category | VARCHAR | |
| auc_combined | FLOAT | |
| auc_appearance | FLOAT | |
| auc_mahalanobis | FLOAT | |
| auc_composition | FLOAT | |
| minio_path | VARCHAR | "model-registry/{id}/" |
| status | ENUM('active','archived','exported') | |
| created_at | TIMESTAMP | |
| notes | TEXT | |

### workers

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| name | VARCHAR UNIQUE | "worker-a" |
| gpu_ids | INT[] | {2} |
| status | ENUM('idle','busy','offline','error') | |
| current_job_id | UUID FK → training_jobs | Nullable |
| gpu_server_host | VARCHAR | "192.168.1.100" |
| last_heartbeat | TIMESTAMP | |
| created_at | TIMESTAMP | |

### Relationships

```
users ──1:N──▶ datasets ──1:N──▶ dataset_versions
pipelines ──1:N──▶ pipeline_prerequisites
pipelines + dataset_versions ──▶ training_jobs ──1:N──▶ training_metrics
                                                ──1:N──▶ training_evaluations
                                 training_jobs ──1:1──▶ models (khi thành công)
workers ──1:1──▶ training_jobs.current_job_id
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
| POST | /api/v1/datasets/ | Upload zip + create dataset |
| GET | /api/v1/datasets/{id} | Dataset detail |
| DELETE | /api/v1/datasets/{id} | Delete dataset |
| GET | /api/v1/datasets/{id}/images | Browse images (paginated) |
| POST | /api/v1/datasets/{id}/versions | Apply split → create version |
| GET | /api/v1/datasets/{id}/versions | List versions |
| GET | /api/v1/datasets/{id}/versions/{ver_id} | Version detail + stats |

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
| GET | /api/v1/training/jobs/{id}/metrics | Loss data cho charts |
| GET | /api/v1/training/jobs/{id}/evaluations | AUC data cho charts |
| WS | /api/v1/training/jobs/{id}/logs | WebSocket live log streaming |
| GET | /api/v1/training/jobs/{id}/logs/{step} | Historical logs (từ MinIO) |

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
  "dataset_id": "ds_001",
  "version_id": "v1",
  "config": {
    "step_02_pseudo_labels": {
      "n_clusters": 5,
      "iou_threshold": 0.8,
      "min_mask_area": 50
    },
    "step_03_comp_seg": {
      "epochs": 10,
      "learning_rate": 1e-4,
      "batch_size": 8
    },
    "step_04_salad_training": {
      "train_steps": 70000,
      "seed": 42,
      "lr_appearance": 1e-4,
      "lr_composition": 1e-5,
      "weight_decay": 1e-5,
      "lr_decay_gamma": 0.1,
      "lr_decay_at_percent": 95,
      "focal_loss_gamma": 2,
      "comp_mask_loss_weight": 5,
      "hard_sample_quantile": 0.999,
      "image_size": 256,
      "batch_size": 1,
      "num_workers": 4,
      "imagenet_penalty": false,
      "checkpoint_interval": 10000
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
- Loss curves (total, appearance, comp_recon, comp_mask) theo iteration
- AUC over checkpoints (combined, appearance, mahalanobis, composition)
- Per-branch breakdown bar chart
- Compare mode: overlay charts của job khác

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
