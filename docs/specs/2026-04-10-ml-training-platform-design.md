# ML Training Platform — Design Spec

**Date:** 2026-04-10
**Status:** Draft (Revision 9 — post Codex review Round 7)
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

**Execution ownership:**
- **Celery = dispatch mechanism only** — nhận lệnh "run job X on worker Y", NOT execution owner
- **Worker daemon = execution owner** — pull data, spawn container, watch IPC, report status
- **DB = sole source of truth** for job state — daemon reports, API updates, UI reads
- Container lifecycle do daemon state machine điều khiển, NOT Celery task

**Phase 1 execution unit (R2-ISSUE-1 clarification):**
- **1 Celery task = 1 job = 1 container** — single container runs all 4 steps sequentially
- Celery dispatches ONE task per job. Daemon receives task, spawns ONE container that runs `salad_pipeline.pipeline.SALADPipeline.run()` which internally chains step1→step2→step3→step4
- Per-step tracking (job_steps table) is for observability and resume, NOT for per-step Celery dispatch
- Daemon's IPC watcher receives step_start/step_end events from container → updates job_steps rows via internal API
- Retry scope: if step fails, container exits → daemon retries entire container from last completed step (resume semantics)
- Worker lease = entire job duration (hours). Celery visibility_timeout = 24h to prevent phantom redelivery

**Registration flow (pull-based — ISSUE-8 fix):**

1. Admin vào Dashboard → khai báo "worker-a, GPU 2" và "worker-b, GPU 3"
2. Worker Daemon boot → gọi `POST /api/v1/internal/workers/register` → nhận config
3. Daemon poll `GET /api/v1/internal/workers/{name}/config` mỗi 60s → reconcile local state
4. Admin update config trên Dashboard → daemon pick up next poll cycle
5. Daemon tự spawn Celery workers với đúng `CUDA_VISIBLE_DEVICES`
6. Admin không cần SSH hay chạy lệnh tay

### Job Scheduling

**Dispatch Contract:**

- **Queue routing:** Mỗi worker có dedicated queue (`celery -Q worker-a`). Job dispatch chỉ định queue cụ thể, không dùng shared queue.
- **Atomic job lease:** Trước khi dispatch, API set `training_jobs.status = 'queued'` + `worker_name = 'worker-a'` trong 1 transaction. Nếu worker đã busy → reject, chọn worker khác.
- **Idempotent start:** Mỗi job có `idempotency_token` (UUID). Worker kiểm tra token trước khi bắt đầu — nếu đã có container chạy cùng token → skip (tránh duplicate từ Celery redelivery).
- **Late acknowledgement:** Task sử dụng `acks_late=True` + `reject_on_worker_lost=True`. Celery chỉ ack khi job hoàn tất/fail, không ack khi nhận.
- **Heartbeat:** Worker gửi heartbeat mỗi 30s (`workers.last_heartbeat`). API coi worker offline nếu miss 3 heartbeats liên tiếp (90s). Job trên offline worker sẽ được re-queue — **nhưng CHỈ sau khi fencing check:** API gọi daemon health endpoint trên GPU server. Requeue CHỈ khi (a) daemon confirms container exited, HOẶC (b) daemon itself unreachable (heartbeat miss + health-check timeout 10s). Tránh requeue khi daemon healthy nhưng chỉ bị network blip tạm thời.
- **Visibility timeout:** Celery `visibility_timeout = 86400` (24h, vì training jobs chạy lâu). Ngắn hơn sẽ gây phantom redelivery.
- **Run attempt fencing (ISSUE-2 fix):** Mỗi job có `current_attempt_id` (UUID, updated mỗi lần dispatch/requeue). Worker daemon nhận `attempt_id` khi bắt đầu job. Mọi write operations (metrics, checkpoint upload, status update) phải kèm `attempt_id` — API reject nếu `attempt_id != training_jobs.current_attempt_id`. Khi requeue: API tạo new `attempt_id` trước, old attempt tự bị fenced out dù container vẫn chạy. Container sẽ bị kill bởi daemon cleanup hoặc tự exit khi writes bị reject.
- **MinIO attempt fencing (R2-ISSUE-2 fix):** ALL attempt-scoped artifacts write under `training-jobs/{job_id}/attempts/{attempt_id}/` prefix. Only the active fenced attempt can write. On job completion, daemon promotes final weights from active attempt prefix → model registry (`model-registry/{model_id}/weights/`). No canonical `training-jobs/{job_id}/final/` path — all mutable data stays under attempt prefix. Stale attempt prefixes cleaned up by retention policy. This prevents overlapping MinIO writes from duplicate executors.
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

### MinIO Retention Lifecycle (ISSUE-6 fix)

| Bucket path | Retention | Trigger |
|---|---|---|
| `training-jobs/{id}/` | 30 days after job completion (includes all attempt subdirs) | Async cleanup task, configurable via `TRAINING_JOB_RETENTION_DAYS` |
| `training-jobs/{id}/attempts/{attempt_id}/checkpoints/` | Deleted on job completion (final weights promoted to model-registry). Stale attempt checkpoints cleaned after 7 days. | Post-completion hook + retention lifecycle |
| `model-registry/{id}/` | Permanent until admin archive/delete | Manual only |
| `datasets/{id}/versions/` | Permanent until soft-delete + no dependent jobs/models (R2-ISSUE-4 fix) | API enforces dependency check → 409 if references exist |
| `pretrained-models/` | Permanent | Manual only |

**Rules:**
- Job artifacts (logs, metrics, intermediate) auto-expire after retention window
- Registered models are NEVER auto-deleted — only admin action
- Checkpoint cleanup happens AFTER successful model promotion to registry
- Failed jobs retain artifacts for `FAILED_JOB_RETENTION_DAYS` (default 7) for debugging

### Failure, Retry, Resume & Cancellation

**Per-step tracking:** Mỗi step có record riêng trong `job_steps` table (xem Schema bên dưới) với status, timing, retry count, artifact paths.

**Retry policy:**
- Mỗi step tối đa `max_retries = 2` (configurable per pipeline)
- Transient failures (OOM, container crash) → auto-retry cùng step, không reset steps trước
- Persistent failures (3 fails liên tiếp) → job status = 'failed', partial artifacts giữ nguyên trên MinIO
- Mỗi retry attempt ghi vào `job_steps.attempts` (JSONB array)

**Resume semantics (Phase 1 scope — ISSUE-7 fix):**
- API endpoint `POST /api/v1/training/jobs/{id}/resume` cho phép resume từ step cuối thành công
- Resume tạo new attempt cho failed step, giữ nguyên output các steps trước
- **Phase 1: checkpoint resume CHỈ support Step 4.** Step 1–3 fail → rerun nguyên step (không checkpoint). Step 4 resume: nếu `job_steps.last_checkpoint_path` có giá trị → resume training từ checkpoint iteration (best-effort, not bit-exact — xem Section 4 Checkpoint & Resume). Nếu NULL → rerun Step 4 từ iteration 0.
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

### Pipeline Overview

SALAD code gốc (tại `Anomaly-Dectection/SALAD/`) là research scripts — params hard-coded, không callback, không resume, không structured output. Platform sẽ sử dụng **`salad_pipeline/`** — production library viết lại hoàn toàn từ SALAD, giữ nguyên logic training nhưng:
- Tất cả params configurable qua Pydantic configs (2-tier: Basic + Advanced)
- Callback Protocol cho platform integration (structured metrics, progress reporting)
- Full checkpoint/resume với optimizer + RNG state
- Device-agnostic (không hard-code `.cuda()`)
- state_dict serialization (không dùng pickle whole model)

> **Reference:** `Anomaly-Dectection/SALAD/` giữ nguyên làm baseline — không sửa.

### Pipeline Steps (tuần tự)

```
Step 1: FG Masks (SAM) → Step 2: Pseudo Labels (DINO+SAM-HQ) → Step 3: Comp Seg (UNet) → Step 4: Main Training (5 models)
```

#### Step 1: Foreground Mask Generation (SAM)

- **Module:** `salad_pipeline.steps.step1_fg_masks`
- **Input:** Raw images từ dataset
- **Output:** Binary foreground masks
- **Prerequisites:** `pretrained-models/salad/sam_vit_h_4b8939.pth` (~2.5GB)

| Tier | Parameter | Type | Default | UI Control | Notes |
|------|-----------|------|---------|------------|-------|
| Basic | category | str | — | Text | Required |
| Basic | dataset_path | Path | — | Auto (from job) | |
| Basic | output_path | Path | — | Auto (from job) | |
| Basic | sam_checkpoint | Path | — | Auto (from MinIO) | |
| Advanced | image_size | int | 256 | Number | Frozen — tied to architecture |
| Advanced | sam_model_type | str | "vit_h" | Select | |

#### Step 2: Pseudo Label Generation (DINO + SAM-HQ)

- **Module:** `salad_pipeline.steps.step2_pseudo_labels`
- **Input:** Raw images + FG masks từ Step 1
- **Output:** Noisy composition maps (6-class one-hot)
- **Prerequisites:**
  - `pretrained-models/salad/sam_hq_vit_h.pth` (~2.5GB)
  - DINO-ViT-B/8 — **phải mirror sẵn trong Docker image** (script dùng `torch.hub` để load, cần bake weights vào image hoặc cache trong MinIO tại `pretrained-models/salad/dino_vitbase8/`)

| Tier | Parameter | Type | Default | UI Control | Notes |
|------|-----------|------|---------|------------|-------|
| Basic | category | str | — | Text | Required |
| Basic | n_clusters | int | 5 | Number input | KMeans clusters cho DINO features |
| Basic | dataset_path | Path | — | Auto | |
| Basic | mask_path | Path | — | Auto (Step 1 output) | |
| Basic | output_path | Path | — | Auto | |
| Basic | sam_hq_checkpoint | Path | — | Auto (MinIO) | |
| Advanced | points_per_side | int | 64 | Number | SAM-HQ mask sampling density |
| Advanced | pred_iou_thresh | float | 0.8 | Number | SAM-HQ prediction confidence |
| Advanced | min_mask_region_area | int | 50 | Number | Minimum mask area in pixels |
| Advanced | crop_n_layers | int | 0 | Number | SAM-HQ crop layers |
| Advanced | feature_sampling_divisor | int | 500 | Number | 256²/N pixels sampled for DINO |

#### Step 3: Composition Segmentation Training

- **Module:** `salad_pipeline.steps.step3_comp_seg`
- **Input:** Raw images + noisy composition maps từ Step 2
- **Output:** Clean composition maps cho train/validation/test (NOT a model artifact)

| Tier | Parameter | Type | Default | UI Control | Notes |
|------|-----------|------|---------|------------|-------|
| Basic | category | str | — | Text | Required |
| Basic | epochs | int | 10 | Number input | Training epochs |
| Basic | n_clusters | int | 5 | Number | Phải match Step 2 |
| Advanced | batch_size | int | 8 | Number | |
| Advanced | learning_rate | float | 1e-4 | Number | Adam optimizer |
| Advanced | lr_milestones | list[int] | [100] | JSON | MultiStepLR milestones |
| Advanced | lr_gamma | float | 0.2 | Number | MultiStepLR decay factor |
| Advanced | unet_base_channels | int | 64 | Number | UNet architecture |
| Advanced | unet_ch_mults | list[int] | [1,2,2,2] | JSON | Channel multipliers |

#### Step 4: SALAD Main Training

- **Module:** `salad_pipeline.steps.step4_training`
- **Input:** Raw images + clean composition maps từ Step 3
- **Output:** 5 trained models (.pth) + calibration artifacts
- **Prerequisites:** `pretrained-models/salad/teacher_medium.pth` (~32MB)

**Basic Parameters (UI hiển thị mặc định):**

| Parameter | Type | Default | UI Control | Notes |
|-----------|------|---------|------------|-------|
| category | str | — | Text | Required |
| train_steps | int | 70000 | Number input | Total iterations |
| seed | int | 42 | Number input | RNG seed (re-applied after model init) |
| imagenet_path | Path \| None | None | Toggle + path | Optional penalty transform |
| dataset_path | Path | — | Auto (from job) | |
| comp_map_path | Path | — | Auto (Step 3 output) | |
| output_dir | Path | — | Auto (from job) | |
| teacher_weights | Path | — | Auto (MinIO) | |

**Advanced Parameters (ẩn trong "Advanced Settings" accordion):**

| Category | Parameter | Type | Default | Notes |
|----------|-----------|------|---------|-------|
| Optimizer | lr_main | float | 1e-4 | Student + AE learning rate |
| Optimizer | lr_comp | float | 1e-5 | Comp AE + Comp UNet LR (10x lower) |
| Optimizer | weight_decay | float | 1e-5 | Adam weight decay |
| Scheduler | scheduler_milestone_ratio | float | 0.95 | StepLR milestone = ratio × train_steps |
| Scheduler | scheduler_gamma | float | 0.1 | LR decay factor |
| Training | batch_size | int | 1 | DataLoader batch size |
| Training | num_workers | int | 4 | DataLoader workers |
| Training | checkpoint_interval | int | 10000 | Save checkpoint every N iters |
| Training | eval_interval | int | 10000 | Run eval every N iters. **Constraint: eval_interval must equal checkpoint_interval** (R2-ISSUE-3 fix — ensures every eval has a matching durable checkpoint for promotion) |
| Training | log_interval | int | 10 | Log losses every N iters |
| Loss | focal_loss_gamma | int | 2 | Multiclass focal loss gamma |
| Loss | comp_mask_weight | float | 5.0 | Focal loss weight for comp mask |
| Loss | hard_quantile | float | 0.999 | Student-teacher hard mining quantile |
| Loss | dice_smooth | float | 1e-5 | Dice loss smoothing factor |
| Norm | map_quantile_start | float | 0.9 | Anomaly map normalization start quantile |
| Norm | map_quantile_end | float | 0.995 | Anomaly map normalization end quantile |

**Frozen Parameters (read-only, tied to architecture):**

| Parameter | Value | Reason |
|-----------|-------|--------|
| out_channels | 384 | PDN Medium output dimension — mọi downstream code phụ thuộc |
| image_size | 256 | AE decoder upsample stages hard-wired [3,8,15,32,63,127,56] |

**ImageNet Penalty:** Optional. Khi ON → pull ImageNet data từ MinIO (`pretrained-models/imagenet/`). Mặc định OFF — đủ tốt cho custom datasets.

**Canonical Parameter Mapping (Library ↔ Baseline SALAD):**

| Library Param | Baseline Equivalent | Source | Mapping |
|---------------|--------------------|---------| --------|
| `train_steps` | `argparser -t` (default 70000) | Baseline-equivalent | 1:1 |
| `seed` | `argparser --seed` (default 42) | Baseline-equivalent | 1:1 |
| `imagenet_path` | `argparser -i` (default `./data/imagenet/train`) | Intentional deviation | Baseline ON by default; library OFF by default (custom datasets ko có ImageNet) |
| `lr_main` | hard-coded `1e-4` (L202 train_salad.py) | Baseline-equivalent | Extracted from hard-code |
| `lr_comp` | hard-coded `1e-5` (L201 train_salad.py) | Baseline-equivalent | Extracted from hard-code |
| `weight_decay` | hard-coded `1e-5` (L202) | Baseline-equivalent | Extracted |
| `batch_size` | hard-coded `1` (L169) | Baseline-equivalent | Extracted |
| `checkpoint_interval` | hard-coded `10000` (L293) | Baseline-equivalent | Extracted |
| `eval_interval` | hard-coded `10000` (L305) | Baseline-equivalent | Extracted |
| `log_interval` | hard-coded `10` (L280) | Baseline-equivalent | Extracted |
| `n_clusters` (Step 2) | `argparser --n-clusters` (default 5) | Baseline-equivalent | 1:1 |
| `epochs` (Step 3) | `argparser --epochs` (default 10) | Baseline-equivalent | 1:1 |
| `points_per_side` (Step 2) | hard-coded `64` (L94 create_pseudo_labels.py) | Baseline-equivalent | Extracted |
| `lr_milestones` (Step 3) | hard-coded `[100]` (L97 train_composition_segmentation_model.py) | Baseline-equivalent | Extracted |
| `lr_gamma` (Step 3) | hard-coded `0.2` (L102) | Baseline-equivalent | Extracted |
| `comp_mask_weight` (Step 4) | hard-coded `5` (L271: `5*focal_loss(...)`) | Baseline-equivalent | Extracted from inline constant |

> **Key deviation:** `imagenet_path` default flipped from ON→OFF. Baseline SALAD assumed MVTec LOCO environment where ImageNet was always available. Platform default OFF because custom user datasets may not include ImageNet — user explicitly enables if desired.

### Training Output (5 models)

| Model | Actual Filename | Vai trò |
|-------|----------------|---------|
| Teacher | teacher_final.pth | Frozen reference network |
| Student | student_final.pth | Appearance anomaly detection |
| AutoEncoder | autoencoder_final.pth | Appearance reconstruction |
| Comp AE | comp_autoencoder_final.pth | Composition reconstruction |
| Comp UNet | comp_unet_final.pth | Composition anomaly mask |

> **Important:** Filenames PHẢI match exactly. Spec dùng `comp_autoencoder_final.pth` (không phải `comp_ae_final.pth`).

### Evaluation Metrics (mỗi checkpoint)

- AUC Combined (0.5 * logical + 0.5 * structural)
- AUC Logical (logical anomalies vs good — all branches fused)
- AUC Structural (structural anomalies vs good — all branches fused)
- AUC Appearance (combined) + Appearance Logical + Appearance Structural
- AUC Mahalanobis (combined) + Mahalanobis Logical + Mahalanobis Structural
- AUC Composition (combined) + Composition Logical + Composition Structural

> Total 12 AUC metrics per checkpoint (4 branches × 3 types each). Per-branch logical/structural tính từ per-image branch scores grouped by anomaly type. Library computes all 12 internally.

### Callback Protocol (Platform Integration)

`salad_pipeline` expose callback protocol để platform hook vào — thay thế hoàn toàn ClearML integration + stdout parsing.

```python
class TrainingCallback(Protocol):
    """Platform implements this to receive structured events."""

    def on_step_start(self, step_name: str, step_index: int, total_steps: int) -> None: ...
    def on_step_end(self, step_name: str, step_index: int, artifacts: dict[str, Path]) -> None: ...

    def on_train_start(self, config: TrainingConfig, total_iterations: int) -> None: ...
    def on_train_iteration(self, iteration: int, losses: dict[str, float]) -> None: ...
    def on_checkpoint_saved(self, iteration: int, checkpoint_dir: Path) -> None: ...
    def on_eval_start(self, iteration: int) -> None: ...
    def on_eval_end(self, iteration: int, metrics: dict[str, float]) -> None: ...
    def on_train_end(self, final_metrics: dict[str, float], artifacts: dict[str, Path]) -> None: ...

    def on_error(self, step_name: str, error: Exception) -> None: ...
    def should_stop(self) -> bool: ...  # early termination check (cancel support)
```

**Loss dict keys** (fired every `log_interval` iterations via `on_train_iteration`):
```python
{"loss_total", "loss_st", "loss_ae", "loss_stae", "loss_comp_recon", "loss_comp_mask"}
```

**Eval metrics dict keys** (fired every `eval_interval` iterations via `on_eval_end`):
```python
{
    # Combined (all branches fused) — 3 fields
    "auc_combined",                    # 0.5 * (logical + structural)
    "auc_combined_logical",            # all-branches fused, logical anomalies vs good
    "auc_combined_structural",         # all-branches fused, structural anomalies vs good
    # Appearance branch — 3 fields
    "auc_appearance",                  # 0.5 * (appearance_logical + appearance_structural)
    "auc_appearance_logical",
    "auc_appearance_structural",
    # Mahalanobis branch — 3 fields
    "auc_mahalanobis",
    "auc_mahalanobis_logical",
    "auc_mahalanobis_structural",
    # Composition branch — 3 fields
    "auc_composition",
    "auc_composition_logical",
    "auc_composition_structural",
}
# Total: 12 fields. Library computes all 12 internally from per-image branch scores.
```

Platform ships a `FilesystemCallback` implementation (see Section 5 IPC Architecture for details):
1. `on_train_iteration` → append `loss.jsonl` to `/workspace/ipc/` (local file only)
2. `on_eval_end` → append `eval.jsonl` to `/workspace/ipc/` (12 fields, all computed by library)
3. `on_checkpoint_saved` → write checkpoint to `/workspace/ipc/checkpoint/` (atomic: write to `.tmp/` then rename)
4. `on_train_end` → write final artifacts to `/workspace/ipc/final/`
5. `should_stop()` → check if `/workspace/ipc/should_stop` file exists (touched by host watcher on cancel)

> **Ownership model:** Library computes all metrics (losses + 12 AUCs). FilesystemCallback only writes IPC files. Host-side watcher only forwards to Redis/DB/MinIO. No component reaches beyond its boundary.

### Checkpoint & Resume

**Checkpoint format** (saved mỗi `checkpoint_interval` iterations):

```python
{
    "iteration": int,
    "models": {
        "teacher": state_dict,       # teacher.state_dict()
        "student": state_dict,
        "autoencoder": state_dict,
        "comp_ae": state_dict,
        "comp_unet": state_dict,
    },
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "rng_states": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    },
    "teacher_mean": Tensor,          # [1, 384, 1, 1]
    "teacher_std": Tensor,           # [1, 384, 1, 1]
    "config": config.model_dump(),   # full config snapshot
    "loss_weights": Tensor,          # focal/dice class weights
}
```

**Resume logic (best-effort, NOT bit-exact):**
1. `POST /api/v1/training/jobs/{id}/resume` → API finds the latest completed checkpoint for the failed step via `job_steps.last_checkpoint_path`
2. Worker receives resume request with explicit checkpoint path
3. Reconstruct models từ config (architecture) + load `state_dict` (weights)
4. Restore optimizer, scheduler, RNG states
5. Skip tới `checkpoint["iteration"]` trong training loop
6. Training tiếp tục từ checkpoint — **sample order and augmentations will diverge** vì `DataLoader` shuffle state và worker-process RNG không thể capture hoàn toàn (PyTorch `InfiniteDataloader` + `num_workers>0`)

> **Best-effort guarantee:** Model weights, optimizer momentum, và LR schedule được khôi phục chính xác. Data ordering sẽ khác so với nếu train liên tục — nhưng với 70k iterations, divergence này không ảnh hưởng đáng kể tới kết quả cuối. Đây là trade-off chuẩn trong production training systems.

> **Tại sao state_dict thay vì torch.save(model)?** `torch.save(model)` dùng pickle — security risk (arbitrary code execution khi load) + fragile (code path phải match). `state_dict` chỉ lưu weights, portable và an toàn.

> **RNG re-seed:** Bắt buộc seed lại SAU khi init models, TRƯỚC training loop. Khác model size → khác lượng random numbers consumed lúc init → RNG divergence → data shuffle khác → kết quả khác. (Discovered in exp002 — lightweight Comp AE)

### Library Public API

```python
from salad_pipeline import SALADPipeline, TrainingConfig
from salad_pipeline.steps.step4_training import run_salad_training
from salad_pipeline.evaluation.inference import SALADInference

# Full pipeline (4 steps tuần tự)
pipeline = SALADPipeline(
    fg_mask_config=FGMaskConfig(...),
    pseudo_label_config=PseudoLabelConfig(...),
    comp_seg_config=CompSegConfig(...),
    training_config=TrainingConfig(...),
    callback=my_platform_callback,   # optional
)
pipeline.run()

# Individual step (library API supports calling single steps; Phase 1 does NOT use this — whole pipeline via .run())
result = run_salad_training(config=TrainingConfig(...), callback=my_callback)

# Inference
inference = SALADInference.from_model_dir(model_dir=Path("..."))
score, anomaly_map = inference.predict(image)
```

> **Phase 1 execution model:** 1 Celery task = 1 job = 1 container running all 4 steps sequentially via `SALADPipeline.run()`. Per-step tracking qua `job_steps` table cho observability, retry from last completed step, và artifact tracking — NOT per-step Celery dispatch.

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
│                   ├── validation/good/...
│                   └── test/
│                       ├── good/...
│                       ├── logical_anomalies/...   ← exact SALAD folder names
│                       └── structural_anomalies/...
│
├── training-jobs/
│   └── ... (see below)
│
├── model-registry/
│   └── ... (see below)
│
└── (continued below after validation rules)
```

### Dataset Validation Rules (ISSUE-9 fix)

Upload zip PHẢI match MVTec LOCO structure. Backend validates on upload:

| Rule | Check | On Fail |
|------|-------|---------|
| `{category}/train/good/` exists | Required | `dataset.status = 'failed'`, `validation_error = "Missing train/good"` |
| `{category}/validation/good/` exists | Required | Fail |
| `{category}/test/good/` exists | Required | Fail |
| Anomaly folders match exact names | `logical_anomalies/`, `structural_anomalies/` only | Fail on unknown folder names |
| **Phase 1 SALAD: anomaly data required** | `test/logical_anomalies/` AND `test/structural_anomalies/` must exist with ≥1 image each (R3-ISSUE-4 fix: eval AUCs and best_eval promotion require anomaly samples) | Fail — dataset not usable for SALAD pipeline |
| `ground_truth/` required | **Phase 1: REQUIRED — fail validation if anomaly test/validation folders exist but no matching ground_truth subfolders** (R2-ISSUE-5 fix: ground_truth mandatory for SALAD pipeline to ensure eval AUCs + best_eval promotion work) | Fail — not warning |
| Allowed extensions | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` | Skip non-image files, log warning |
| Corrupt images | Attempt PIL.Image.open + verify | Skip corrupt, log warning, continue if ≥1 valid image per split |
| Min images per split | train/good ≥ 1 | Fail if empty |
| Single category root | Zip must contain exactly 1 category folder | Fail if 0 or >1 |

**Upload lifecycle:** `pending_upload` → `uploading` → `processing` (validation + extract) → `ready` / `failed`

**Training-jobs and model-registry storage (continued):**

```
├── training-jobs/
│   └── {job_id}/
│       ├── job.json                        ← Job metadata (immutable)
│       ├── pipeline-config.json            ← Config snapshot (immutable)
│       └── attempts/                       ← R4-ISSUE-1 fix: all mutable artifacts scoped by attempt
│           └── {attempt_id}/
│               ├── steps/
│               │   ├── 01_fg_masks/{status.json, masks/...}
│               │   ├── 02_pseudo_labels/{status.json, composition_maps/...}
│               │   ├── 03_comp_seg_training/{status.json, composition_maps/...}
│               │   └── 04_salad_training/
│               │       ├── status.json
│               │       ├── checkpoints/{iter_10000/,iter_20000/,...}
│               │       └── final/{teacher,student,autoencoder,comp_autoencoder,comp_unet}_final.pth
│               ├── logs/
│               │   ├── step_01/chunks/{offset}.jsonl   (immutable log chunks)
│               │   ├── step_01/index.json
│               │   ├── step_02/chunks/...
│               │   ├── step_03/chunks/...
│               │   └── step_04/chunks/...
│               └── metrics/
│                   ├── loss.jsonl      (1 dòng = 1 log event, every 10 iters → ~7000 entries for 70k steps)
│                   ├── eval.jsonl      (1 dòng = 1 checkpoint evaluation, every 10k iters → ~7 entries)
│                   └── summary.json    (kết quả cuối cùng)
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

**Platform Adapter (IPC approach — container has no network):** Container chạy `salad_pipeline` library với một **file-based callback** implementation. Container **không thể** truy cập Redis, DB, hay MinIO trực tiếp vì chạy `--network=none` (Section 13).

**IPC Architecture:**

```
┌── Training Container (--network=none) ─────────────┐
│  salad_pipeline + FilesystemCallback                │
│  Writes to /workspace/ipc/:                         │
│    loss.jsonl    (append per log_interval)           │
│    eval.jsonl    (append per eval_interval)          │
│    events.jsonl  (step_start, step_end, error, ...)  │
│    checkpoint/   (saved every checkpoint_interval)   │
│    should_stop   (read: if exists → graceful stop)   │
└─────────────────────────────────────────────────────┘
        ↕ /workspace mounted as shared volume
┌── Worker Daemon (host, has network) ────────────────┐
│  IPC Watcher (inotify/polling /workspace/ipc/):     │
│    loss.jsonl  → parse + publish Redis Pub/Sub      │
│    eval.jsonl  → parse + POST /api/v1/internal/...   │
│    events.jsonl → update job_steps via internal API    │
│    checkpoint/ → upload to MinIO                    │
│    should_stop → touch file when cancel received    │
└─────────────────────────────────────────────────────┘
```

**FilesystemCallback** (ships with `salad_pipeline`):
1. **`on_train_iteration()`** → append `loss.jsonl` (6 fields: `loss_total, loss_st, loss_ae, loss_stae, loss_comp_recon, loss_comp_mask`)
2. **`on_eval_end()`** → append `eval.jsonl` (12 fields — see canonical list in Section 4)
3. **`on_checkpoint_saved()`** → write checkpoint dir to `/workspace/ipc/checkpoint/{iteration}/` + `.done` marker
4. **`on_train_end()`** → save `calibration.pt` + write final model artifacts to `/workspace/ipc/final/` + `.done` marker
5. **`on_step_start/end()`** → append `events.jsonl` with structured event
6. **`should_stop()`** → check if `/workspace/ipc/should_stop` file exists

**Atomic write contract (prevents watcher from reading partial data):**
- **JSONL files** (`loss.jsonl`, `eval.jsonl`, `events.jsonl`): Append a complete line ending with `\n` via a single `write()` syscall. Watcher reads only complete lines (ignore trailing partial line).
- **Directories** (checkpoint, final): Write all files into `{dir}.tmp/`, then `os.rename("{dir}.tmp/", "{dir}/")` + `touch {dir}/.done`. Watcher ignores any directory without `.done` marker.
- **Rationale:** `os.rename()` is atomic on same filesystem (guaranteed by POSIX). JSONL single-line append is atomic up to `PIPE_BUF` (4096 bytes on Linux, each line is ~200 bytes).

**Worker Daemon IPC Watcher** (bên ngoài container):
- Watches `/workspace/ipc/` via `inotify` (Linux) hoặc polling (1s interval)
- Forward metrics → Redis Pub/Sub (cho WebSocket live streaming)
- Forward eval results → internal API (→ `training_evaluations` DB table)
- Upload checkpoints → MinIO, then **update `job_steps.last_checkpoint_path`** via internal API (only AFTER successful upload — partial/failed uploads do NOT update the column)
- Cancel: polls `/api/v1/internal/jobs/{id}/cancel-status` mỗi 2s → touch `should_stop` file khi API reports cancelled (single mechanism — no Celery signal dependency)

**Checkpoint upload lifecycle (R3-ISSUE-2 fix — attempt-scoped paths):**
1. Container writes checkpoint to `/workspace/ipc/checkpoint/{iteration}/` + renames `.done` marker (atomic)
2. Watcher detects `.done` marker → uploads entire dir to MinIO `training-jobs/{job_id}/attempts/{attempt_id}/checkpoints/{iteration}/`
3. Upload succeeds → watcher calls `PATCH /api/v1/internal/jobs/{job_id}/steps/{step_id}` with `last_checkpoint_path = "training-jobs/{job_id}/attempts/{attempt_id}/checkpoints/{iteration}/"` and `durable_checkpoint_iterations = [..., iteration]` (R3-ISSUE-3 fix)
4. Upload fails → watcher retries 3x with exponential backoff. If all retries fail → log error, do NOT update paths (resume will use previous valid checkpoint)
5. New checkpoint arrives while uploading → watcher queues it; after current upload completes, **uploads ALL pending checkpoints** (no skip — R3-ISSUE-3 fix: every checkpoint must be durably uploaded for promotion eligibility)

**Attempt artifact promotion (R3-ISSUE-2 fix):**
On successful job completion, daemon promotes artifacts from active attempt prefix → canonical paths:
- `training-jobs/{job_id}/attempts/{attempt_id}/final/` → `model-registry/{model_id}/weights/`
- `training-jobs/{job_id}/attempts/{attempt_id}/checkpoints/` → retained for debugging, cleaned by retention policy
- Stale attempt prefixes (`attempts/{old_attempt_id}/`) cleaned up by the same retention policy

> **Rationale:** Giữ container `--network=none` cho security (không thể exfiltrate data) trong khi vẫn có real-time metric streaming. Worker daemon là trusted process bên ngoài container, có network access.

> **Metrics cadence note:** Losses logged mỗi `log_interval` iterations (default 10) → ~7,000 entries cho 70k steps. Eval metrics logged mỗi `eval_interval` iterations (default 10k) → ~7 entries. Both cadences configurable via Advanced params.

**Inference Manifest Schema (`manifest.json`):**
```json
{
  "version": "1.0",
  "pipeline": "salad",
  "category": "chip_pcb",
  "job_id": "uuid",
  "created_at": "2026-04-10T12:00:00Z",
  "runtime": {
    "docker_image_digest": "sha256:abc123...",
    "pipeline_schema_version": "1.0",
    "salad_pipeline_version": "0.1.0",
    "prerequisite_checksums": {
      "teacher_medium.pth": "sha256:def456...",
      "sam_vit_h_4b8939.pth": "sha256:789abc...",
      "sam_hq_vit_h.pth": "sha256:012def..."
    },
    "dataset_version_id": "uuid",
    "dataset_fingerprint": "sha256:..."
  },
  "promotion": {
    "source": "best_eval",
    "selected_iteration": 60000,
    "selection_criterion": "auc_combined"
  },
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
| status | ENUM('pending_upload','uploading','processing','ready','failed','archived') | Upload/validation lifecycle + soft-delete |
| raw_file_name | VARCHAR | Original zip filename |
| raw_file_size | BIGINT | Bytes |
| validation_error | TEXT | Validation failure reason (NULL if valid) |
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
| runtime_manifest | JSONB | Immutable snapshot: docker_image_digest, pipeline_version, prerequisite_checksums (ISSUE-3 fix) |
| status | ENUM('queued','running','completed','failed','cancelled','cancelling') | Job-level status |
| current_step | INT | 1,2,3,4 |
| worker_id | UUID FK → workers | Nullable — assigned khi dispatch |
| idempotency_token | UUID UNIQUE | Prevent duplicate execution on redelivery |
| current_attempt_id | UUID | Fencing token — updated on every dispatch/requeue. All writes must match. |
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
| last_checkpoint_path | VARCHAR NULLABLE | MinIO path to latest valid checkpoint (Step 4 only) |
| durable_checkpoint_iterations | INT[] | List of iterations with successfully uploaded checkpoints (R3-ISSUE-3 fix — used for promotion eligibility) |
| created_at | TIMESTAMP | |

### training_metrics

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL PK | |
| job_id | UUID FK → training_jobs | |
| step | INT | Pipeline step (4 = SALAD) |
| iteration | INT | 0..70000 |
| metrics | JSONB | {"loss_total":0.023,"loss_st":0.008,...} |
| created_at | TIMESTAMP | |

Index: `(job_id, step, iteration)` **UNIQUE constraint — upsert-or-skip on conflict**

> **Why JSONB (not typed columns)?** Phase 1 SALAD has fixed 6 loss fields, but future pipelines sẽ có loss schema khác hoàn toàn. JSONB cho phép add pipeline mà không cần migration. Trade-off: chart adapter phải biết pipeline-specific schema để extract đúng fields. Mỗi pipeline đăng ký `loss_schema` trong `pipelines.default_config` (JSONB) — chart frontend đọc schema → render đúng fields. **Không giả định mọi pipeline đều có cùng loss keys.**

**Idempotent Ingestion Rules (ISSUE-4 fix):**
- `training_metrics`: UNIQUE on `(job_id, step, iteration)`. INSERT ON CONFLICT DO NOTHING — watcher retry safe.
- `training_evaluations`: UNIQUE on `(job_id, iteration)`. INSERT ON CONFLICT DO NOTHING.
- `job_steps.last_checkpoint_path`: Update only if new `iteration > current iteration` (monotonic progress). Watcher includes `iteration` in PATCH request.
- Log chunks: MinIO key includes `{offset}` → same offset = same key = PUT is idempotent.
- All internal API writes include `attempt_id` header (see fencing above) — stale attempt writes rejected with 409.

### training_evaluations

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | |
| iteration | INT | 10000, 20000, ... |
| auc_combined | FLOAT | 0.5 * (auc_combined_logical + auc_combined_structural) |
| auc_combined_logical | FLOAT | AUROC on logical anomalies vs good (all branches fused) |
| auc_combined_structural | FLOAT | AUROC on structural anomalies vs good (all branches fused) |
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

Index: `(job_id, iteration)` **UNIQUE constraint — INSERT ON CONFLICT DO NOTHING**

> **Per-branch logical/structural AUC:** Library tính nội bộ vì có per-image scores từ mỗi branch (appearance_score, mahalanobis_score, composition_score) + labels từ folder structure (good/logical_anomalies/structural_anomalies). Library compute AUROC(branch_scores[logical_images], branch_scores[good_images]) cho mỗi branch x anomaly_type, trả 12 fields qua `on_eval_end` callback. Host watcher chỉ forward kết quả → DB.

### models

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| job_id | UUID FK → training_jobs | Single lineage source (pipeline, dataset, version derive from job) |
| owner_id | UUID FK → users | Denormalized from job.owner_id — list/filter without join |
| pipeline_id | UUID FK → pipelines | Denormalized from job.pipeline_id — filter by pipeline type |
| version_id | UUID FK → dataset_versions | Denormalized from job.version_id — trace data lineage |
| category | VARCHAR | Denormalized for quick query |
| auc_combined | FLOAT | |
| auc_appearance | FLOAT | |
| auc_mahalanobis | FLOAT | |
| auc_composition | FLOAT | |
| minio_path | VARCHAR | "model-registry/{id}/" |
| has_calibration | BOOLEAN | True if calibration artifacts exist |
| status | ENUM('active','archived','exported') | |
| source_iteration | INT | Which checkpoint iteration these weights came from |
| promotion_strategy | ENUM('best_eval','final','manual') | How this model was selected from training checkpoints |
| created_at | TIMESTAMP | |
| notes | TEXT | |

**Promotion eligibility (R3-ISSUE-3 fix):**
- An iteration is promotion-eligible ONLY if: (1) `training_evaluations` row exists for that iteration, AND (2) that iteration is in `job_steps.durable_checkpoint_iterations` array (i.e. checkpoint was successfully uploaded to MinIO)
- `eval_interval == checkpoint_interval` constraint ensures every eval iteration has a matching checkpoint attempt
- Watcher uploads ALL checkpoints (no skip) to ensure `durable_checkpoint_iterations` is complete
- Phase 1 `best_eval` strategy: select iteration with highest `auc_combined` among promotion-eligible iterations

**Chọn weights nào đưa vào model registry (Promotion — Phase 1):**

Training chạy xong → hệ thống phải chọn 1 bộ weights để promote vào `model-registry/`. Logic chọn theo thứ tự ưu tiên:

```
Training xong
  ├─ Có checkpoint nào vừa uploaded MinIO thành công, vừa có eval AUC?
  │   → YES: Chọn checkpoint có auc_combined cao nhất (strategy = best_eval)
  │
  ├─ Không có checkpoint nào eligible (MinIO upload fail hết, hoặc crash trước eval đầu tiên)?
  │   → Dùng final weights từ on_train_end() (strategy = final)
  │     (final weights luôn có vì watcher upload /workspace/ipc/final/ là bước cuối)
  │
  └─ Container crash sớm, chưa produce weights nào?
      → Job = failed, KHÔNG tạo model. Không fallback.
```

> **Edge case:** Nếu `train_steps` không chia hết cho `eval_interval` (vd: 65000 steps, eval mỗi 10000) → iteration cuối (65000) không có AUC score → `best_eval` bỏ qua nó. Nhưng `final` strategy vẫn dùng được weights cuối này.

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
| token_hash | VARCHAR | bcrypt hash of X-Worker-Token (ISSUE-8 fix) |
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
| DELETE | /api/v1/datasets/{id} | **Archive** dataset (NOT physical delete). Sets `datasets.status = 'archived'`, hides from default list. MinIO data retained. Returns 409 if active jobs or models reference this dataset. Reversible via `POST /datasets/{id}/unarchive` (Phase 2). |
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

### Internal Worker Daemon Endpoints

> **Auth model:** Worker daemon authenticates via `X-Worker-Token` header (pre-shared secret per worker, stored hashed in `workers.token_hash`). These endpoints are NOT exposed to UI/external clients — only accessible from GPU server network.

> **Authorization binding (ISSUE-8 fix):** All internal write endpoints validate:
> 1. `X-Worker-Token` → identify worker
> 2. `X-Attempt-Id` → must match `training_jobs.current_attempt_id`
> 3. Worker must be the current leaseholder (`training_jobs.worker_id = authenticated_worker.id`)
> 4. Reject with 403 if any check fails

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/internal/workers/register | Worker daemon bootstrap — registers with API, receives initial config + assigned GPU IDs. Auth: `X-Worker-Token`. Response: `{worker_id, gpu_ids, queue_name, concurrency, docker_defaults}` |
| GET | /api/v1/internal/workers/{name}/config | Poll for config updates (daemon polls every 60s). Returns latest admin-configured settings. Auth: `X-Worker-Token` |

**Worker re-registration semantics:**
- `POST /register` is **idempotent by worker name**: nếu `workers.name` đã tồn tại VÀ `X-Worker-Token` match `workers.token_hash` → update `last_heartbeat` + return current config (treat as reconnect after daemon restart)
- Name exists nhưng token mismatch → **reject 403** (prevents impersonation)
- Name exists + token match + GPU config khác admin config → return admin config (daemon phải reconcile, không override admin intent)
- Name not exists + valid token format → **reject 404** (worker must be pre-registered by admin via `POST /api/v1/admin/workers` trước, daemon chỉ bootstrap — không tự-register)
| PATCH | /api/v1/internal/jobs/{id}/steps/{step_id} | Update step status, timing, `last_checkpoint_path`, `durable_checkpoint_iterations` |
| POST | /api/v1/internal/jobs/{id}/steps/{step_id}/metrics | Bulk insert loss data (watcher batches ~100 lines) |
| POST | /api/v1/internal/jobs/{id}/steps/{step_id}/evaluations | Insert eval results (12 AUC fields per checkpoint) |
| POST | /api/v1/internal/jobs/{id}/steps/{step_id}/logs | Upload log chunk (JSONL lines) |
| PATCH | /api/v1/internal/jobs/{id} | Update job status (running → completed/failed) |
| GET | /api/v1/internal/jobs/{id}/cancel-status | Check if job has been cancelled (watcher polls this) |

**Flow: IPC Watcher → Internal API:**
1. Watcher reads `loss.jsonl` new lines → batch POST to `/internal/.../metrics` every 5s
2. Watcher reads `eval.jsonl` new line → POST to `/internal/.../evaluations`
3. Watcher detects checkpoint `.done` → upload to MinIO → PATCH `/internal/.../steps/{step_id}` with `last_checkpoint_path`
4. Watcher polls `/internal/.../cancel-status` every 2s → touch `should_stop` file on cancel
5. Container exits → watcher uploads final artifacts → PATCH job status to completed/failed

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
      "imagenet_path": null
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
├── salad_pipeline/                 # SALAD training library (rewrite from scratch)
│   ├── __init__.py                 # Public API: SALADPipeline, configs, inference
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py                 # Pydantic BaseSettings
│   │   ├── step1_fg_masks.py       # FGMaskConfig
│   │   ├── step2_pseudo_labels.py  # PseudoLabelConfig
│   │   ├── step3_comp_unet.py      # CompSegConfig
│   │   ├── step4_training.py       # TrainingConfig (main)
│   │   └── inference.py            # InferenceConfig
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pdn.py                  # PDN Small/Medium factory
│   │   ├── autoencoder.py          # Appearance AE
│   │   ├── comp_ae.py              # Composition AE (128→6ch)
│   │   ├── comp_unet.py            # Composition UNet (12→1ch)
│   │   ├── unet_blocks.py          # Shared: ResidualBlock, Attention, Up/Down
│   │   └── dino.py                 # DINO ViT-B/8 featurizer
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Unified dataset classes
│   │   ├── augmentation.py         # Anomaly synthesis (perlin, label swap)
│   │   └── transforms.py           # Standard transforms
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py                 # DiceLoss
│   │   └── focal.py                # Focal loss wrapper
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── step1_fg_masks.py       # run_fg_mask_generation()
│   │   ├── step2_pseudo_labels.py  # run_pseudo_label_creation()
│   │   ├── step3_comp_seg.py       # run_comp_seg_training()
│   │   └── step4_training.py       # run_salad_training() — main
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── calibration.py          # Compute & save calibration.pt
│   │   ├── metrics.py              # AUC (per-branch × per-type)
│   │   └── inference.py            # 3-branch inference pipeline
│   ├── callbacks.py                # TrainingCallback Protocol
│   ├── checkpoint.py               # Save/load full state (models+opt+RNG)
│   └── pipeline.py                 # SALADPipeline orchestrator
│
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
│       ├── Dockerfile              # includes salad_pipeline + all deps
│       └── entrypoint.sh           # invokes salad_pipeline with platform callback
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

### Timing Contracts (consolidated)

Tất cả polling/flush intervals gom lại 1 chỗ để team implement không lệch:

| Contract | Value | Owner | Configurable? |
|----------|-------|-------|---------------|
| `HEARTBEAT_INTERVAL` | 30s | Worker daemon → API | Env var |
| `OFFLINE_THRESHOLD` | 90s (3 missed heartbeats) | API health checker | Env var |
| `HEALTH_CHECK_TIMEOUT` | 10s | API → daemon health endpoint | Env var |
| `CONFIG_POLL_INTERVAL` | 60s | Worker daemon → `GET /internal/workers/{name}/config` | Env var |
| `CANCEL_POLL_INTERVAL` | 2s | IPC watcher → `GET /internal/.../cancel-status` | Env var |
| `LOG_FLUSH_INTERVAL` | 1s hoặc 100 lines (whichever first) | Worker daemon log buffer | Hardcoded |
| `METRICS_BATCH_INTERVAL` | 5s | IPC watcher → `POST /internal/.../metrics` | Env var |
| `IPC_WATCH_INTERVAL` | 1s (polling fallback khi không có inotify) | IPC watcher | Hardcoded |
| `CHECKPOINT_UPLOAD_RETRY` | 3x, exponential backoff (1s, 2s, 4s) | IPC watcher → MinIO | Hardcoded |
| `CANCEL_GRACE_PERIOD` | 30s SIGTERM → SIGKILL | Worker daemon → container | Env var |
| `CELERY_VISIBILITY_TIMEOUT` | 86400s (24h) | Celery broker config | Config file |

> **Rule:** Tất cả env-configurable intervals đều có sane defaults (values trên). Team KHÔNG được hardcode các giá trị khác với bảng này trừ khi update spec trước.

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

**Write path (R4-ISSUE-1 fix — attempt-scoped paths):**
1. Container stdout → worker daemon tee vào local buffer
2. Buffer flush mỗi 1s hoặc 100 lines (whichever first):
   - Publish batch tới Redis channel `job:{job_id}:attempt:{attempt_id}:step:{step}:logs`
   - Write immutable chunk tới MinIO: `training-jobs/{job_id}/attempts/{attempt_id}/logs/step_{N}/chunks/{offset}.jsonl`
   - Each chunk file: `[{offset, timestamp, line}, ...]`
3. Chunk index file updated: `training-jobs/{job_id}/attempts/{attempt_id}/logs/step_{N}/index.json` — list of chunk offsets + sizes

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
2. **DINO ViT-B/8:** `salad_pipeline.steps.step2_pseudo_labels` loads DINO via `torch.hub.load_state_dict_from_url()` internally. Container image phải bake weights vào `TORCH_HOME=/cache/torch/hub/` (set env var trong Dockerfile). Library sẽ tìm cached weights trước khi cố download → hoạt động với `--network=none`.
3. **SAM / SAM-HQ:** Weights từ MinIO, không download runtime.
4. **Python packages:** Tất cả pip packages bake vào Docker image. Không `pip install` lúc runtime.
5. **Focal loss:** `kornia.losses.FocalLoss` hoặc custom implementation — bake vào image.

**Docker image build contract:**
```dockerfile
# salad-training:latest must include:
# - salad_pipeline package (pip install -e ./salad_pipeline)
# - All pip packages (torch, torchvision, timm, segment-anything-hq, etc.)
# - DINO ViT-B/8 weights baked in TORCH_HOME=/cache/torch/hub/
#   (download during build: torch.hub.load_state_dict_from_url(DINO_URL))
# - Python 3.11
# - CUDA 12.4 runtime
# - ENV TORCH_HOME=/cache/torch/hub
# NO network access during training (--network=none)
```

**Verification:** CI pipeline build image + run smoke test (dry-run 10 iterations) without network access.

---

## 13. Security

### Authentication & Authorization

| Scope | Rule |
|-------|------|
| JWT tokens | Access token (15m) + Refresh token (7d). HttpOnly cookies, SameSite=Strict. |
| Datasets | Owner-only: user chỉ thấy/sửa datasets mình tạo. Delete = soft-delete (archived), blocked if dependent jobs/models exist. Admin thấy all. |
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
- Worker daemon authenticates to API with **`X-Worker-Token` header** (pre-shared secret per worker, stored hashed in `workers` table). Used for all `/api/v1/internal/*` endpoints. Token rotated monthly, revoked on worker deregistration. No JWT needed — worker daemon only calls internal endpoints, not user-facing APIs.
