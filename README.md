# Personalized Virtual Dermatology Health Advisor (Local ADK + Kaggle)

This repo scaffolds a **virtual dermatology advisor agent** using **Google ADK (Agent Development Kit)** backed by
a **local Ollama-hosted model** plus a reproducible **deep-learning training/evaluation pipeline** for dermatology
image datasets (e.g., HAM10000 / ISIC-style).

## What you get

- **ADK agent** (shared definition in `src/derm_advisor/adk_agent.py`, re-exported from `agents/derm_advisor_agent/agent.py` for `adk run` / `adk web`)
  - Tool-driven image classification (`classify_lesion(image_path)`)
  - Safety-guarded, non-prescriptive triage tool (`safety_triage`)
  - Local LLM backend via Ollama (`qwen3:8b` by default, with Qwen thinking disabled to keep CPU-only local chat responsive)
- **Training pipeline** (PyTorch + timm)
  - Fine-tune a strong pretrained backbone (default: ConvNeXt Tiny IN22K)
  - Outputs `reports/metrics.json` and plots (`reports/loss_curve.png`, `reports/val_accuracy_curve.png`)
- **Streamlit demo UI** (`apps/streamlit_inference.py`)
  - **Lesion classifier** tab (upload + probabilities)
  - **Derm advisor (ADK chat)** tab — uses the same ADK `Runner` + local Ollama model as `adk run`, so you get a full chat UI without the CLI (single browser app)

## Setup

Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Start the local model runtime

Container-first setup (validated locally with Podman, but the compose file also works with Docker):

```bash
podman compose -f compose.local-model.yml up -d
podman compose -f compose.local-model.yml exec ollama ollama pull qwen3:8b
podman compose -f compose.local-model.yml exec ollama ollama show qwen3:8b
```

The final command should list `tools` under capabilities. The app defaults to `http://localhost:11434` and can be
overridden with `OLLAMA_API_BASE`. For Qwen3, the runtime also defaults `DERM_ADVISOR_DISABLE_THINKING=true` so local
CPU inference returns practical answers instead of long reasoning traces.

Copy the sample environment file:

```bash
cp agents/derm_advisor_agent/.env.example agents/derm_advisor_agent/.env
```

## Get dermatology data (Kaggle HAM10000 example)

1) Configure Kaggle credentials:

- Download your Kaggle API token from Kaggle → Account → API.
- Put it at `~/.kaggle/access_token` and set permissions:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/access_token
```

2) Download the dataset:

```bash
python scripts/download_kaggle_dataset.py \
  --dataset "kmader/skin-cancer-mnist-ham10000" \
  --out data/kaggle_ham10000
```

3) Prepare ImageFolder splits (train/val/test):

Point `--ham-root` at the folder containing `HAM10000_metadata.csv` and the images.
If the Kaggle download unzips into nested folders, use the nested folder as `--ham-root`.

```bash
python scripts/prepare_ham10000_imagefolder.py \
  --ham-root data/kaggle_ham10000 \
  --out data/ham10000_imagefolder
```

After this, you should have:

```text
data/ham10000_imagefolder/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Train + generate accuracy visualizations

```bash
python scripts/train_vision_model.py --dataset-root data/ham10000_imagefolder --epochs 5 --device mps
```

Artifacts:
- `artifacts/best_model.pt`
- `reports/metrics.json`
- `reports/loss_curve.png`
- `reports/val_accuracy_curve.png`

## Run the Streamlit demo

```bash
streamlit run apps/streamlit_inference.py
```

The advisor chat expects the local Ollama runtime above to be available. In the **Derm advisor** tab you can chat with
the agent; after you upload an image in the sidebar, use **Get Advisor Analysis** so the agent receives the local file
path for `classify_lesion`.

You can still run **`adk web`** separately for debugging (another port). For day-to-day use, the Streamlit app is intended to replace needing the CLI.

## Run the ADK agent

1) Copy env file and keep the default local-model settings:

```bash
cp agents/derm_advisor_agent/.env.example agents/derm_advisor_agent/.env
```

2) Run the agent from the `agents/` parent directory:

```bash
cd agents
adk run derm_advisor_agent
```

In chat, provide a local image path and ask for a safety-guarded summary. The agent will use `classify_lesion`.
The Python agent definition in `agents/derm_advisor_agent/agent.py` is the canonical entrypoint for the local-model
setup.

## Safety / intended use

This is an educational prototype:
- It **does not diagnose** medical conditions.
- It **does not prescribe** treatments.
- For concerning lesions (ABCDE changes, rapid growth, bleeding, pain) seek professional care.
