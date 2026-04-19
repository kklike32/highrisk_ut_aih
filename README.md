# Personalized Virtual Dermatology Health Advisor (ADK + Kaggle)

This repo scaffolds a **virtual dermatology advisor agent** using **Google ADK (Agent Development Kit)** plus a
reproducible **deep-learning training/evaluation pipeline** for dermatology image datasets (e.g., HAM10000 / ISIC-style).

## What you get

- **ADK agent** (`agents/derm_advisor_agent/agent.py`)
  - Tool-driven image classification (`classify_lesion(image_path)`)
  - Safety-guarded, non-prescriptive triage tool (`safety_triage`)
- **Training pipeline** (PyTorch + timm)
  - Fine-tune a strong pretrained backbone (default: ConvNeXt Tiny IN22K)
  - Outputs `reports/metrics.json` and plots (`reports/loss_curve.png`, `reports/val_accuracy_curve.png`)
- **Streamlit demo UI** (`apps/streamlit_inference.py`) for uploading an image and seeing predictions

## Setup

Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Get dermatology data (Kaggle HAM10000 example)

1) Configure Kaggle credentials:

- Download your Kaggle API token (`kaggle.json`) from Kaggle → Account → API.
- Put it at `~/.kaggle/kaggle.json` and set permissions:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
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

## Run the ADK agent

1) Copy env file and add your Gemini API key:

```bash
cp agents/derm_advisor_agent/.env.example agents/derm_advisor_agent/.env
```

2) Run the agent from the `agents/` parent directory:

```bash
cd agents
adk run derm_advisor_agent
```

In chat, provide a local image path and ask for a safety-guarded summary. The agent will use `classify_lesion`.

## Safety / intended use

This is an educational prototype:
- It **does not diagnose** medical conditions.
- It **does not prescribe** treatments.
- For concerning lesions (ABCDE changes, rapid growth, bleeding, pain) seek professional care.

