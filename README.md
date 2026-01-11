# Text-to-Face Generation Demo

Generate face images from text descriptions using BERT + StyleGAN2.

Training repo: https://github.com/thangthewinner/t2f_training 

## Requirements

- Python 3.12.9 (for local setup)
- Docker (for Docker setup)
- 4GB VRAM (optional, for GPU acceleration)

## API Key

For AI text formatting feature, create `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

Get free API key at: https://console.groq.com/keys

## Quick Start

### Option 1: Smart Launcher

```bash
./run.sh
```

That's it! Access at: **http://localhost:7860**

### Option 2: Docker Compose

```bash
# For CPU
docker compose --profile cpu up

# For GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```

### Option 3: Manual Docker

**CPU Version (lighter):**
```bash
# Build
docker build -t t2f-demo:cpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .

# Run
docker run -p 7860:7860 -v $(pwd)/models:/app/models t2f-demo:cpu
```

**GPU Version (faster, requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):**
```bash
# Build
docker build -t t2f-demo:gpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .

# Run
docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models t2f-demo:gpu
```

### Option 4: Local Python Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone StyleGAN2
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

# 3. Download models (see below)

# 4. Run
python app.py
```

## Required Models

Place these files in `models/` directory:

1. **checkpoint_epoch0500.pt** (629 MB)
   - Download: [Google Drive](https://drive.google.com/file/d/1FHynclzxW_KTxkekz1pMGTUludYIkKhv/view?usp=drive_link)

2. **ffhq.pkl** (350 MB)
   - Auto-downloaded by `run.sh`, or:
   ```bash
   wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P models/
   ```

## Demo

![T2F Demo Interface](images/demo.png)