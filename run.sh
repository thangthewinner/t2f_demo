#!/bin/bash

# Text-to-Face Demo - Smart Docker Runner
# Automatically detects GPU availability and runs appropriate version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Text-to-Face Demo - Smart Launcher${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if models directory exists
if [ ! -d "models" ]; then
    echo -e "${RED} Error: models/ directory not found${NC}"
    echo -e "${YELLOW}Please create models/ directory and download required files:${NC}"
    echo -e "  1. checkpoint_epoch0500.pt (629 MB)"
    echo -e "  2. ffhq.pkl (350 MB)"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "models/checkpoint_epoch0500.pt" ]; then
    echo -e "${YELLOW}  Warning: checkpoint_epoch0500.pt not found in models/${NC}"
    echo -e "Download from: https://drive.google.com/file/d/1FHynclzxW_KTxkekz1pMGTUludYIkKhv/view"
fi

# Check if ffhq.pkl exists
if [ ! -f "models/ffhq.pkl" ]; then
    echo -e "${YELLOW}  Warning: ffhq.pkl not found in models/${NC}"
    echo -e "${YELLOW}Downloading ffhq.pkl (350 MB)...${NC}"
    
    if command -v wget &> /dev/null; then
        wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P models/
    elif command -v curl &> /dev/null; then
        curl -L https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -o models/ffhq.pkl
    else
        echo -e "${RED} Error: wget or curl required to download ffhq.pkl${NC}"
        echo -e "Please download manually from:"
        echo -e "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        exit 1
    fi
    echo -e "${GREEN} Downloaded ffhq.pkl${NC}\n"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}  .env file not found. Creating empty .env...${NC}"
    touch .env
    echo -e "${YELLOW}Note: GROQ API key is optional. Add it to .env for AI text formatting.${NC}\n"
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED} Error: Docker is not running${NC}"
    echo -e "Please start Docker and try again."
    exit 1
fi

# Check for GPU support
echo -e "${BLUE}Checking GPU availability...${NC}"

GPU_AVAILABLE=false
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
    echo -e "${GREEN} GPU detected! NVIDIA Container Toolkit is available.${NC}"
    echo -e "${GREEN}   Running with GPU acceleration...${NC}\n"
else
    echo -e "${YELLOW}  GPU not available or NVIDIA Container Toolkit not installed.${NC}"
    echo -e "${YELLOW}   Running with CPU (slower but works everywhere)...${NC}\n"
fi

# Check if image exists, if not build it
if [ "$GPU_AVAILABLE" = true ]; then
    IMAGE_NAME="t2f-demo:gpu"
    if ! docker image inspect $IMAGE_NAME &> /dev/null; then
        echo -e "${BLUE}Building GPU version (first time only)...${NC}"
        docker build -t $IMAGE_NAME \
            --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
    fi
    
    # Run with GPU
    echo -e "${GREEN} Starting Text-to-Face Demo with GPU...${NC}"
    docker run --rm --gpus all \
        -p 7860:7860 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/.env:/app/.env" \
        --name t2f-demo \
        $IMAGE_NAME
else
    IMAGE_NAME="t2f-demo:cpu"
    if ! docker image inspect $IMAGE_NAME &> /dev/null; then
        echo -e "${BLUE}Building CPU version (first time only)...${NC}"
        docker build -t $IMAGE_NAME \
            --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
    fi
    
    # Run with CPU
    echo -e "${GREEN} Starting Text-to-Face Demo with CPU...${NC}"
    docker run --rm \
        -p 7860:7860 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/.env:/app/.env" \
        --name t2f-demo \
        $IMAGE_NAME
fi

echo -e "\n${GREEN}Access the demo at: ${BLUE}http://localhost:7860${NC}"

