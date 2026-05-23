#!/bin/bash
# AutoDL 一键部署脚本
# 使用方法: 在 AutoDL 终端中运行 bash setup_autodl.sh

set -e

echo "=== Alpha EAI AutoDL 部署 ==="

# 1. 安装依赖
echo "[1/3] 安装依赖..."
pip install datasets transformers accelerate -q

# 2. 设置 HF 镜像
echo "[2/3] 设置 HF 镜像..."
export HF_ENDPOINT=https://hf-mirror.com

# 3. 检查 GPU
echo "[3/3] GPU 信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== 部署完成 ==="
echo ""
echo "运行训练:"
echo "  python3 run_comparison.py --epochs 5 --batch-size 16"
echo ""
echo "运行 TinyStories 训练:"
echo "  python3 run_mvp.py --use-tiny-stories --epochs 5 --batch-size 16"
