#!/bin/bash
# Brandenburg Gate 图像数据集过滤脚本
# 使用 Ollama VLM 模型评估图像质量，过滤出高质量图片

set -e

# ================== 配置参数 ==================
# 数据集路径
DATASET_DIR="/home/wangyz/data/PT/brandenburg_gate"
INPUT_TSV="${DATASET_DIR}/brandenburg.tsv"
OUTPUT_TSV="${DATASET_DIR}/brandenburg_filtered_v5.tsv"
IMAGE_DIR="${DATASET_DIR}/dense/images"

# Ollama 服务配置
OLLAMA_HOST="localhost"
OLLAMA_PORT="11434"
MODEL_NAME="llava:34b"  # 使用更强的模型

# 地标名称
LANDMARK_NAME="Brandenburg Gate"

# 断点续传（如果需要从某一行继续，修改这里）
RESUME_FROM=0

# ==============================================

echo "========================================"
echo "Brandenburg Gate 图像过滤"
echo "========================================"
echo "输入文件: ${INPUT_TSV}"
echo "输出文件: ${OUTPUT_TSV}"
echo "图像目录: ${IMAGE_DIR}"
echo "Ollama 服务: ${OLLAMA_HOST}:${OLLAMA_PORT}"
echo "模型: ${MODEL_NAME}"
echo "========================================"
echo ""

# 检查输入文件是否存在
if [ ! -f "${INPUT_TSV}" ]; then
    echo "错误：输入文件不存在: ${INPUT_TSV}"
    exit 1
fi

# 检查图像目录是否存在
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "错误：图像目录不存在: ${IMAGE_DIR}"
    exit 1
fi

# 测试 Ollama 连接
echo "测试 Ollama 服务连接..."
if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "✓ Ollama 服务连接成功"
else
    echo "✗ 无法连接到 Ollama 服务: http://${OLLAMA_HOST}:${OLLAMA_PORT}"
    echo "请确保 Ollama 服务正在运行"
    exit 1
fi

# 检查模型是否可用
echo "检查模型 ${MODEL_NAME} 是否可用..."
if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" | grep -q "${MODEL_NAME}"; then
    echo "✓ 模型 ${MODEL_NAME} 可用"
else
    echo "警告：模型 ${MODEL_NAME} 可能未安装"
    echo "继续执行，如果模型不存在将会报错"
fi

echo ""
echo "开始过滤图像..."
echo "========================================"

# 运行过滤脚本
python filter_images_with_vlm.py \
    --tsv_path "${INPUT_TSV}" \
    --output_path "${OUTPUT_TSV}" \
    --image_dir "${IMAGE_DIR}" \
    --ollama_host "${OLLAMA_HOST}" \
    --ollama_port "${OLLAMA_PORT}" \
    --model_name "${MODEL_NAME}" \
    --landmark_name "${LANDMARK_NAME}" \
    --resume_from "${RESUME_FROM}"

echo ""
echo "========================================"
echo "过滤完成！"
echo "结果已保存到: ${OUTPUT_TSV}"
echo "========================================"
