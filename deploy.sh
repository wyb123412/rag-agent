#!/usr/bin/env bash
# 一键部署脚本：在全新 Ubuntu 22.04/24.04 服务器上以 root 运行。
#
# 用法（推荐）：
#   git clone https://github.com/wyb123412/rag-agent.git /opt/rag-agent
#   cd /opt/rag-agent
#   cp .env.example .env && nano .env   # 粘贴你的 API Key（公网务必设 GRADIO_USERNAME/GRADIO_PASSWORD）
#   bash deploy.sh
#
# 前置：必须先创建好 .env，脚本检测不到会中止。可重复运行（幂等）。
set -euo pipefail

REPO_URL="https://github.com/wyb123412/rag-agent.git"

# 定位项目目录：当前目录已是仓库就用当前目录，否则 clone 到 /opt/rag-agent
if [ -f "docker-compose.yml" ] && [ -f "app.py" ]; then
    APP_DIR="$(pwd)"
else
    APP_DIR="/opt/rag-agent"
    if [ -d "$APP_DIR/.git" ]; then
        git -C "$APP_DIR" pull --ff-only
    else
        git clone "$REPO_URL" "$APP_DIR"
    fi
fi
cd "$APP_DIR"

echo "==> [1/5] 安装 Docker（已装则跳过）"
if ! command -v docker >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y docker.io docker-compose-plugin
    systemctl enable --now docker
fi
docker --version
docker compose version

echo "==> [2/5] 检查 .env"
if [ ! -f .env ]; then
    echo "    ❌ 未找到 $APP_DIR/.env —— 请先执行："
    echo "       cp .env.example .env && nano .env"
    echo "       至少填 LLM_API_KEY 和 EMBED_API_KEY；公网务必填 GRADIO_USERNAME/GRADIO_PASSWORD"
    exit 1
fi
chmod 600 .env

echo "==> [3/5] 目录权限（容器以 uid 1000 运行）"
mkdir -p chroma_db data
chown -R 1000:1000 chroma_db data

echo "==> [4/5] 构建并启动"
docker compose up -d --build
docker compose ps

echo "==> [5/5] 入库示例文档（让 demo 开箱即用）"
docker compose exec -T rag-agent python ingest.py \
    || echo "    ⚠️ 入库失败：请检查 .env 里的 key/余额，之后重试：docker compose exec rag-agent python ingest.py"

echo ""
echo "✅ 部署完成！"
echo "    本机自检：curl -f http://127.0.0.1:7860/ && echo OK"
echo "    公网访问：http://你的服务器公网IP:7860"
echo "    （云控制台安全组需放行 7860；若设了 GRADIO 登录，会要求用户名/密码）"
echo "    可选图谱：docker compose exec rag-agent python graph_build.py"
