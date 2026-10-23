# 部署指南

项目已 Docker 化：`Dockerfile` + `docker-compose.yml`。服务默认监听 `0.0.0.0:7860`（Gradio 网页）。

## 本地 Docker 起服务

```bash
# 1. 确认 .env 已填好 key（不含 key 的话网页问答会报错）
ls .env

# 2. 构建并后台启动
docker compose up -d --build

# 3. 打开 http://localhost:7860
```

查看日志 / 停止：

```bash
docker compose logs -f
docker compose down
```

## 数据持久化

`docker-compose.yml` 里挂载了两个目录，删容器也不会丢数据：

- `./chroma_db` —— 向量库（入库结果）
- `./data` —— 文档与知识图谱（`data/samples/` 源文档、`data/graph.json`）

## 更新知识库

把新文档放进 `data/samples/` 后重新入库即可：

```bash
docker compose exec rag-agent python ingest.py
docker compose exec rag-agent python graph_build.py   # 需要图谱时
docker compose restart rag-agent
```

（也可以在网页「上传文档」Tab 上传，会自动触发入库。）

## 部署到云服务器

以一台装了 Docker 的 Linux 服务器为例：

```bash
# 1. 把项目传上去（或 git clone），并在项目根目录放好 .env
# 2. 启动
docker compose up -d --build
```

此时服务只在本机 `7860` 端口。对外用域名 + HTTPS 建议加一层反向代理（Caddy 最省事）：

**Caddyfile 示例**（`caddy.example.com` 换成你的域名，反向代理到本机 7860）：

```
your-domain.com {
    reverse_proxy 127.0.0.1:7860
}
```

用 Caddy 跑：

```bash
# Caddy 会自动申请并续期 HTTPS 证书
caddy run --config Caddyfile
```

> 注意：`.env` 里有真实 API key，**不要**提交进 git 或打进镜像（`.gitignore` 已忽略、`docker-compose.yml` 用 `env_file` 注入而非 COPY 进镜像）。

## 健康检查

Gradio 根路径返回 200，可用：

```bash
curl -f http://localhost:7860/ && echo OK
```
