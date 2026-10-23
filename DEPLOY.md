# 部署指南（云端完整版）

> 目标：在一台有公网 IP 的 Linux 服务器上，用 Docker 把本 RAG Agent 跑起来。
> 全程只需要在服务器上装 Docker，**服务器上不用装 Python**（Docker 镜像里自带）。

---

## 最快方式：一键脚本（推荐，开箱即用的 demo）

买好服务器、放行端口后，`ssh root@服务器IP` 登录，跑这 4 行：

```bash
git clone https://github.com/wyb123412/rag-agent.git /opt/rag-agent
cd /opt/rag-agent
cp .env.example .env && nano .env   # 粘贴 API Key；公网务必填 GRADIO_USERNAME / GRADIO_PASSWORD
bash deploy.sh
```

`deploy.sh` 会自动：装 Docker → 检查 .env → 修目录权限 → `docker compose up -d --build` → 入库示例文档。跑完直接访问 `http://服务器IP:7860`。

> 用原始 `IP:7860` 做 demo 无需域名、无需 ICP 备案，最快上线。想要 `https://域名` 再走下面的「域名 + HTTPS」小节。

---

## 0. 选服务器

| 项 | 建议 |
|---|---|
| 云厂商 | 阿里云 / 腾讯云 轻量应用服务器 |
| 配置 | **2 核 2G 起步**（向量化在 API 侧，容器里只有 Chroma + 应用，2G 够用） |
| 地域 | **中国大陆** |
| 系统 | Ubuntu 22.04 / 24.04 LTS |

> 为什么选大陆地域：本项目运行时只调 **DeepSeek（api.deepseek.com）** 和 **硅基流动（api.siliconflow.cn）**，都是国内 API，大陆服务器直连最快最稳；反过来海外服务器连硅基流动反而可能不稳。运行时**不依赖 HuggingFace**，所以不怕 HF 被墙（HF 只影响跑 BEIR 评测时的数据下载，与线上服务无关）。

---

## 1. 登录服务器、装 Docker

```bash
ssh root@你的服务器IP

# 用 Ubuntu 官方源装（推荐，可控）
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker

docker --version
docker compose version   # 出现版本号即成功（新版是 docker compose，不是 docker-compose）
```

> 也可以 `curl -fsSL https://get.docker.com | sh` 一键装，但国内网络有时会卡，apt 源更稳。

---

## 2. 安全组 / 防火墙：只开 22、80、443

- **云控制台「安全组」**放行：`22`（SSH）、`80`、`443`。
- **不要放行 `7860`**——Gradio 端口只给本机反向代理用，绝不能直接对公网。

服务器本机若开了 ufw：

```bash
sudo ufw allow 22 && sudo ufw allow 80 && sudo ufw allow 443 && sudo ufw enable
```

---

## 3. 上传代码（两种方式）

**方式 A：git（推荐）**

```bash
# 先在本地把项目 push 到 GitHub【私有仓库】，再在服务器上：
git clone https://github.com/你的用户名/rag-agent.git
cd rag-agent
```

**方式 B：scp 上传**（不上 GitHub 的话）

```bash
# 在本地执行，排除 venv/.git/chroma_db/data/beir 这些大目录
rsync -av --exclude 'venv' --exclude '.git' --exclude 'chroma_db' \
  --exclude 'data/beir' --exclude '__pycache__' \
  ./ root@你的服务器IP:/root/rag-agent/
```

> `.env` 一定不要用这两种方式传上去——它被 gitignore、也被 rsync 手动排除，服务器上**手动新建**（见下一步）。

---

## 4. 服务器上手动放好 .env

```bash
cd rag-agent
nano .env        # 粘贴你本地 .env 里的内容（LLM_API_KEY、EMBED_API_KEY 等）
chmod 600 .env   # 只给当前用户可读
```

**三个硬性要求：**
1. `.env` 只在服务器上手动创建，**绝不 git 提交、绝不 COPY 进镜像**（`.gitignore` 已忽略、`.dockerignore` 已排除）。
2. 少了 `.env` 这一步，`docker compose` 会直接报错 `env file not found`。
3. 仓库设**私有**：里面有个人知识库文档和评测集，虽无 key，但仍是个人数据。
4. **公网部署务必在 `.env` 里同时设置 `GRADIO_USERNAME` / `GRADIO_PASSWORD`**，开启网页登录；不设则任何人拿到网址都能用你的知识库、烧你的 API 额度。

---

## 5. 构建并启动

```bash
# 容器以非 root 用户（uid 1000）运行，首次需让挂载目录可写（否则向量库/上传会写不进去）
sudo chown -R 1000:1000 chroma_db data

docker compose up -d --build
docker compose ps          # STATUS 应为 Up（healthy）
docker compose logs -f     # 看到「服务已启动」即成功；Ctrl+C 退出日志
```

---

## 6. 先用 SSH 隧道验证（不暴露端口）

在**你自己电脑**上执行，把服务器 7860 转发到本地：

```bash
ssh -L 7860:127.0.0.1:7860 root@你的服务器IP
# 浏览器打开 http://localhost:7860，确认能问答
```

先确认网页能开、能正常问答，再接域名。

---

## 7. 域名 + HTTPS 反向代理（Caddy 最省事）

1. 买域名，加一条 **A 记录** 指向服务器 IP，等解析生效。
2. 装 Caddy：

```bash
sudo apt install -y caddy
sudo nano /etc/caddy/Caddyfile
```

Caddyfile 内容（换成你的域名）：

```
你的域名.com {
    reverse_proxy 127.0.0.1:7860
}
```

3. 重载，Caddy 自动申请并续期 Let's Encrypt 证书：

```bash
sudo systemctl reload caddy
```

> - 80/443 必须在安全组放行，否则证书申请会失败。
> - 域名要**先解析生效**再让 Caddy 申请证书，否则失败。

---

## 8. 入库 / 更新知识库

把文档放进 `data/samples/`，然后在容器里执行：

```bash
docker compose exec rag-agent python ingest.py
docker compose exec rag-agent python graph_build.py   # 需要图谱时
docker compose restart rag-agent
```

也可以直接用网页「上传文档」Tab 上传（会自动触发入库）。

---

## 9. 更新代码 / 回滚

```bash
git pull
docker compose up -d --build    # 重新构建
```

---

## 注意事项（重点）

1. **⚠️ `.env` 是真实 key，绝不能进镜像 / 进 git。** 当前 `Dockerfile` 是 `COPY . .`，若没有 `.dockerignore`，`docker build` 会把 `.env`、`venv/`、`.git/` 全拷进镜像——等于把 key 打进镜像层。**本项目已带 `.dockerignore` 排除这些**，构建前务必确认它存在。
2. **⚠️ 网页登录**：本项目已内置 Gradio 登录，`.env` 里同时设置 `GRADIO_USERNAME` / `GRADIO_PASSWORD` 即自动开启。公网部署**必须**设这两项；否则任何拿到网址的人都能用你的知识库、烧你的 API 额度。想再加一层，可用 Caddy `basicauth` 或先不公开、只用第 6 步的 SSH 隧道。
3. **只对公网暴露 443，不暴露 7860**——7860 只给本机 Caddy 反代。
4. **数据持久化**：`chroma_db` 与 `data` 都挂载在宿主机，删容器不丢数据。定期备份这两个目录即可：`tar czf backup-$(date +%F).tar.gz chroma_db data`。
5. **成本**：DeepSeek / 硅基流动都很便宜，但「上传文档」会重复向量化、构建图谱时重复调 LLM，注意别让公开页面被刷。
6. **重启自愈**：`restart: unless-stopped` 已配好，服务器重启后容器自动拉起。
7. **内存**：2G 够跑线上服务；若批量入库时内存吃紧，可先在**本地**入库好，再把 `chroma_db/` 目录同步到服务器，服务器上直接 `docker compose up -d` 即可。
8. **排查**：`docker compose logs -f` 看实时日志；网页问答出错时，`app.py` 会把错误友好地显示在聊天里并提示「检查 key / 余额 / 网络」，照提示排查即可。
