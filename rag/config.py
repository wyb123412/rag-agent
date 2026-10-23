"""集中读取配置：从 .env 加载 LLM 与 Embedding 的 key/地址/模型，并暴露客户端工厂。"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 全局日志级别可通过 LOG_LEVEL 环境变量控制（默认 INFO）
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# 无论从哪个目录运行脚本，都固定去项目根目录读 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------- LLM（对话模型）----------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---------- Embedding（向量化模型）----------
# 若没单独填 EMBED_API_KEY，就复用 LLM_API_KEY（方便「LLM 和 Embedding 都用硅基流动」的场景）
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "") or LLM_API_KEY
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ---------- 重排序（Rerank）----------
# 未单独填 RERANK_API_KEY 就复用 EMBED_API_KEY（同为硅基流动的 key）
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "") or EMBED_API_KEY
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ---------- 混合检索 ----------
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF 融合常数，越大名次权重越平均

# ---------- 切片 / 检索 / 路径 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_MODE = os.getenv("CHUNK_MODE", "semantic")  # semantic=按标题语义切片 / fixed=固定大小
TOP_K = int(os.getenv("TOP_K", "4"))
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "samples"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
GRAPH_PATH = os.getenv("GRAPH_PATH", str(PROJECT_ROOT / "data" / "graph.json"))

# ---------- 查询改写 ----------
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "none")  # none / hyde / multi

# ---------- 图片 OCR（走硅基流动的视觉模型）----------
OCR_MODEL = os.getenv("OCR_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# ---------- Corrective RAG / RAPTOR（进阶）----------
CORRECTIVE_RAG = os.getenv("CORRECTIVE_RAG", "1") == "1"  # 检索后相关性自检，不相关则改写重检索
RAPTOR_SIM_THRESHOLD = float(os.getenv("RAPTOR_SIM_THRESHOLD", "0.6"))  # RAPTOR 聚类相似度阈值


def get_llm_client() -> OpenAI:
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def get_embed_client() -> OpenAI:
    return OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)


def get_ocr_client() -> OpenAI:
    return OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
