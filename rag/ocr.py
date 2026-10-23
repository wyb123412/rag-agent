"""图片 OCR：调用硅基流动的视觉模型，把图片里的文字提取成纯文本。

默认用 Qwen3-VL-8B（多模态，可直接输出干净的纯文本）；如需更强的表格/文档
结构化识别，可在 .env 里把 OCR_MODEL 改成 deepseek-ai/DeepSeek-OCR。
"""
import base64
import re
from pathlib import Path

from .config import get_ocr_client, OCR_MODEL

# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

_MIME = {
    ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".bmp": "bmp",
    ".webp": "webp", ".gif": "gif", ".tiff": "tiff",
}

_PROMPT = "请完整转录这张图片中的所有文字，按阅读顺序分行输出，只输出文字本身，不要解释。"


def ocr_image(path) -> str:
    """把一张图片 OCR 成纯文本；任何失败都返回空串（不阻断其它文档入库）。"""
    try:
        suffix = Path(path).suffix.lower()
        mime = _MIME.get(suffix, "jpeg")
        data_url = "data:image/{};base64,{}".format(
            mime, base64.b64encode(Path(path).read_bytes()).decode()
        )
        client = get_ocr_client()
        resp = client.chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=4000,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return ""

    # 压掉多余空白，去掉空行
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
