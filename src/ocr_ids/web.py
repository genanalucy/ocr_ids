"""A small browser UI and JSON API for the phase-one IDS model."""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from .inference import IDSInferencer


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR-IDS</title>
<style>
body{max-width:760px;margin:48px auto;padding:0 20px;font:16px system-ui;color:#172033;background:#f8fafc}main{background:#fff;padding:32px;border-radius:16px;box-shadow:0 4px 20px #0f172a12}h1{margin-top:0}input,button{font:inherit}button{padding:9px 16px;border:0;border-radius:8px;background:#1769e0;color:white;cursor:pointer}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px;border-radius:8px}img{display:none;max-width:224px;max-height:224px;border:1px solid #cbd5e1;margin:16px 0}.hint{color:#64748b}</style>
<main><h1>方块字 → IDS</h1><p>上传一张已裁切的单字图片，模型输出其 IDS 结构。</p>
<form id="form"><input id="image" type="file" accept="image/png,image/jpeg,image/webp" required> <button>识别</button></form>
<img id="preview" alt="上传的字图"><img id="normalized" alt="预处理后的字图"><p id="status" class="hint">正在检查模型状态…</p><pre id="result" hidden></pre></main>
<script>
const image=document.querySelector('#image'), preview=document.querySelector('#preview'), normalized=document.querySelector('#normalized'), status=document.querySelector('#status'), result=document.querySelector('#result');
fetch('/api/status').then(r=>r.json()).then(x=>status.textContent=x.ready?'模型已就绪。':'模型未就绪：'+x.detail).catch(()=>status.textContent='无法连接服务');
image.onchange=()=>{const f=image.files[0];if(f){preview.src=URL.createObjectURL(f);preview.style.display='block'}};
document.querySelector('#form').onsubmit=async e=>{e.preventDefault();status.textContent='正在推理…';result.hidden=true;const fd=new FormData();fd.append('image',image.files[0]);const r=await fetch('/api/predict',{method:'POST',body:fd});const body=await r.json();if(!r.ok){status.textContent='识别失败：'+(body.detail||r.status);return}normalized.src=body.normalized_preview;normalized.style.display='block';status.textContent=body.preprocess.warnings.length?'完成，但输入提示：'+body.preprocess.warnings.join(', '):'完成';result.textContent=JSON.stringify(body,null,2);result.hidden=false};
</script></html>"""


def create_app(checkpoint: str | Path | None = None, *, device: str | None = None) -> FastAPI:
    app = FastAPI(title="OCR-IDS", version="0.1.0")
    checkpoint_value = checkpoint or os.environ.get("OCR_IDS_CHECKPOINT")
    inferencer: IDSInferencer | None = None
    load_error: str | None = None
    if checkpoint_value:
        try:
            inferencer = IDSInferencer(checkpoint_value, device=device)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            load_error = str(exc)
    else:
        load_error = "未设置 OCR_IDS_CHECKPOINT，也未传入 --checkpoint"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return {"ready": inferencer is not None, "checkpoint": str(checkpoint_value or ""), "detail": load_error}

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)) -> dict[str, object]:
        if inferencer is None:
            raise HTTPException(status_code=503, detail=load_error or "模型未就绪")
        if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise HTTPException(status_code=415, detail="仅支持 PNG、JPEG、WebP 图片")
        try:
            raw = await image.read()
            with Image.open(io.BytesIO(raw)) as uploaded:
                prepared = inferencer.prepare(uploaded.copy())
                prediction = inferencer.predict_prepared(prepared.image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"无法读取图片：{exc}") from exc
        preview = io.BytesIO()
        prepared.image.save(preview, format="PNG")
        return {
            "ids": prediction.ids,
            "tree": prediction.tree,
            "confidence": round(prediction.confidence, 6),
            "syntax_valid": prediction.syntax_valid,
            "tokens": prediction.tokens,
            "preprocess": prepared.to_dict(),
            "normalized_preview": "data:image/png;base64,"
            + base64.b64encode(preview.getvalue()).decode("ascii"),
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 OCR-IDS 前端与推理 API")
    parser.add_argument("--checkpoint", help="训练输出目录中的 last.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", help="例如 cuda、cuda:0 或 cpu")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.checkpoint, device=args.device), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
