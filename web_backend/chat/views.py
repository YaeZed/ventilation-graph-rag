"""HTTP endpoints for text, image, and SSE chat."""

import json
import os
import tempfile
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .pipeline_service import get_pipeline_service


def _json_error(message: str, status: int = 400):
    return JsonResponse({"ok": False, "error": message}, status=status)


def _load_json_body(request):
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return None


def _get_pipeline():
    return get_pipeline_service().get_pipeline()


@csrf_exempt
@require_POST
def chat(request):
    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    question = (payload.get("question") or payload.get("message") or "").strip()
    if not question:
        return _json_error("缺少 question/message")

    top_k = int(payload.get("top_k") or 5)
    answer = _get_pipeline().query(question, top_k=top_k, stream=False)
    return JsonResponse({"ok": True, "answer": answer})


@csrf_exempt
@require_POST
def chat_upload(request):
    question = (request.POST.get("question") or request.POST.get("message") or "请判断图片中的通风安全隐患").strip()
    image = request.FILES.get("image")
    if not image:
        return _json_error("缺少 image 文件")

    top_k = int(request.POST.get("top_k") or 5)
    image_path = _save_upload(image)
    try:
        answer = _get_pipeline().query(question, top_k=top_k, image_path=str(image_path), stream=False)
        return JsonResponse({"ok": True, "answer": answer})
    finally:
        _safe_unlink(image_path)


@csrf_exempt
@require_POST
def chat_stream(request):
    question = ""
    image_path = None
    top_k = 5

    if request.content_type and request.content_type.startswith("multipart/form-data"):
        question = (request.POST.get("question") or request.POST.get("message") or "请判断图片中的通风安全隐患").strip()
        top_k = int(request.POST.get("top_k") or 5)
        image = request.FILES.get("image")
        if image:
            image_path = _save_upload(image)
    else:
        payload = _load_json_body(request)
        if payload is None:
            return _json_error("请求体必须是 JSON 或 multipart/form-data")
        question = (payload.get("question") or payload.get("message") or "").strip()
        top_k = int(payload.get("top_k") or 5)

    if not question:
        return _json_error("缺少 question/message")

    def event_stream():
        try:
            yield _sse("status", {"message": "started"})
            chunks = _get_pipeline().query(
                question,
                top_k=top_k,
                stream=True,
                image_path=str(image_path) if image_path else None,
            )
            for chunk in chunks:
                yield _sse("token", {"content": chunk})
            yield _sse("done", {"message": "completed"})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
        finally:
            if image_path:
                _safe_unlink(image_path)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream; charset=utf-8")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_upload(uploaded_file) -> Path:
    settings.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(get_valid_filename(uploaded_file.name)).suffix or ".jpg"
    fd, raw_path = tempfile.mkstemp(prefix="ventilation_", suffix=suffix, dir=settings.MEDIA_ROOT)
    path = Path(raw_path)
    with os.fdopen(fd, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return path


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
