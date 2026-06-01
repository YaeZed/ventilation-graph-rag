"""HTTP endpoints for text, image, SSE chat, and vision evaluation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .pipeline_service import get_pipeline_service
from .vision_evaluation import evaluate_vision_samples


SSE_HEARTBEAT_SECONDS = 15


def _json_error(message: str, status: int = 400):
    return JsonResponse({"ok": False, "error": message}, status=status)


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lower_message = message.lower()
    if (
        "AllocationQuota.FreeTierOnly" in message
        or "free tier" in lower_message
        or "quota" in lower_message
        or "insufficient_quota" in lower_message
    ):
        return (
            "模型服务额度不足：当前 DashScope/Qwen 账号的免费额度已耗尽，"
            "并且控制台开启了“仅使用免费额度/free tier only”。请在 DashScope 控制台关闭该限制并开通付费，"
            "或更换仍有额度的 DASHSCOPE_API_KEY / 模型后重试。"
        )
    return message


def _load_json_body(request):
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return None


def _load_sensor_data(value: Any):
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    return value


def _get_pipeline():
    return get_pipeline_service().get_pipeline()


@require_GET
def vision_scenes(request):
    pipeline = _get_pipeline()
    scenes = []
    if pipeline.template_engine:
        for scene in pipeline.template_engine.list_scene_schemas():
            scenes.append(
                {
                    "id": scene.get("id"),
                    "name": scene.get("name"),
                    "schema": scene.get("schema", {}),
                    "aliases": scene.get("aliases", []),
                }
            )
    return JsonResponse({"ok": True, "scenes": scenes})


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
    sensor_data = _load_sensor_data(payload.get("sensor_data") or payload.get("sensorData"))
    try:
        answer = _get_pipeline().query(question, top_k=top_k, stream=False, sensor_data=sensor_data)
        return JsonResponse({"ok": True, "answer": answer})
    except Exception as exc:
        return _json_error(_friendly_error(exc), status=502)


@csrf_exempt
@require_POST
def chat_upload(request):
    question = (request.POST.get("question") or request.POST.get("message") or "请判断图片中的通风安全隐患").strip()
    images = _uploaded_images(request)
    if not images:
        return _json_error("缺少 image 文件")

    top_k = int(request.POST.get("top_k") or 5)
    sensor_data = _load_sensor_data(request.POST.get("sensor_data") or request.POST.get("sensorData"))
    image_paths = [_save_upload(image) for image in images]
    try:
        answer = _get_pipeline().query(
            question,
            top_k=top_k,
            image_paths=[str(path) for path in image_paths],
            sensor_data=sensor_data,
            stream=False,
        )
        return JsonResponse({"ok": True, "answer": answer})
    except Exception as exc:
        return _json_error(_friendly_error(exc), status=502)
    finally:
        for image_path in image_paths:
            _safe_unlink(image_path)


@csrf_exempt
@require_POST
def chat_stream(request):
    question = ""
    image_paths: list[Path] = []
    sensor_data = None
    top_k = 5

    if request.content_type and request.content_type.startswith("multipart/form-data"):
        question = (request.POST.get("question") or request.POST.get("message") or "请判断图片中的通风安全隐患").strip()
        top_k = int(request.POST.get("top_k") or 5)
        sensor_data = _load_sensor_data(request.POST.get("sensor_data") or request.POST.get("sensorData"))
        image_paths = [_save_upload(image) for image in _uploaded_images(request)]
    else:
        payload = _load_json_body(request)
        if payload is None:
            return _json_error("请求体必须是 JSON 或 multipart/form-data")
        question = (payload.get("question") or payload.get("message") or "").strip()
        top_k = int(payload.get("top_k") or 5)
        sensor_data = _load_sensor_data(payload.get("sensor_data") or payload.get("sensorData"))

    if not question:
        return _json_error("缺少 question/message")

    response = StreamingHttpResponse(
        _stream_pipeline_events(question, top_k, image_paths, sensor_data),
        content_type="text/event-stream; charset=utf-8",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
@require_POST
def vision_evaluate(request):
    metadata_raw = request.POST.get("metadata")
    if not metadata_raw:
        return _json_error("缺少 metadata")

    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        return _json_error("metadata 必须是 JSON 字符串")

    samples_meta = metadata.get("samples")
    if not isinstance(samples_meta, list) or not samples_meta:
        return _json_error("metadata.samples 至少需要 1 个样本")

    saved_paths = []
    samples = []
    try:
        for index, sample_meta in enumerate(samples_meta):
            if not isinstance(sample_meta, dict):
                return _json_error(f"samples[{index}] 必须是对象")

            image = request.FILES.get(f"image_{index}")
            if not image:
                return _json_error(f"缺少 image_{index}")

            image_path = _save_upload(image)
            saved_paths.append(image_path)
            samples.append(
                {
                    "id": sample_meta.get("id") or f"sample-{index + 1}",
                    "file_name": image.name,
                    "question": sample_meta.get("question"),
                    "expected_scene_id": sample_meta.get("expected_scene_id"),
                    "expected_fields": sample_meta.get("expected_fields") or {},
                    "image_path": str(image_path),
                }
            )

        pipeline = _get_pipeline()
        if not pipeline.vision_extractor:
            return _json_error("视觉识别模块尚未初始化", status=503)

        report = evaluate_vision_samples(pipeline, samples)
        return JsonResponse({"ok": True, **report}, json_dumps_params={"ensure_ascii": False})
    except Exception as exc:
        return _json_error(_friendly_error(exc), status=502)
    finally:
        for path in saved_paths:
            _safe_unlink(path)


def _stream_pipeline_events(
    question: str,
    top_k: int,
    image_paths: list[Path] | None = None,
    sensor_data: dict[str, Any] | None = None,
):
    queue: Queue[Any] = Queue()
    finished = object()
    started_at = time.monotonic()
    image_paths = image_paths or []
    image_path_texts = [str(path) for path in image_paths]

    def run_pipeline():
        try:
            queue.put(("status", {"message": "正在初始化知识库与辨识流水线..."}))
            if image_path_texts:
                queue.put(("status", {"message": f"正在准备 {len(image_path_texts)} 张图片辨识流程..."}))
            if sensor_data:
                queue.put(("status", {"message": "正在接入传感器数据..."}))

            chunks = _get_pipeline().query(
                question,
                top_k=top_k,
                stream=True,
                image_paths=image_path_texts,
                sensor_data=sensor_data,
            )
            if not image_path_texts:
                queue.put(("status", {"message": "正在生成辨识报告..."}))

            for chunk in chunks:
                if isinstance(chunk, dict):
                    event_type = chunk.get("type")
                    if event_type == "step":
                        queue.put(("step", chunk))
                    elif event_type == "token":
                        queue.put(("token", {"content": chunk.get("content", "")}))
                    elif event_type == "done":
                        queue.put(("done", {"message": chunk.get("message", "completed")}))
                    elif event_type == "error":
                        queue.put(("error", {"message": chunk.get("message", "流式响应失败")}))
                    elif event_type == "status":
                        queue.put(("status", {"message": chunk.get("message", "")}))
                    continue

                queue.put(("token", {"content": chunk}))

            if not image_path_texts:
                queue.put(("done", {"message": "completed"}))
        except Exception as exc:
            queue.put(("error", {"message": _friendly_error(exc)}))
        finally:
            for image_path in image_paths:
                _safe_unlink(image_path)
            queue.put(finished)

    worker = Thread(target=run_pipeline, daemon=True)
    worker.start()

    yield _sse("status", {"message": "started"})

    while True:
        try:
            item = queue.get(timeout=SSE_HEARTBEAT_SECONDS)
        except Empty:
            elapsed = int(time.monotonic() - started_at)
            yield _sse("status", {"message": f"仍在处理中，已等待 {elapsed} 秒..."})
            continue

        if item is finished:
            break

        event, data = item
        yield _sse(event, data)

        if event in {"done", "error"}:
            break


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


def _uploaded_images(request) -> list[Any]:
    images = list(request.FILES.getlist("images"))
    if not images:
        image = request.FILES.get("image")
        if image:
            images = [image]
    return images


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
