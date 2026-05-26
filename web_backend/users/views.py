"""Session-based account and conversation persistence endpoints."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .models import ConversationRecord, UserProfile


DEFAULT_SETTINGS = {
    "useStream": True,
    "autoExpandSteps": True,
    "temperature": 0.2,
}


def _json_error(message: str, status: int = 400):
    return JsonResponse({"ok": False, "error": message}, status=status, json_dumps_params={"ensure_ascii": False})


def _load_json_body(request):
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return None


def _profile_for(user):
    profile, _ = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            "nickname": user.first_name or user.username,
            "avatar_text": (user.first_name or user.username or "用")[:2],
            "settings": DEFAULT_SETTINGS.copy(),
        },
    )
    if not profile.settings:
        profile.settings = DEFAULT_SETTINGS.copy()
        profile.save(update_fields=["settings", "updated_at"])
    return profile


def _serialize_user(user):
    profile = _profile_for(user)
    nickname = profile.nickname or user.first_name or user.username
    return {
        "id": user.id,
        "username": user.username,
        "nickname": nickname,
        "avatarText": profile.avatar_text or nickname[:2] or "用",
        "settings": {**DEFAULT_SETTINGS, **(profile.settings or {})},
    }


def _parse_client_datetime(value: str | None):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed)
    return parsed


def _serialize_conversation(record: ConversationRecord):
    created_at = record.client_created_at or record.created_at
    updated_at = record.client_updated_at or record.updated_at
    return {
        "id": record.client_id,
        "title": record.title,
        "messages": record.messages or [],
        "createdAt": created_at.isoformat(),
        "updatedAt": updated_at.isoformat(),
        "sceneType": record.scene_type or None,
        "hazardLevel": record.hazard_level or None,
        "isArchived": record.is_archived,
        "previewImageUrl": record.preview_image_url or None,
        "isTitleManual": record.is_title_manual,
    }


def _clean_text(value: Any, fallback: str, max_length: int):
    text = str(value or "").strip() or fallback
    return text[:max_length]


def _clean_messages(value: Any):
    return value if isinstance(value, list) else []


def _upsert_conversation(user, item: dict[str, Any]):
    client_id = _clean_text(item.get("id"), "", 80)
    if not client_id:
        return None

    payload = {
        "title": _clean_text(item.get("title"), "新建辨识", 120),
        "messages": _clean_messages(item.get("messages")),
        "scene_type": _clean_text(item.get("sceneType"), "", 120),
        "hazard_level": _clean_text(item.get("hazardLevel"), "", 80),
        "is_archived": bool(item.get("isArchived")),
        "preview_image_url": str(item.get("previewImageUrl") or ""),
        "is_title_manual": bool(item.get("isTitleManual")),
        "client_created_at": _parse_client_datetime(item.get("createdAt")),
        "client_updated_at": _parse_client_datetime(item.get("updatedAt")),
    }
    record, _ = ConversationRecord.objects.update_or_create(
        user=user,
        client_id=client_id,
        defaults=payload,
    )
    return record


def _require_auth(request):
    if not request.user.is_authenticated:
        return _json_error("请先登录", status=401)
    return None


@csrf_exempt
@require_POST
def register_view(request):
    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    username = _clean_text(payload.get("username"), "", 150)
    password = str(payload.get("password") or "")
    nickname = _clean_text(payload.get("nickname"), username, 32)
    if not username or not password:
        return _json_error("请填写用户名和密码")
    if len(password) < 6:
        return _json_error("密码至少 6 位")

    try:
        user = User.objects.create_user(username=username, password=password, first_name=nickname)
    except IntegrityError:
        return _json_error("用户名已存在", status=409)

    UserProfile.objects.create(
        user=user,
        nickname=nickname,
        avatar_text=(payload.get("avatarText") or nickname[:2] or username[:2])[:4],
        settings={**DEFAULT_SETTINGS, **(payload.get("settings") or {})},
    )
    login(request, user)
    return JsonResponse({"ok": True, "user": _serialize_user(user)}, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def login_view(request):
    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    username = _clean_text(payload.get("username"), "", 150)
    password = str(payload.get("password") or "")
    user = authenticate(request, username=username, password=password)
    if user is None:
        return _json_error("用户名或密码不正确", status=401)

    login(request, user)
    return JsonResponse({"ok": True, "user": _serialize_user(user)}, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def logout_view(request):
    logout(request)
    return JsonResponse({"ok": True})


@require_GET
def me_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"ok": True, "user": None})
    return JsonResponse({"ok": True, "user": _serialize_user(request.user)}, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_http_methods(["PATCH", "POST"])
def profile_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    profile = _profile_for(request.user)
    nickname = payload.get("nickname")
    avatar_text = payload.get("avatarText")
    settings = payload.get("settings")

    if nickname is not None:
        profile.nickname = _clean_text(nickname, request.user.username, 32)
        request.user.first_name = profile.nickname
        request.user.save(update_fields=["first_name"])
    if avatar_text is not None:
        profile.avatar_text = _clean_text(avatar_text, profile.nickname[:2] or "用", 4)
    if isinstance(settings, dict):
        profile.settings = {**DEFAULT_SETTINGS, **settings}
    profile.save()

    return JsonResponse({"ok": True, "user": _serialize_user(request.user)}, json_dumps_params={"ensure_ascii": False})


@require_GET
def conversations_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    records = request.user.conversation_records.order_by("-client_updated_at", "-updated_at")
    return JsonResponse(
        {"ok": True, "conversations": [_serialize_conversation(record) for record in records]},
        json_dumps_params={"ensure_ascii": False},
    )


@csrf_exempt
@require_POST
def sync_conversations_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return _json_error("conversations 必须是数组")

    for item in conversations:
        if isinstance(item, dict):
            _upsert_conversation(request.user, item)

    records = request.user.conversation_records.order_by("-client_updated_at", "-updated_at")
    return JsonResponse(
        {"ok": True, "conversations": [_serialize_conversation(record) for record in records]},
        json_dumps_params={"ensure_ascii": False},
    )


@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def delete_conversation_view(request, client_id: str):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    ConversationRecord.objects.filter(user=request.user, client_id=client_id).delete()
    return JsonResponse({"ok": True})
