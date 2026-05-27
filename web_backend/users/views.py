"""Session-based account and conversation persistence endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .models import ConversationAttachment, ConversationRecord, UserProfile


DEFAULT_SETTINGS = {
    "useStream": True,
    "autoExpandSteps": True,
    "temperature": 0.2,
}
MAX_ATTACHMENT_SIZE = 8 * 1024 * 1024
HAZARD_TONES = {
    "高风险": "danger",
    "中风险": "warning",
    "低风险": "success",
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


def _serialize_attachment(attachment: ConversationAttachment, request):
    file_url = request.build_absolute_uri(attachment.file.url) if attachment.file else ""
    return {
        "id": str(attachment.id),
        "kind": "image",
        "messageClientId": attachment.message_client_id or None,
        "name": attachment.original_name,
        "url": file_url,
        "thumbnailUrl": file_url,
        "size": attachment.size,
        "mimeType": attachment.mime_type,
        "createdAt": attachment.created_at.isoformat(),
    }


def _normalize_hazard_label(value: Any):
    label = str(value or "").strip()
    if not label:
        return "未分级"
    lower = label.lower()
    if any(token in label for token in ["高", "重大", "严重"]) or any(
        token in lower for token in ["danger", "high"]
    ):
        return "高风险"
    if any(token in label for token in ["中", "较大"]) or any(
        token in lower for token in ["warning", "medium"]
    ):
        return "中风险"
    if any(token in label for token in ["低", "一般", "轻微"]) or any(
        token in lower for token in ["success", "low"]
    ):
        return "低风险"
    return label


def _hazard_rank(label: str):
    if label == "高风险":
        return 1
    if label == "中风险":
        return 2
    if label == "低风险":
        return 3
    if label == "未分级":
        return 9
    return 4


def _hazard_tone(label: str):
    return HAZARD_TONES.get(label, "neutral")


def _record_updated_at(record: ConversationRecord):
    return record.client_updated_at or record.updated_at or record.created_at


def _count_completed_reports(messages: Any):
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("status") == "done"
    )


def _build_recent_days(records, days: int):
    safe_days = min(max(days, 1), 90)
    today = timezone.localdate()
    buckets = {
        (today - timedelta(days=safe_days - 1 - index)).isoformat(): 0
        for index in range(safe_days)
    }
    for record in records:
        updated_at = _record_updated_at(record)
        if not updated_at:
            continue
        local_day = timezone.localtime(updated_at).date().isoformat()
        if local_day in buckets:
            buckets[local_day] += 1
    return [{"date": date, "count": count} for date, count in buckets.items()]


def _build_stats_payload(user, days: int = 7):
    records = list(user.conversation_records.all())
    active_records = [record for record in records if not record.is_archived]
    active_records.sort(key=_record_updated_at, reverse=True)

    scene_counts: dict[str, int] = {}
    hazard_counts: dict[str, int] = {}
    completed_reports = 0
    completed_conversations = 0
    total_messages = 0

    for record in active_records:
        messages = record.messages if isinstance(record.messages, list) else []
        total_messages += len(messages)
        report_count = _count_completed_reports(messages)
        completed_reports += report_count
        if report_count:
            completed_conversations += 1

        scene_label = record.scene_type or record.hazard_level or "未分类"
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        hazard_label = _normalize_hazard_label(record.hazard_level)
        hazard_counts[hazard_label] = hazard_counts.get(hazard_label, 0) + 1

    hazard_items = sorted(
        (
            {
                "label": label,
                "count": count,
                "tone": _hazard_tone(label),
            }
            for label, count in hazard_counts.items()
        ),
        key=lambda item: (-item["count"], _hazard_rank(item["label"])),
    )
    recent_days = _build_recent_days(active_records, days)
    latest_activity = _record_updated_at(active_records[0]).isoformat() if active_records else ""

    return {
        "totalConversations": len(active_records),
        "totalMessages": total_messages,
        "completedReports": completed_reports,
        "archivedCount": len(records) - len(active_records),
        "completionRate": round((completed_conversations / len(active_records)) * 100)
        if active_records
        else 0,
        "activeDays": sum(1 for item in recent_days if item["count"] > 0),
        "latestActivity": latest_activity,
        "recentSevenDays": recent_days,
        "sceneCounts": sorted(
            ({"label": label, "count": count} for label, count in scene_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        ),
        "hazardCounts": hazard_items,
        "topHazardLabel": next(
            (item["label"] for item in hazard_items if item["label"] != "未分级"),
            "未分级",
        ),
    }


def _delete_attachment_file(attachment: ConversationAttachment):
    if attachment.file:
        attachment.file.delete(save=False)


def _clean_text(value: Any, fallback: str, max_length: int):
    text = str(value or "").strip() or fallback
    return text[:max_length]


def _clean_messages(value: Any):
    return value if isinstance(value, list) else []


def _parse_positive_int(value: Any, fallback: int):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


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


def _get_or_create_conversation_for_attachment(user, client_id: str):
    record, _ = ConversationRecord.objects.get_or_create(
        user=user,
        client_id=client_id,
        defaults={
            "title": "新建辨识",
            "messages": [],
            "client_created_at": timezone.now(),
            "client_updated_at": timezone.now(),
        },
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


@require_GET
def stats_summary_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    days = _parse_positive_int(request.GET.get("days"), 7)
    return JsonResponse(
        {"ok": True, "stats": _build_stats_payload(request.user, days)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def stats_trends_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    days = _parse_positive_int(request.GET.get("days"), 7)
    records = list(request.user.conversation_records.filter(is_archived=False))
    return JsonResponse(
        {"ok": True, "trends": _build_recent_days(records, days)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def stats_hazards_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    return JsonResponse(
        {"ok": True, "hazards": _build_stats_payload(request.user)["hazardCounts"]},
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

    records = ConversationRecord.objects.filter(user=request.user, client_id=client_id)
    for attachment in ConversationAttachment.objects.filter(user=request.user, conversation__in=records):
        _delete_attachment_file(attachment)
    records.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def upload_attachment_view(request, client_id: str):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    uploaded = request.FILES.get("image") or request.FILES.get("file")
    if uploaded is None:
        return _json_error("请上传图片文件")
    mime_type = uploaded.content_type or ""
    if not mime_type.startswith("image/"):
        return _json_error("只支持图片附件")
    if uploaded.size > MAX_ATTACHMENT_SIZE:
        return _json_error("图片不能超过 8MB")

    message_client_id = _clean_text(request.POST.get("messageClientId"), "", 80)
    conversation = _get_or_create_conversation_for_attachment(request.user, client_id)
    attachment = ConversationAttachment.objects.create(
        user=request.user,
        conversation=conversation,
        message_client_id=message_client_id,
        file=uploaded,
        original_name=_clean_text(uploaded.name, "现场图片", 160),
        mime_type=mime_type[:80],
        size=uploaded.size,
    )
    if not conversation.preview_image_url:
        conversation.preview_image_url = attachment.file.url
        conversation.save(update_fields=["preview_image_url", "updated_at"])

    return JsonResponse(
        {"ok": True, "attachment": _serialize_attachment(attachment, request)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def attachments_view(request, client_id: str):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    record = get_object_or_404(ConversationRecord, user=request.user, client_id=client_id)
    attachments = record.attachments.order_by("created_at")
    return JsonResponse(
        {
            "ok": True,
            "attachments": [_serialize_attachment(attachment, request) for attachment in attachments],
        },
        json_dumps_params={"ensure_ascii": False},
    )


@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def delete_attachment_view(request, attachment_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    attachment = get_object_or_404(ConversationAttachment, user=request.user, id=attachment_id)
    _delete_attachment_file(attachment)
    attachment.delete()
    return JsonResponse({"ok": True})
