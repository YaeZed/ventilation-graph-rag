"""Session-based account and conversation persistence endpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .models import (
    ConversationAttachment,
    ConversationRecord,
    SecurityEvent,
    Team,
    TeamMembership,
    UserProfile,
)


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
TEAM_ROLES = {
    TeamMembership.ROLE_OWNER,
    TeamMembership.ROLE_ADMIN,
    TeamMembership.ROLE_MEMBER,
}
TEAM_ADMIN_ROLES = {
    TeamMembership.ROLE_OWNER,
    TeamMembership.ROLE_ADMIN,
}
def _client_ip(request):
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()[:45]
    return str(request.META.get("REMOTE_ADDR") or "")[:45]


def _security_cache_key(prefix: str, *parts: Any):
    raw = ":".join(str(part or "").lower() for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"users:security:{prefix}:{digest}"


def _login_throttle_key(request, username: str):
    return _security_cache_key("login", _client_ip(request), username.strip().lower())


def _register_rate_key(request):
    return _security_cache_key("register", _client_ip(request))


def _login_retry_wait_seconds(key: str):
    state = cache.get(key) or {}
    locked_until = float(state.get("locked_until") or 0)
    now = timezone.now().timestamp()
    if locked_until > now:
        return max(1, int(locked_until - now))
    return 0


def _record_login_failure(key: str):
    limit = max(1, int(getattr(settings, "ACCOUNT_LOGIN_FAILURE_LIMIT", 5)))
    lockout_seconds = max(30, int(getattr(settings, "ACCOUNT_LOGIN_LOCKOUT_SECONDS", 300)))
    state = cache.get(key) or {}
    count = int(state.get("count") or 0) + 1
    locked_until = timezone.now().timestamp() + lockout_seconds if count >= limit else 0
    cache.set(key, {"count": count, "locked_until": locked_until}, timeout=lockout_seconds)
    return count, locked_until


def _clear_login_failures(key: str):
    cache.delete(key)


def _consume_register_attempt(request):
    limit = max(1, int(getattr(settings, "ACCOUNT_REGISTER_RATE_LIMIT", 5)))
    window_seconds = max(60, int(getattr(settings, "ACCOUNT_REGISTER_WINDOW_SECONDS", 600)))
    key = _register_rate_key(request)
    now = timezone.now().timestamp()
    state = cache.get(key) or {}
    started_at = float(state.get("started_at") or now)
    count = int(state.get("count") or 0)
    if now - started_at >= window_seconds:
        started_at = now
        count = 0
    if count >= limit:
        return max(1, int(window_seconds - (now - started_at)))
    cache.set(key, {"started_at": started_at, "count": count + 1}, timeout=window_seconds)
    return 0


def _record_security_event(
    request,
    event_type: str,
    user=None,
    username: str = "",
    metadata: dict[str, Any] | None = None,
):
    SecurityEvent.objects.create(
        user=user,
        username=(username or getattr(user, "username", "") or "")[:150],
        event_type=event_type,
        ip_address=_client_ip(request),
        user_agent=str(request.META.get("HTTP_USER_AGENT") or "")[:240],
        metadata=metadata or {},
    )


def _serialize_security_event(event: SecurityEvent):
    return {
        "id": event.id,
        "type": event.event_type,
        "username": event.username,
        "ipAddress": event.ip_address,
        "userAgent": event.user_agent,
        "metadata": event.metadata or {},
        "createdAt": event.created_at.isoformat(),
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


def _serialize_conversation(record: ConversationRecord, include_owner: bool = False, request=None):
    created_at = record.client_created_at or record.created_at
    updated_at = record.client_updated_at or record.updated_at
    messages = record.messages or []
    if request is not None:
        messages = _attach_images_to_messages(record, messages, request)
    payload = {
        "id": record.client_id,
        "title": record.title,
        "messages": messages,
        "createdAt": created_at.isoformat(),
        "updatedAt": updated_at.isoformat(),
        "sceneType": record.scene_type or None,
        "hazardLevel": record.hazard_level or None,
        "isArchived": record.is_archived,
        "previewImageUrl": record.preview_image_url or None,
        "isTitleManual": record.is_title_manual,
        "teamId": str(record.team_id) if record.team_id else None,
        "teamName": record.team.name if record.team_id and record.team else None,
    }
    if include_owner:
        profile = _profile_for(record.user)
        nickname = profile.nickname or record.user.first_name or record.user.username
        payload["owner"] = {
            "id": record.user_id,
            "username": record.user.username,
            "nickname": nickname,
            "avatarText": profile.avatar_text or nickname[:2] or "用",
        }
        payload["isOwnedByCurrentUser"] = False
    return payload


def _attach_images_to_messages(record: ConversationRecord, messages: Any, request):
    if not isinstance(messages, list):
        return []

    attachments = [_serialize_attachment(attachment, request) for attachment in record.attachments.order_by("created_at")]
    if not attachments:
        return messages

    by_message_id: dict[str, list[dict[str, Any]]] = {}
    for attachment in attachments:
        message_client_id = attachment.get("messageClientId")
        if message_client_id:
            by_message_id.setdefault(str(message_client_id), []).append(attachment)

    next_messages = []
    for message in messages:
        if not isinstance(message, dict):
            next_messages.append(message)
            continue
        next_message = {**message}
        message_attachments = by_message_id.get(str(next_message.get("id") or ""), [])
        if message_attachments:
            existing = next_message.get("attachments") if isinstance(next_message.get("attachments"), list) else []
            next_message["attachments"] = _dedupe_attachments([*existing, *message_attachments])
            image_items = [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "url": item["url"],
                    "size": item["size"],
                    "mimeType": item["mimeType"],
                    "createdAt": item["createdAt"],
                }
                for item in message_attachments
                if item.get("url")
            ]
            if image_items:
                next_message["images"] = image_items
                next_message["imageUrl"] = next_message.get("imageUrl") or image_items[0]["url"]
                next_message["sourceFileName"] = next_message.get("sourceFileName") or image_items[0]["name"]
        next_messages.append(next_message)
    return next_messages


def _dedupe_attachments(items: list[dict[str, Any]]):
    seen: set[str] = set()
    result = []
    for item in items:
        item_id = str(item.get("id") or item.get("url") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        result.append(item)
    return result


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


def _membership_for(user, team: Team):
    return TeamMembership.objects.filter(user=user, team=team).first()


def _serialize_team(team: Team, membership: TeamMembership | None = None):
    role = membership.role if membership else TeamMembership.ROLE_MEMBER
    return {
        "id": str(team.id),
        "name": team.name,
        "description": team.description,
        "role": role,
        "memberCount": team.memberships.count(),
        "createdAt": team.created_at.isoformat(),
        "updatedAt": team.updated_at.isoformat(),
    }


def _serialize_team_member(membership: TeamMembership):
    profile = _profile_for(membership.user)
    nickname = profile.nickname or membership.user.first_name or membership.user.username
    return {
        "id": membership.user.id,
        "username": membership.user.username,
        "nickname": nickname,
        "avatarText": profile.avatar_text or nickname[:2] or "用",
        "role": membership.role,
        "joinedAt": membership.joined_at.isoformat(),
    }


def _parse_team_id(value: Any):
    if value in (None, "", "personal", "null"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _team_for_user(user, team_id: Any):
    parsed_id = _parse_team_id(team_id)
    if parsed_id is None:
        return None
    return Team.objects.filter(id=parsed_id, memberships__user=user).first()


def _require_team_membership(request, team_id: Any):
    parsed_id = _parse_team_id(team_id)
    if parsed_id is None:
        return None, None, _json_error("团队不存在", status=404)
    membership = (
        TeamMembership.objects.select_related("team", "user")
        .filter(team_id=parsed_id, user=request.user)
        .first()
    )
    if membership is None:
        return None, None, _json_error("无权访问该团队", status=403)
    return membership.team, membership, None


def _require_team_admin(request, team_id: Any):
    team, membership, error = _require_team_membership(request, team_id)
    if error:
        return None, None, error
    if membership.role not in TEAM_ADMIN_ROLES:
        return None, None, _json_error("只有团队所有者或管理员可以执行该操作", status=403)
    return team, membership, None


def _team_scope_from_request(request):
    raw_team_id = request.GET.get("teamId")
    if raw_team_id in (None, "", "personal", "null"):
        return None, None
    team, _, error = _require_team_membership(request, raw_team_id)
    return team, error


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


def _build_stats_payload(user, days: int = 7, team: Team | None = None):
    if team is None:
        records = list(user.conversation_records.filter(team__isnull=True))
    else:
        records = list(team.conversation_records.select_related("user").all())
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
    if "teamId" in item:
        payload["team"] = _team_for_user(user, item.get("teamId"))
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


@ensure_csrf_cookie
@require_GET
def csrf_view(request):
    return JsonResponse({"ok": True, "csrfToken": get_token(request)})


@require_POST
def register_view(request):
    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    retry_after = _consume_register_attempt(request)
    if retry_after:
        _record_security_event(
            request,
            SecurityEvent.EVENT_REGISTER_THROTTLED,
            username=str((payload or {}).get("username") or ""),
            metadata={"retryAfterSeconds": retry_after},
        )
        response = _json_error(f"注册请求过于频繁，请 {retry_after} 秒后再试", status=429)
        response["Retry-After"] = str(retry_after)
        return response

    username = _clean_text(payload.get("username"), "", 150)
    password = str(payload.get("password") or "")
    nickname = _clean_text(payload.get("nickname"), username, 32)
    if not username or not password:
        return _json_error("请填写用户名和密码")

    try:
        validate_password(password, user=User(username=username, first_name=nickname))
    except ValidationError as exc:
        _record_security_event(
            request,
            SecurityEvent.EVENT_PASSWORD_REJECTED,
            username=username,
            metadata={"messages": list(exc.messages)},
        )
        return _json_error("；".join(exc.messages), status=400)

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
    _record_security_event(request, SecurityEvent.EVENT_REGISTER, user=user, username=username)
    return JsonResponse({"ok": True, "user": _serialize_user(user)}, json_dumps_params={"ensure_ascii": False})


@require_POST
def login_view(request):
    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    username = _clean_text(payload.get("username"), "", 150)
    password = str(payload.get("password") or "")
    if not username or not password:
        return _json_error("请填写用户名和密码")

    throttle_key = _login_throttle_key(request, username)
    retry_after = _login_retry_wait_seconds(throttle_key)
    if retry_after:
        target_user = User.objects.filter(username=username).first()
        _record_security_event(
            request,
            SecurityEvent.EVENT_LOGIN_THROTTLED,
            user=target_user,
            username=username,
            metadata={"retryAfterSeconds": retry_after},
        )
        response = _json_error(f"尝试次数过多，请 {retry_after} 秒后再试", status=429)
        response["Retry-After"] = str(retry_after)
        return response

    user = authenticate(request, username=username, password=password)
    if user is None:
        count, locked_until = _record_login_failure(throttle_key)
        target_user = User.objects.filter(username=username).first()
        _record_security_event(
            request,
            SecurityEvent.EVENT_LOGIN_FAILURE,
            user=target_user,
            username=username,
            metadata={"failureCount": count, "locked": bool(locked_until)},
        )
        return _json_error("用户名或密码不正确", status=401)

    _clear_login_failures(throttle_key)
    login(request, user)
    _record_security_event(request, SecurityEvent.EVENT_LOGIN_SUCCESS, user=user, username=username)
    return JsonResponse({"ok": True, "user": _serialize_user(user)}, json_dumps_params={"ensure_ascii": False})


@require_POST
def logout_view(request):
    if request.user.is_authenticated:
        _record_security_event(
            request,
            SecurityEvent.EVENT_LOGOUT,
            user=request.user,
            username=request.user.username,
        )
    logout(request)
    return JsonResponse({"ok": True})


@ensure_csrf_cookie
@require_GET
def me_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"ok": True, "user": None})
    return JsonResponse({"ok": True, "user": _serialize_user(request.user)}, json_dumps_params={"ensure_ascii": False})


@require_GET
def security_events_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    events = request.user.security_events.order_by("-created_at")[:20]
    return JsonResponse(
        {"ok": True, "events": [_serialize_security_event(event) for event in events]},
        json_dumps_params={"ensure_ascii": False},
    )


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


@require_http_methods(["GET", "POST"])
def teams_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    if request.method == "GET":
        memberships = (
            TeamMembership.objects.select_related("team")
            .filter(user=request.user)
            .order_by("team__name", "team_id")
        )
        return JsonResponse(
            {"ok": True, "teams": [_serialize_team(item.team, item) for item in memberships]},
            json_dumps_params={"ensure_ascii": False},
        )

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    name = _clean_text(payload.get("name"), "", 80)
    description = _clean_text(payload.get("description"), "", 240)
    if not name:
        return _json_error("请填写团队名称")

    with transaction.atomic():
        team = Team.objects.create(name=name, description=description, created_by=request.user)
        membership = TeamMembership.objects.create(
            team=team,
            user=request.user,
            role=TeamMembership.ROLE_OWNER,
        )

    return JsonResponse(
        {"ok": True, "team": _serialize_team(team, membership)},
        status=201,
        json_dumps_params={"ensure_ascii": False},
    )


@require_http_methods(["PATCH", "DELETE"])
def team_detail_view(request, team_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, membership, error = _require_team_admin(request, team_id)
    if error:
        return error

    if request.method == "DELETE":
        if membership.role != TeamMembership.ROLE_OWNER:
            return _json_error("只有团队所有者可以删除团队", status=403)
        team.delete()
        return JsonResponse({"ok": True})

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    name = payload.get("name")
    description = payload.get("description")
    if name is not None:
        team.name = _clean_text(name, team.name, 80)
    if description is not None:
        team.description = _clean_text(description, "", 240)
    if not team.name:
        return _json_error("请填写团队名称")
    team.save(update_fields=["name", "description", "updated_at"])

    return JsonResponse(
        {"ok": True, "team": _serialize_team(team, membership)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_http_methods(["GET", "POST"])
def team_members_view(request, team_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, membership, error = _require_team_membership(request, team_id)
    if error:
        return error

    if request.method == "GET":
        members = (
            team.memberships.select_related("user", "user__profile")
            .order_by("role", "user__username")
        )
        return JsonResponse(
            {"ok": True, "members": [_serialize_team_member(item) for item in members]},
            json_dumps_params={"ensure_ascii": False},
        )

    if membership.role not in TEAM_ADMIN_ROLES:
        return _json_error("只有团队所有者或管理员可以添加成员", status=403)

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    username = _clean_text(payload.get("username"), "", 150)
    role = str(payload.get("role") or TeamMembership.ROLE_MEMBER)
    if role not in (TeamMembership.ROLE_ADMIN, TeamMembership.ROLE_MEMBER):
        role = TeamMembership.ROLE_MEMBER
    if not username:
        return _json_error("请填写成员用户名")

    target_user = User.objects.filter(username=username).first()
    if target_user is None:
        return _json_error("用户不存在", status=404)

    target_membership, created = TeamMembership.objects.get_or_create(
        team=team,
        user=target_user,
        defaults={"role": role},
    )
    if not created and target_membership.role != TeamMembership.ROLE_OWNER:
        target_membership.role = role
        target_membership.save(update_fields=["role"])

    return JsonResponse(
        {"ok": True, "member": _serialize_team_member(target_membership)},
        status=201 if created else 200,
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def team_conversations_view(request, team_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, _, error = _require_team_membership(request, team_id)
    if error:
        return error

    records = (
        team.conversation_records.select_related("team", "user", "user__profile")
        .filter(is_archived=False)
        .order_by("-client_updated_at", "-updated_at")
    )
    conversations = []
    for record in records:
        item = _serialize_conversation(record, include_owner=True, request=request)
        item["isOwnedByCurrentUser"] = record.user_id == request.user.id
        conversations.append(item)
    return JsonResponse(
        {"ok": True, "team": _serialize_team(team, _membership_for(request.user, team)), "conversations": conversations},
        json_dumps_params={"ensure_ascii": False},
    )


@require_http_methods(["PATCH", "DELETE"])
def team_member_detail_view(request, team_id: int, user_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, membership, error = _require_team_membership(request, team_id)
    if error:
        return error

    target_membership = get_object_or_404(TeamMembership, team=team, user_id=user_id)
    is_self = target_membership.user_id == request.user.id
    can_manage = membership.role in TEAM_ADMIN_ROLES
    if not can_manage and not is_self:
        return _json_error("无权管理该成员", status=403)
    if target_membership.role == TeamMembership.ROLE_OWNER:
        return _json_error("不能修改或移除团队所有者", status=403)

    if request.method == "DELETE":
        target_membership.delete()
        return JsonResponse({"ok": True})

    if not can_manage:
        return _json_error("只有团队所有者或管理员可以修改角色", status=403)

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")
    role = str(payload.get("role") or TeamMembership.ROLE_MEMBER)
    if role not in (TeamMembership.ROLE_ADMIN, TeamMembership.ROLE_MEMBER):
        return _json_error("角色只能是 admin 或 member")
    target_membership.role = role
    target_membership.save(update_fields=["role"])
    return JsonResponse(
        {"ok": True, "member": _serialize_team_member(target_membership)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def conversations_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    records = request.user.conversation_records.select_related("team").order_by("-client_updated_at", "-updated_at")
    return JsonResponse(
        {"ok": True, "conversations": [_serialize_conversation(record, request=request) for record in records]},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def stats_summary_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, team_error = _team_scope_from_request(request)
    if team_error:
        return team_error
    days = _parse_positive_int(request.GET.get("days"), 7)
    return JsonResponse(
        {"ok": True, "stats": _build_stats_payload(request.user, days, team=team)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def stats_trends_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, team_error = _team_scope_from_request(request)
    if team_error:
        return team_error
    days = _parse_positive_int(request.GET.get("days"), 7)
    if team is None:
        records = list(request.user.conversation_records.filter(team__isnull=True, is_archived=False))
    else:
        records = list(team.conversation_records.filter(is_archived=False))
    return JsonResponse(
        {"ok": True, "trends": _build_recent_days(records, days)},
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def stats_hazards_view(request):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    team, team_error = _team_scope_from_request(request)
    if team_error:
        return team_error
    return JsonResponse(
        {"ok": True, "hazards": _build_stats_payload(request.user, team=team)["hazardCounts"]},
        json_dumps_params={"ensure_ascii": False},
    )


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

    records = request.user.conversation_records.select_related("team").order_by("-client_updated_at", "-updated_at")
    return JsonResponse(
        {"ok": True, "conversations": [_serialize_conversation(record, request=request) for record in records]},
        json_dumps_params={"ensure_ascii": False},
    )


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


@require_http_methods(["PATCH", "POST"])
def conversation_team_view(request, client_id: str):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    payload = _load_json_body(request)
    if payload is None:
        return _json_error("请求体必须是 JSON")

    raw_team_id = payload.get("teamId")
    team = None
    if raw_team_id not in (None, "", "personal", "null"):
        team = _team_for_user(request.user, raw_team_id)
        if team is None:
            return _json_error("无权分配到该团队", status=403)

    record = get_object_or_404(ConversationRecord, user=request.user, client_id=client_id)
    record.team = team
    record.save(update_fields=["team", "updated_at"])
    return JsonResponse(
        {"ok": True, "conversation": _serialize_conversation(record, request=request)},
        json_dumps_params={"ensure_ascii": False},
    )


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


@require_http_methods(["DELETE", "POST"])
def delete_attachment_view(request, attachment_id: int):
    auth_error = _require_auth(request)
    if auth_error:
        return auth_error

    attachment = get_object_or_404(ConversationAttachment, user=request.user, id=attachment_id)
    _delete_attachment_file(attachment)
    attachment.delete()
    return JsonResponse({"ok": True})
