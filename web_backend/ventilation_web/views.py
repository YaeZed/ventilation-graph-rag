"""Shared project-level error views."""

from django.http import JsonResponse


def csrf_failure(request, reason=""):
    message = "CSRF 校验失败，请刷新页面后重试。"
    if reason and "Origin checking failed" in reason:
        message = (
            "CSRF 来源校验失败：当前前端地址未加入后端信任来源。"
            "请把该地址加入 DJANGO_CSRF_TRUSTED_ORIGINS 并重启 Django 后端。"
        )
    return JsonResponse(
        {"ok": False, "error": message, "reason": reason},
        status=403,
        json_dumps_params={"ensure_ascii": False},
    )
