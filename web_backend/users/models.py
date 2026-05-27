"""Models for lightweight account settings and conversation persistence."""

from __future__ import annotations

from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    nickname = models.CharField(max_length=32, blank=True)
    avatar_text = models.CharField(max_length=4, blank=True)
    settings = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "user profile"
        verbose_name_plural = "user profiles"

    def __str__(self) -> str:
        return self.nickname or self.user.username


class ConversationRecord(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conversation_records",
    )
    client_id = models.CharField(max_length=80)
    title = models.CharField(max_length=120)
    messages = models.JSONField(default=list, blank=True)
    scene_type = models.CharField(max_length=120, blank=True)
    hazard_level = models.CharField(max_length=80, blank=True)
    is_archived = models.BooleanField(default=False)
    preview_image_url = models.TextField(blank=True)
    is_title_manual = models.BooleanField(default=False)
    client_created_at = models.DateTimeField(null=True, blank=True)
    client_updated_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "client_id"], name="unique_user_conversation_client_id")
        ]
        indexes = [
            models.Index(fields=["user", "-client_updated_at"]),
            models.Index(fields=["user", "is_archived"]),
        ]
        verbose_name = "conversation record"
        verbose_name_plural = "conversation records"

    def __str__(self) -> str:
        return f"{self.user_id}:{self.title}"


class ConversationAttachment(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conversation_attachments",
    )
    conversation = models.ForeignKey(
        ConversationRecord,
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    message_client_id = models.CharField(max_length=80, blank=True)
    file = models.FileField(upload_to="conversation_attachments/%Y/%m/%d/")
    original_name = models.CharField(max_length=160)
    mime_type = models.CharField(max_length=80)
    size = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "conversation", "-created_at"]),
            models.Index(fields=["user", "message_client_id"]),
        ]
        verbose_name = "conversation attachment"
        verbose_name_plural = "conversation attachments"

    def __str__(self) -> str:
        return f"{self.user_id}:{self.original_name}"
