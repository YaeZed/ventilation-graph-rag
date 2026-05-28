"""Models for lightweight account settings and conversation persistence."""

from __future__ import annotations

from django.conf import settings
from django.db import models


class Team(models.Model):
    name = models.CharField(max_length=80)
    description = models.CharField(max_length=240, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_teams",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["created_by", "-updated_at"]),
        ]
        verbose_name = "team"
        verbose_name_plural = "teams"

    def __str__(self) -> str:
        return self.name


class TeamMembership(models.Model):
    ROLE_OWNER = "owner"
    ROLE_ADMIN = "admin"
    ROLE_MEMBER = "member"
    ROLE_CHOICES = [
        (ROLE_OWNER, "Owner"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_MEMBER, "Member"),
    ]

    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="memberships")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="team_memberships",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default=ROLE_MEMBER)
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["team", "user"], name="unique_team_member")
        ]
        indexes = [
            models.Index(fields=["user", "role"]),
            models.Index(fields=["team", "role"]),
        ]
        verbose_name = "team membership"
        verbose_name_plural = "team memberships"

    def __str__(self) -> str:
        return f"{self.team_id}:{self.user_id}:{self.role}"


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


class SecurityEvent(models.Model):
    EVENT_REGISTER = "register"
    EVENT_REGISTER_THROTTLED = "register_throttled"
    EVENT_PASSWORD_REJECTED = "password_rejected"
    EVENT_LOGIN_SUCCESS = "login_success"
    EVENT_LOGIN_FAILURE = "login_failure"
    EVENT_LOGIN_THROTTLED = "login_throttled"
    EVENT_LOGOUT = "logout"
    EVENT_CHOICES = [
        (EVENT_REGISTER, "Register"),
        (EVENT_REGISTER_THROTTLED, "Register throttled"),
        (EVENT_PASSWORD_REJECTED, "Password rejected"),
        (EVENT_LOGIN_SUCCESS, "Login success"),
        (EVENT_LOGIN_FAILURE, "Login failure"),
        (EVENT_LOGIN_THROTTLED, "Login throttled"),
        (EVENT_LOGOUT, "Logout"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="security_events",
    )
    username = models.CharField(max_length=150, blank=True)
    event_type = models.CharField(max_length=32, choices=EVENT_CHOICES)
    ip_address = models.CharField(max_length=45, blank=True)
    user_agent = models.CharField(max_length=240, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["event_type", "-created_at"]),
            models.Index(fields=["ip_address", "-created_at"]),
        ]
        verbose_name = "security event"
        verbose_name_plural = "security events"

    def __str__(self) -> str:
        return f"{self.event_type}:{self.username or self.user_id}"


class ConversationRecord(models.Model):
    team = models.ForeignKey(
        Team,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="conversation_records",
    )
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
            models.Index(fields=["team", "-client_updated_at"]),
            models.Index(fields=["team", "is_archived"]),
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
