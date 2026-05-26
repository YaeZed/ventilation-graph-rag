# Generated for the ventilation user module.

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="UserProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("nickname", models.CharField(blank=True, max_length=32)),
                ("avatar_text", models.CharField(blank=True, max_length=4)),
                ("settings", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="profile",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "user profile",
                "verbose_name_plural": "user profiles",
            },
        ),
        migrations.CreateModel(
            name="ConversationRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("client_id", models.CharField(max_length=80)),
                ("title", models.CharField(max_length=120)),
                ("messages", models.JSONField(blank=True, default=list)),
                ("scene_type", models.CharField(blank=True, max_length=120)),
                ("hazard_level", models.CharField(blank=True, max_length=80)),
                ("is_archived", models.BooleanField(default=False)),
                ("preview_image_url", models.TextField(blank=True)),
                ("is_title_manual", models.BooleanField(default=False)),
                ("client_created_at", models.DateTimeField(blank=True, null=True)),
                ("client_updated_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="conversation_records",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "conversation record",
                "verbose_name_plural": "conversation records",
                "indexes": [
                    models.Index(fields=["user", "-client_updated_at"], name="users_conve_user_id_699d52_idx"),
                    models.Index(fields=["user", "is_archived"], name="users_conve_user_id_26fe8f_idx"),
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("user", "client_id"), name="unique_user_conversation_client_id"
                    )
                ],
            },
        ),
    ]
