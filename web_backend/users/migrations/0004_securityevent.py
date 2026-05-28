# Generated for P5 production account security.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0003_team_membership_conversation_team"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="SecurityEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("username", models.CharField(blank=True, max_length=150)),
                (
                    "event_type",
                    models.CharField(
                        choices=[
                            ("register", "Register"),
                            ("register_throttled", "Register throttled"),
                            ("password_rejected", "Password rejected"),
                            ("login_success", "Login success"),
                            ("login_failure", "Login failure"),
                            ("login_throttled", "Login throttled"),
                            ("logout", "Logout"),
                        ],
                        max_length=32,
                    ),
                ),
                ("ip_address", models.CharField(blank=True, max_length=45)),
                ("user_agent", models.CharField(blank=True, max_length=240)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="security_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "security event",
                "verbose_name_plural": "security events",
                "indexes": [
                    models.Index(fields=["user", "-created_at"], name="users_secur_user_id_f93b37_idx"),
                    models.Index(fields=["event_type", "-created_at"], name="users_secur_event_t_e5448a_idx"),
                    models.Index(fields=["ip_address", "-created_at"], name="users_secur_ip_addr_ee069f_idx"),
                ],
            },
        ),
    ]
