# Generated for P4 team permissions and statistics.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0002_conversationattachment"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Team",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=80)),
                ("description", models.CharField(blank=True, max_length=240)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "created_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="created_teams",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "team",
                "verbose_name_plural": "teams",
                "indexes": [
                    models.Index(fields=["created_by", "-updated_at"], name="users_team_created_8adda7_idx"),
                ],
            },
        ),
        migrations.CreateModel(
            name="TeamMembership",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "role",
                    models.CharField(
                        choices=[("owner", "Owner"), ("admin", "Admin"), ("member", "Member")],
                        default="member",
                        max_length=16,
                    ),
                ),
                ("joined_at", models.DateTimeField(auto_now_add=True)),
                (
                    "team",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memberships",
                        to="users.team",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="team_memberships",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "team membership",
                "verbose_name_plural": "team memberships",
                "indexes": [
                    models.Index(fields=["user", "role"], name="users_teamm_user_id_850184_idx"),
                    models.Index(fields=["team", "role"], name="users_teamm_team_id_867024_idx"),
                ],
                "constraints": [
                    models.UniqueConstraint(fields=("team", "user"), name="unique_team_member"),
                ],
            },
        ),
        migrations.AddField(
            model_name="conversationrecord",
            name="team",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="conversation_records",
                to="users.team",
            ),
        ),
        migrations.AddIndex(
            model_name="conversationrecord",
            index=models.Index(fields=["team", "-client_updated_at"], name="users_conve_team_id_3a4db0_idx"),
        ),
        migrations.AddIndex(
            model_name="conversationrecord",
            index=models.Index(fields=["team", "is_archived"], name="users_conve_team_id_50355d_idx"),
        ),
    ]
