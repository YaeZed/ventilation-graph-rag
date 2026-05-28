"""User module API routes."""

from django.urls import path

from . import views


urlpatterns = [
    path("auth/csrf/", views.csrf_view, name="user_csrf"),
    path("auth/register/", views.register_view, name="user_register"),
    path("auth/login/", views.login_view, name="user_login"),
    path("auth/logout/", views.logout_view, name="user_logout"),
    path("me/", views.me_view, name="user_me"),
    path("profile/", views.profile_view, name="user_profile"),
    path("security/events/", views.security_events_view, name="user_security_events"),
    path("teams/", views.teams_view, name="user_teams"),
    path("teams/<int:team_id>/", views.team_detail_view, name="user_team_detail"),
    path("teams/<int:team_id>/members/", views.team_members_view, name="user_team_members"),
    path(
        "teams/<int:team_id>/conversations/",
        views.team_conversations_view,
        name="user_team_conversations",
    ),
    path(
        "teams/<int:team_id>/members/<int:user_id>/",
        views.team_member_detail_view,
        name="user_team_member_detail",
    ),
    path("conversations/", views.conversations_view, name="user_conversations"),
    path("conversations/sync/", views.sync_conversations_view, name="user_conversations_sync"),
    path("conversations/<str:client_id>/delete/", views.delete_conversation_view, name="user_conversation_delete"),
    path("conversations/<str:client_id>/team/", views.conversation_team_view, name="user_conversation_team"),
    path("stats/summary/", views.stats_summary_view, name="user_stats_summary"),
    path("stats/trends/", views.stats_trends_view, name="user_stats_trends"),
    path("stats/hazards/", views.stats_hazards_view, name="user_stats_hazards"),
    path(
        "conversations/<str:client_id>/attachments/upload/",
        views.upload_attachment_view,
        name="user_attachment_upload",
    ),
    path(
        "conversations/<str:client_id>/attachments/",
        views.attachments_view,
        name="user_attachments",
    ),
    path("attachments/<int:attachment_id>/delete/", views.delete_attachment_view, name="user_attachment_delete"),
]
