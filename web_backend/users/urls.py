"""User module API routes."""

from django.urls import path

from . import views


urlpatterns = [
    path("auth/register/", views.register_view, name="user_register"),
    path("auth/login/", views.login_view, name="user_login"),
    path("auth/logout/", views.logout_view, name="user_logout"),
    path("me/", views.me_view, name="user_me"),
    path("profile/", views.profile_view, name="user_profile"),
    path("conversations/", views.conversations_view, name="user_conversations"),
    path("conversations/sync/", views.sync_conversations_view, name="user_conversations_sync"),
    path("conversations/<str:client_id>/delete/", views.delete_conversation_view, name="user_conversation_delete"),
]
