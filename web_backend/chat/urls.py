"""Chat API routes."""

from django.urls import path

from . import views


urlpatterns = [
    path("", views.chat, name="chat"),
    path("upload/", views.chat_upload, name="chat_upload"),
    path("stream/", views.chat_stream, name="chat_stream"),
]

