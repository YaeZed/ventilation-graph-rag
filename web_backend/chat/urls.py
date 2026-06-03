"""Chat API routes."""

from django.urls import path

from . import views


urlpatterns = [
    path("", views.chat, name="chat"),
    path("upload/", views.chat_upload, name="chat_upload"),
    path("stream/", views.chat_stream, name="chat_stream"),
    path("model/test/", views.model_test, name="model_test"),
    path("vision/scenes/", views.vision_scenes, name="vision_scenes"),
    path("vision/evaluate/", views.vision_evaluate, name="vision_evaluate"),
]
