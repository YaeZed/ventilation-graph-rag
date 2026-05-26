"""URL routing for the ventilation web backend."""

from django.urls import include, path


urlpatterns = [
    path("api/chat/", include("chat.urls")),
    path("api/users/", include("users.urls")),
]
