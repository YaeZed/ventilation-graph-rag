"""URL routing for the ventilation web backend."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path


urlpatterns = [
    path("api/chat/", include("chat.urls")),
    path("api/users/", include("users.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
