"""Offline task placeholders for future batch processing."""

from ventilation_web.celery import app


@app.task(name="chat.health_check")
def health_check():
    return {"ok": True}

