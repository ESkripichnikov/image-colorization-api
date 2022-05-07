from __future__ import annotations

from fastapi import FastAPI
from routers import admin_only, users

app = FastAPI(
    title='Image Colorizer',
    description='Colorize black and white pictures by using the Deep Learning technology. Support batch process.',
    contact={'email': 'skrijeka14@gmail.com'},
    version='0.0.1',
    servers=[{'url': 'http://127.0.0.1:8000'}],
)
app.include_router(users.router)
app.include_router(admin_only.router)
