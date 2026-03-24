from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from sms_app.routers import auth, catalog, enrollments, grades, students

app = FastAPI(
    title="Student Management System API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 添加 CORS 中间件，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(students.router)
app.include_router(catalog.router)
app.include_router(enrollments.router)
app.include_router(grades.router)

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")


@app.get("/healthz")
def healthz():
    return {"ok": True}

