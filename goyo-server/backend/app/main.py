from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, devices, profile  # profile 추가
from app.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="GOYO Backend API",
    description="AI-Based Active Noise Control System",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(devices.router)
app.include_router(profile.router)  # 추가

@app.get("/")
def root():
    return {
        "message": "GOYO Backend API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)