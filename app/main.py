# FastAPI entry point: — it will handle a health check and prepare future endpoints if needed.

# app/main.py

from fastapi import FastAPI
from app.bot import start_bot

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Goal-Based Trading Alert Bot...")
    await start_bot()

@app.get("/")
async def root():
    return {"message": "Goal-Based Trading Alert is running"}