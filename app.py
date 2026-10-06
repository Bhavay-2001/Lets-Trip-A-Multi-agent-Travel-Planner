import os
from pathlib import Path
import traceback
import uvicorn
import nest_asyncio

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent

# This allows nested event loops for async calls in FastAPI
nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Let's Trip - AI Travel Planner Agent",
    description = "LangGraph based Multi-Agent Travel Planner."
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

## Pydantic Schema
class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


## routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_msg = request_data.message.strip()

        if not user_msg:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_travel_agent(
            user_input=user_msg,
            thread_id=request_data.thread_id
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "weather_results": result["weather_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "AVIATIONSTACK_API_KEY",
    "TAVILY_API_KEY",
    "POSTGRES_DATABASE_URL",
]

@app.get("/health")
async def health_check():
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    
    if missing_vars:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "unhealthy",
                "message": "Missing required environment configuration",
                "missing": missing_vars,
            },
        )
    
    return {
        "status": "healthy",
        "message": "All API keys and Database URL are configured",
    }



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )