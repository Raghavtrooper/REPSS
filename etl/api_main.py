# api_main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import sys
import os

# Add the parent directory of both 'etl' and 'shared' to the Python path.
# This allows imports like 'from shared.config import ...' to resolve correctly.
# os.path.dirname(__file__) is 'E:\Projects\Raghavbranch\REPSS_raghav\etl'
# os.path.dirname(os.path.dirname(__file__)) will be 'E:\Projects\Raghavbranch\REPSS_raghav'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from your ETL script
# Assuming main_etl.py is in the 'etl' directory, which is now accessible via sys.path
from etl.main_etl import main as run_etl_main

app = FastAPI(
    title="ETL Trigger API",
    description="API to trigger the ETL process for resume data.",
    version="1.0.0",
)

@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the ETL Trigger API. Use /trigger-etl to start the process."}

@app.post("/trigger-etl")
async def trigger_etl():
    """
    Triggers the main ETL process.
    This endpoint will execute the 'main' function from main_etl.py.
    The ETL process can be long-running, so consider asynchronous execution
    or a background task queue for production environments.
    """
    try:
        # Run the ETL main function in a separate thread or process
        # to avoid blocking the FastAPI event loop, especially for long-running tasks.
        # For simplicity, we'll run it directly here, but for production,
        # you might use `asyncio.to_thread` or a task queue like Celery.
        
        # Since main_etl.py is a synchronous script, we'll run it in a thread pool
        # provided by asyncio.to_thread to prevent blocking the Uvicorn worker.
        await asyncio.to_thread(run_etl_main)
        
        return JSONResponse(
            status_code=200,
            content={"message": "ETL process triggered successfully. Check server logs for progress."}
        )
    except Exception as e:
        # Log the full exception for debugging
        print(f"Error triggering ETL process: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger ETL process: {str(e)}"
        )
