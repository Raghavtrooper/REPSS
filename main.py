from fastapi import FastAPI
from resume_upload import router as resume_router

app = FastAPI()

app.include_router(resume_router)
