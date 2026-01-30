from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pipeline import YouTubeSentimentPipeline

app = FastAPI()

pipeline = YouTubeSentimentPipeline(
    model_path="models/go_emotions_model.h5",
    tokenizer_path="models/tokenizer.pickle"
)

class YouTubeRequest(BaseModel):
    youtube_url: str

@app.post("/analyze")
def analyze(request: YouTubeRequest):
    try:
        return pipeline.analyze_youtube_video(request.youtube_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
