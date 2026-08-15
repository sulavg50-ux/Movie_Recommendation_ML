from fastapi import FastAPI
from pydantic import BaseModel
from model.recommender import Recommender
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Movie Recommendation Service")
recommender = Recommender()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    titles: list[str]
    top_k: int = 10


class SimilarRequest(BaseModel):
    title: str
    top_k: int = 10


class RecommendationResponse(BaseModel):
    titles: list[str]


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendRequest):
    titles = recommender.recommend_from_history(req.titles, req.top_k)
    return {"titles": titles}


@app.post("/recommend/similar", response_model=RecommendationResponse)
def recommend_similar(req: SimilarRequest):
    titles = recommender.similar_to(req.title, req.top_k)
    return {"titles": titles}


@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": len(recommender.titles)}