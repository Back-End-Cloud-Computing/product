from fastapi import APIRouter

from app.routes import products, recommendations, search

api_router = APIRouter()
api_router.include_router(products.router)
api_router.include_router(search.router)
api_router.include_router(recommendations.router)
