from fastapi import APIRouter

from app.routes import products, search

api_router = APIRouter()
api_router.include_router(products.router)
api_router.include_router(search.router)
