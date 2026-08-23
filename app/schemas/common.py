from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
