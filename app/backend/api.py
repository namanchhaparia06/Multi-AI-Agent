from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal

from app.core.ai_agent import get_response_from_ai_agents
from app.config.settings import settings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)
app = FastAPI(title="MULTI_AI_AGENT")


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[Message]
    allow_search: bool


@app.post("/chat")
def chat_endpoint(request: RequestState):
    logger.info(f"Received request for model : {request.model_name}")

    if request.model_name not in settings.ALLOWED_MODEL_NAMES:
        logger.warning("Invalid model name")
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        # Convert Pydantic models to plain dicts for ai_agent
        msgs = [m.model_dump() for m in request.messages]

        response = get_response_from_ai_agents(
            request.model_name,
            msgs,
            request.allow_search,
            request.system_prompt,
        )

        logger.info(f"Successfully got response from AI Agent {request.model_name}")
        return {"response": response}

    except Exception as e:
        logger.exception("Error while getting AI response")
        raise HTTPException(
            status_code=500,
            detail=str(CustomException("Failed to get AI response", error_detail=e)),
        )
