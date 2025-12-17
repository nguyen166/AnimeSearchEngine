import os
import yaml
import uuid
import time
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from schemas import ModelsResponse, Model, EmbeddingResponse
from services.clip import clip_service
from configs.config import CONFIG
from schemas import HealthStatus, EmbeddingRequest
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    return HealthStatus(status="ok")

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    model_cards = []
    for model_config in CONFIG["models"].values():
        model_cards.append(Model(
            id=model_config["name"],
            object="model",
            created=int(time.time()),
            owned_by="user",
        ))
    return ModelsResponse(
        object="list",
        data=model_cards,
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for texts and/or images."""
    try:
        hosted_models = CONFIG["models"].keys()
        if request.model not in hosted_models:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
        texts = request.texts
        b64_images = request.b64_images
        
        if not texts and not b64_images:
            return EmbeddingResponse(
                object="list",
                data=[],
                model=request.model
            )
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Select the appropriate service based on model
        if request.model == "clip":
            service = clip_service
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
        
        # Process base64 images
        images = []
        if b64_images:
            images = await service.process_b64_images(b64_images)
        
        # Add task to queue
        await service.add_task(task_id, texts, images)
        
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            task_status = service.get_task_status(task_id)
            if task_status and task_status['status'] == 'completed':
                embeddings = task_status['embeddings']
                return EmbeddingResponse(
                    object="list",
                    data=[
                        {
                            "embedding": embedding,
                            "index": idx,
                            "object": "embedding"
                        }
                        for idx, embedding in enumerate(embeddings)
                    ],
                    model=request.model
                )
            elif task_status and task_status['status'] == 'failed':
                raise HTTPException(status_code=500, detail=f"Task failed: {task_status['error']}")
            
            await asyncio.sleep(0.1)
        
        raise HTTPException(status_code=408, detail="Request timeout")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    HOST = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=HOST, port=PORT)