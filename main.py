import os
import json
import httpx
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from google import genai
from fastapi.responses import JSONResponse

CACHE_FILE = "gemini_models_cache.json"
# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("app")

# --- Env & Clients ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Google GenAI client if key exists
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI()
client = httpx.AsyncClient(timeout=None)

@app.get("/v1/models")
async def list_models():
    """List models from Gemini if available, else from OpenRouter."""
    logger.info("📡 GET /v1/models called")

    # --- check if cache exists ---
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.info(f"📂 Loaded {len(cached_data['data'])} models from cache")
                return JSONResponse(content=cached_data, status_code=200)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}, falling back to live fetch")

    # --- fetch from Gemini ---
    if genai_client:
        try:
            logger.info("Using Gemini client to list models...")
            models = genai_client.models.list()
            mapped_models = []

            for m in models:
                canonical_slug = m.name.replace("models/", "")
                mapped_models.append({
                    "id": canonical_slug,
                    "canonical_slug": canonical_slug,
                    "name": getattr(m, "display_name", canonical_slug),
                    "created": 0,
                    "description": getattr(m, "description", ""),
                    "context_length": getattr(m, "input_token_limit", 4096),
                    "architecture": {
                        "modality": "text->text",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "tokenizer": "Gemini",
                        "instruct_type": None
                    },
                    "top_provider": {
                        "context_length": getattr(m, "input_token_limit", 4096),
                        "max_completion_tokens": getattr(m, "output_token_limit", 4096),
                        "is_moderated": False
                    },
                    "per_request_limits": None,
                    "supported_parameters": getattr(m, "supported_generation_methods", [])
                })

            response_data = {"object": "list", "data": mapped_models}

            # --- save to cache ---
            try:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Cached {len(mapped_models)} models to {CACHE_FILE}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to write cache: {e}")

            logger.info(f"✅ Retrieved {len(mapped_models)} models from Gemini")
            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"❌ Gemini model list failed: {e}", exc_info=True)
            return JSONResponse(content={"error": str(e)}, status_code=500)

    # --- fallback: OpenRouter ---
    try:
        logger.info("Using OpenRouter to list models...")
        url = f"{OPENROUTER_BASE_URL}/models"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        r = await client.get(url, headers=headers)
        data = r.json()
        filtered = data.get("data", [])
        logger.info(f"✅ Retrieved {len(filtered)} models from OpenRouter")
        return JSONResponse(content={"object": "list", "data": filtered}, status_code=r.status_code)
    except Exception as e:
        logger.error(f"❌ OpenRouter model list failed: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


def convert_openai_messages_to_gemini_contents(messages: list):
    """Convert OpenAI-style messages to Gemini contents format."""
    gemini_contents = []
    for msg in messages:
        role = msg.get("role")
        text = msg.get("content", "")
        if not text:
            continue
        gemini_contents.append({
            "role": "user" if role == "user" else ("model" if role == "assistant" else "system"),
            "parts": [{"text": text}]
        })
    return gemini_contents

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Stream chat completions from Gemini if available, else OpenRouter."""
    logger.info("📡 POST /v1/chat/completions called")

    try:
        body = await request.json()
        logger.debug(f"Request body: {json.dumps(body, indent=2)}")
    except Exception as e:
        logger.error(f"❌ Failed to parse JSON body: {e}")
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    if genai_client:
        try:
            model = body.get("model", "gemini-2.5-flash")
            contents = body.get("messages") or body.get("contents")
            if not contents:
                logger.warning("⚠️ Missing 'messages' or 'contents'")
                return JSONResponse(content={"error": "Missing 'messages' or 'contents'"}, status_code=400)

            if isinstance(contents, list) and isinstance(contents[0], dict):
                contents = convert_openai_messages_to_gemini_contents(contents)
            async def gemini_event_generator():
                try:
                    logger.info(f"🔄 Streaming response from Gemini model: {model}")

                    stream = genai_client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                    )
                    for event in stream:
                        if hasattr(event, "text"):
                            chunk = {
                                "id": "chatcmpl-gemini",
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "delta": {"content": event.text},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    yield "data: [DONE]\n\n"
                    logger.info("✅ Gemini stream completed")

                except Exception as e:
                    logger.error(f"❌ Gemini streaming failed: {e}", exc_info=True)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(gemini_event_generator(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"❌ Gemini chat failed: {e}", exc_info=True)
            return JSONResponse(content={"error": str(e)}, status_code=500)

    # --- fallback: OpenRouter (streaming mode) ---
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    body["stream"] = True

    async def event_generator():
        try:
            logger.info(f"🔄 Streaming response from OpenRouter model: {body.get('model')}")
            async with client.stream("POST", url, headers=headers, json=body) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    logger.debug(f"OpenRouter chunk: {line[:50]}...")
                    yield line + "\n\n"
                logger.info("✅ OpenRouter stream completed")
                yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"❌ OpenRouter streaming failed: {e}", exc_info=True)
            err = {"error": f"Streaming failed: {str(e)}"}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
