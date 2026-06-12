"""Fridge -> Feast: scan your fridge, cook what's already there.

A tiny FastAPI backend that proxies Google Gemini (vision + text) so the
browser never sees the API key. Three endpoints power the whole flow:

  POST /api/detect   image  -> list of ingredients it sees
  POST /api/recipes  picks  -> 3 recipes that use ONLY what you have
  POST /api/steps    recipe -> detailed step-by-step "cook mode"

Run locally:
    pip install -r requirements.txt
    cp .env.example .env          # then paste your free Gemini key
    uvicorn app:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Fridge -> Feast", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Gemini helper
# --------------------------------------------------------------------------- #
async def _gemini(parts: list, *, want_json: bool = True) -> dict | str:
    """Call Gemini generateContent with a list of parts (text and/or image).

    When ``want_json`` we ask the model for application/json and parse it,
    so the rest of the app can trust the shape of what comes back.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="No GEMINI_API_KEY set. Copy .env.example to .env and add your key.",
        )

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.6},
    }
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GEMINI_URL, params={"key": GEMINI_API_KEY}, json=body
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini: {exc}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Gemini error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Gemini returned no usable content.")

    if not want_json:
        return text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Models sometimes wrap JSON in ```json ... ```; strip and retry.
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(cleaned)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class DetectRequest(BaseModel):
    image_base64: str          # raw base64, no data: prefix
    mime_type: str = "image/jpeg"


class RecipesRequest(BaseModel):
    ingredients: List[str]
    preference: Optional[str] = None  # e.g. "vegetarian", "quick", "high protein"


class StepsRequest(BaseModel):
    title: str
    ingredients: List[str]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL, "key_set": bool(GEMINI_API_KEY)}


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Look at a fridge/pantry photo and list the edible ingredients."""
    prompt = (
        "You are a kitchen assistant. Look at this photo of a fridge, pantry, "
        "or food on a counter. List every distinct edible ingredient you can "
        "identify. Be generous but realistic. Use simple common names "
        '(e.g. "eggs", "cheddar cheese", "spinach"). Ignore non-food items. '
        'Respond as JSON: {"ingredients": ["item1", "item2", ...]}'
    )
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": req.mime_type, "data": req.image_base64}},
    ]
    result = await _gemini(parts)
    items = result.get("ingredients", []) if isinstance(result, dict) else []
    # de-dup while preserving order, drop blanks
    seen, clean = set(), []
    for it in items:
        key = str(it).strip().lower()
        if key and key not in seen:
            seen.add(key)
            clean.append(str(it).strip())
    return {"ingredients": clean}


@app.post("/api/recipes")
async def recipes(req: RecipesRequest):
    """Suggest 3 recipes that use ONLY the selected ingredients (+ staples)."""
    if not req.ingredients:
        raise HTTPException(status_code=400, detail="Pick at least one ingredient.")

    pref = f" The user prefers: {req.preference}." if req.preference else ""
    prompt = (
        "You are a zero-waste chef. The user has ONLY these ingredients: "
        f"{', '.join(req.ingredients)}." + pref + " Suggest 3 distinct recipes "
        "they can actually make. Prioritise recipes that need NOTHING beyond what "
        "they have, aside from basic staples (salt, pepper, oil, water). "
        "If a recipe needs 1-2 small extra items, list them under 'missing'. "
        "Respond as JSON: {\"recipes\": [{\"title\": str, \"time_minutes\": int, "
        "\"difficulty\": \"easy|medium|hard\", \"description\": str, "
        "\"uses\": [str], \"missing\": [str]}]}"
    )
    result = await _gemini([{"text": prompt}])
    return result if isinstance(result, dict) else {"recipes": []}


@app.post("/api/steps")
async def steps(req: StepsRequest):
    """Full step-by-step cook mode for one chosen recipe."""
    prompt = (
        f"Write clear, beginner-friendly cooking instructions for: {req.title}. "
        f"Assume the cook has: {', '.join(req.ingredients)} plus basic staples. "
        "Respond as JSON: {\"title\": str, \"servings\": int, "
        "\"total_time_minutes\": int, "
        "\"ingredients\": [{\"item\": str, \"amount\": str}], "
        "\"steps\": [{\"instruction\": str, \"minutes\": int}], "
        "\"tips\": [str]}"
    )
    result = await _gemini([{"text": prompt}])
    return result if isinstance(result, dict) else {}


# --------------------------------------------------------------------------- #
# Serve the frontend
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
