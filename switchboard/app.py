"""Switchboard: an AI that makes real phone calls for you.

You type one task ("book a table for 4 at 7", or "find the cheapest oil change
near me"). Switchboard turns it into a clear calling objective, dials the
number(s) — in parallel for call-arounds — has a real conversation, then reads
the transcripts back as a ranked comparison.

Architecture
------------
  Gemini  -> the brains: vague task -> {objective, questions}; transcript -> data
  Bland   -> the voice: actually dials and talks (swap-in via CALL_PROVIDER)
  mock    -> a scripted offline call so you can build + demo with no keys

Endpoints
  POST /api/plan          task            -> objective + questions to ask
  POST /api/call          number + plan   -> starts a call, returns call_id
  GET  /api/call/{id}     poll            -> status + live transcript
  POST /api/extract       transcript      -> structured result (price, yes/no…)

Run:
    pip install -r requirements.txt
    cp .env.example .env        # CALL_PROVIDER=mock works with zero keys
    uvicorn app:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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
BLAND_API_KEY = os.environ.get("BLAND_API_KEY", "")
CALL_PROVIDER = os.environ.get("CALL_PROVIDER", "mock").lower()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Switchboard", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# --------------------------------------------------------------------------- #
# Gemini (the brains)
# --------------------------------------------------------------------------- #
async def gemini_json(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        # Degrade gracefully so the mock demo works with zero keys.
        return {}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.5,
            "responseMimeType": "application/json",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(GEMINI_URL, params={"key": GEMINI_API_KEY}, json=body)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Call providers (the voice)
# --------------------------------------------------------------------------- #
# In-memory store of mock calls so we can simulate a live transcript.
_MOCK_CALLS: Dict[str, dict] = {}

_MOCK_SCRIPT = [
    ("assistant", "Hi! I'm calling to book a table for four people tonight at 7pm. Do you have availability?"),
    ("user", "Let me check… yes, 7pm works for a party of four."),
    ("assistant", "Perfect. Could I get that under the name Alex?"),
    ("user", "Sure, Alex, party of four at 7. You're all set."),
    ("assistant", "Wonderful, thank you so much. Have a great night!"),
]


async def bland_start(number: str, task: str, first_line: str, record: bool) -> str:
    if not BLAND_API_KEY:
        raise HTTPException(503, "BLAND_API_KEY not set. Use CALL_PROVIDER=mock to demo offline.")
    payload = {
        "phone_number": number,
        "task": task,
        "first_sentence": first_line or None,
        "wait_for_greeting": True,
        "record": record,  # default False -> respects two-party consent
        "max_duration": 8,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.bland.ai/v1/calls",
            headers={"authorization": BLAND_API_KEY},
            json={k: v for k, v in payload.items() if v is not None},
        )
    if r.status_code >= 300:
        raise HTTPException(502, f"Bland error {r.status_code}: {r.text[:300]}")
    data = r.json()
    cid = data.get("call_id")
    if not cid:
        raise HTTPException(502, f"Bland gave no call_id: {data}")
    return cid


async def bland_status(call_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api.bland.ai/v1/calls/{call_id}",
            headers={"authorization": BLAND_API_KEY},
        )
    if r.status_code >= 300:
        raise HTTPException(502, f"Bland error {r.status_code}: {r.text[:300]}")
    d = r.json()
    turns = [
        {"speaker": "assistant" if t.get("user") == "assistant" else "caller",
         "text": t.get("text", "")}
        for t in d.get("transcripts", [])
    ]
    done = bool(d.get("completed")) or d.get("status") == "completed"
    return {"status": "completed" if done else "in-progress", "transcript": turns}


def mock_start(number: str, task: str) -> str:
    cid = "mock_" + uuid.uuid4().hex[:10]
    _MOCK_CALLS[cid] = {"started": time.time(), "task": task, "number": number}
    return cid


def mock_status(call_id: str) -> dict:
    call = _MOCK_CALLS.get(call_id)
    if not call:
        raise HTTPException(404, "Unknown call id")
    # Reveal one new line of the script roughly every 1.5s, like a live call.
    elapsed = time.time() - call["started"]
    reveal = min(len(_MOCK_SCRIPT), int(elapsed // 1.5) + 1)
    turns = [{"speaker": s, "text": t} for s, t in _MOCK_SCRIPT[:reveal]]
    done = reveal >= len(_MOCK_SCRIPT)
    return {"status": "completed" if done else "in-progress", "transcript": turns}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class PlanRequest(BaseModel):
    task: str


class CallRequest(BaseModel):
    number: str
    objective: str
    questions: List[str] = []
    first_line: str = ""
    label: str = ""            # e.g. business name, for the comparison table
    record: bool = False
    provider: Optional[str] = None  # override CALL_PROVIDER per call ("mock"/"bland")


class ExtractRequest(BaseModel):
    objective: str
    questions: List[str] = []
    transcript: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": CALL_PROVIDER,
        "gemini_key": bool(GEMINI_API_KEY),
        "bland_key": bool(BLAND_API_KEY),
    }


@app.post("/api/plan")
async def plan(req: PlanRequest):
    """Turn a vague request into a concrete calling objective + questions."""
    prompt = (
        "You are an assistant that makes phone calls on someone's behalf. "
        f'The user wants: "{req.task}". '
        "Produce a concise plan for the phone call. "
        "Respond as JSON: {"
        '"objective": "one sentence the caller is trying to achieve", '
        '"questions": ["specific question 1 to ask", "question 2", ...], '
        '"first_line": "a natural, polite opening sentence the AI says when the call connects"'
        "}"
    )
    result = await gemini_json(prompt)
    if not result:
        # Offline fallback so the mock demo always has a plan.
        result = {
            "objective": req.task,
            "questions": ["Do you have availability?", "What is the price?"],
            "first_line": f"Hi! I'm calling about {req.task}.",
        }
    return result


@app.post("/api/call")
async def call(req: CallRequest):
    """Start a single phone call. Returns a call_id to poll."""
    provider = (req.provider or CALL_PROVIDER).lower()
    task = req.objective
    if req.questions:
        task += " Make sure to find out: " + "; ".join(req.questions) + "."

    if provider == "mock":
        cid = mock_start(req.number, task)
    else:
        cid = await bland_start(req.number, task, req.first_line, req.record)

    return {"call_id": cid, "provider": provider, "label": req.label or req.number}


@app.get("/api/call/{provider}/{call_id}")
async def call_status(provider: str, call_id: str):
    """Poll a call's status + live transcript."""
    if provider == "mock" or call_id.startswith("mock_"):
        return mock_status(call_id)
    return await bland_status(call_id)


@app.post("/api/extract")
async def extract(req: ExtractRequest):
    """Read a finished transcript into a structured result for the table."""
    q = "; ".join(req.questions) if req.questions else "the key outcome"
    prompt = (
        f"Objective of the call: {req.objective}\n"
        f"Questions we wanted answered: {q}\n"
        f"Full transcript:\n{req.transcript}\n\n"
        "Summarise the outcome. Respond as JSON: {"
        '"success": true/false, '
        '"summary": "one short sentence on what happened", '
        '"answers": {"<question topic>": "<answer or \'unknown\'>"}, '
        '"price": "price if mentioned else null", '
        '"availability": "availability if mentioned else null"'
        "}"
    )
    result = await gemini_json(prompt)
    if not result:
        result = {"success": True, "summary": "Call completed.", "answers": {},
                  "price": None, "availability": None}
    return result


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
