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
VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "")
VAPI_PHONE_NUMBER_ID = os.environ.get("VAPI_PHONE_NUMBER_ID", "")
VAPI_VOICE = os.environ.get("VAPI_VOICE", "alloy")
VAPI_MODEL = os.environ.get("VAPI_MODEL", "gpt-4o")
CALL_PROVIDER = os.environ.get("CALL_PROVIDER", "mock").lower()

# Public base URL of THIS server, so Vapi can stream live transcript webhooks
# back to us. Render sets RENDER_EXTERNAL_URL automatically.
PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or ""
).rstrip("/")

# language name -> (transcriber code) for multilingual calls
LANG_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "mandarin": "zh", "chinese": "zh", "cantonese": "zh", "vietnamese": "vi",
    "hindi": "hi", "korean": "ko", "japanese": "ja", "tagalog": "tl",
    "portuguese": "pt", "italian": "it", "russian": "ru", "arabic": "ar",
}

# Discovery: find local (often phone-only) businesses to call.
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")
DISCOVER_PROVIDER = os.environ.get(
    "DISCOVER_PROVIDER", "places" if GOOGLE_PLACES_API_KEY else "mock"
).lower()

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

# Live transcript accumulated from Vapi webhooks: call_id -> {turns:[...], done:bool}
_VAPI_LIVE: Dict[str, dict] = {}

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


async def vapi_start(number: str, task: str, first_line: str, language: str = "") -> str:
    if not VAPI_API_KEY:
        raise HTTPException(503, "VAPI_API_KEY not set. Use CALL_PROVIDER=mock to demo offline.")
    if not VAPI_PHONE_NUMBER_ID:
        raise HTTPException(
            503,
            "VAPI_PHONE_NUMBER_ID not set. Create/import a number in the Vapi dashboard "
            "and put its id in .env.",
        )
    lang = (language or "English").strip()
    multilingual = lang.lower() not in ("", "english", "en", "auto")
    system_prompt = (
        "You are a warm, natural-sounding person making a phone call on someone's "
        "behalf. You are NOT a robot reading a script — you are having a real, "
        "two-way conversation, and you listen and react to what the other person says.\n\n"
        f"YOUR GOAL ON THIS CALL:\n{task}\n\n"
        "HOW TO TALK:\n"
        "- Open with a brief, friendly greeting and say why you're calling in ONE sentence, "
        "then stop and let them respond.\n"
        "- Actually listen. Respond directly to what they say — acknowledge it before moving on.\n"
        "- Ask only ONE thing at a time. Never list all your questions in a single breath.\n"
        "- Keep every turn short and conversational, just one or two sentences.\n"
        "- If they ask you something, answer naturally and helpfully.\n"
        "- If they need a moment or put you on hold, be patient and polite.\n"
        "- Once you've gotten what you need (or it's clearly not possible), confirm the key "
        "details back to them, thank them warmly, and say goodbye.\n\n"
        "Sound like a relaxed, polite human — use natural phrases like \"sure\", \"got it\", "
        "\"perfect\", \"no problem\". Do not mention that you are an AI unless you are asked directly."
    )
    if multilingual:
        system_prompt += (
            f"\n\nIMPORTANT: Conduct this ENTIRE phone call in {lang}. Speak only {lang} "
            f"the whole time, like a native {lang} speaker."
        )
    assistant = {
        "model": {
            "provider": "openai",
            "model": VAPI_MODEL,
            "temperature": 0.7,
            "messages": [{"role": "system", "content": system_prompt}],
        },
        "voice": {"provider": "openai", "voiceId": VAPI_VOICE},
    }
    if multilingual:
        # let the model open in the target language, and transcribe the callee in it too
        assistant["firstMessageMode"] = "assistant-speaks-first-with-model-generated-message"
        assistant["transcriber"] = {
            "provider": "deepgram", "model": "nova-2",
            "language": LANG_CODES.get(lang.lower(), "en"),
        }
    else:
        assistant["firstMessage"] = first_line or "Hi there!"
    # stream live transcript back to us if we know our public URL
    if PUBLIC_BASE_URL:
        assistant["serverUrl"] = f"{PUBLIC_BASE_URL}/api/vapi/webhook"
        assistant["serverMessages"] = ["conversation-update", "status-update", "end-of-call-report"]
    body = {
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": number},
        "assistant": assistant,
    }

    async def _post(b):
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.post(
                "https://api.vapi.ai/call",
                headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
                json=b,
            )

    r = await _post(body)
    # If Vapi rejects the optional live-transcript webhook fields, NEVER let that
    # block the call — strip them and retry so the call still goes through.
    if r.status_code >= 300 and ("serverUrl" in assistant):
        assistant.pop("serverUrl", None)
        assistant.pop("serverMessages", None)
        r = await _post(body)
    if r.status_code >= 300:
        raise HTTPException(502, f"Vapi error {r.status_code}: {r.text[:300]}")
    cid = r.json().get("id")
    if not cid:
        raise HTTPException(502, f"Vapi gave no call id: {r.text[:200]}")
    return cid


async def vapi_status(call_id: str) -> dict:
    # Prefer the live webhook stream if we've started receiving it — that's what
    # makes the transcript type out word-by-word during the call.
    live = _VAPI_LIVE.get(call_id)
    if live and live["turns"] and not live["done"]:
        return {"status": "in-progress", "transcript": live["turns"]}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api.vapi.ai/call/{call_id}",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
        )
    if r.status_code >= 300:
        raise HTTPException(502, f"Vapi error {r.status_code}: {r.text[:300]}")
    d = r.json()
    artifact = d.get("artifact") or {}
    msgs = artifact.get("messages") or d.get("messages") or []
    turns = []
    for m in msgs:
        role = (m.get("role") or "").lower()
        if role == "system":
            continue
        text = m.get("message") or m.get("content") or ""
        if not text:
            continue
        turns.append({
            "speaker": "assistant" if role in ("assistant", "bot") else "caller",
            "text": text,
        })
    done = d.get("status") == "ended"
    # If the GET hasn't populated the transcript yet but our webhook captured it, use that.
    if not turns and live and live["turns"]:
        turns = live["turns"]
    if live and live["done"]:
        done = True
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
# Discovery (find local, often phone-only, businesses to call)
# --------------------------------------------------------------------------- #
def _e164(raw: str) -> str:
    """Normalise a phone string to a dial-able form (strip spaces/punctuation)."""
    if not raw:
        return ""
    keep = "".join(c for c in raw if c.isdigit() or c == "+")
    if keep and not keep.startswith("+") and len(keep) == 10:
        keep = "+1" + keep  # assume US if no country code
    return keep


_MOCK_PLACES = [
    {"name": "Mike's Quick Lube", "number": "+14085550111", "website": None,
     "address": "120 S Main St, Milpitas, CA", "rating": 4.6},
    {"name": "Valley Auto Care", "number": "+14085550122", "website": None,
     "address": "455 Calaveras Blvd, Milpitas, CA", "rating": 4.3},
    {"name": "SpeedyOil Express", "number": "+14085550133", "website": "speedyoil.example",
     "address": "78 Dixon Landing Rd, Milpitas, CA", "rating": 4.1},
    {"name": "Tony's Garage", "number": "+14085550144", "website": None,
     "address": "12 Great Mall Pkwy, Milpitas, CA", "rating": 4.8},
    {"name": "AutoNation Service", "number": "+14085550155", "website": "autonation.example",
     "address": "900 Montague Expy, Milpitas, CA", "rating": 3.9},
]


async def places_discover(query: str, limit: int) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.internationalPhoneNumber,"
            "places.nationalPhoneNumber,places.websiteUri,"
            "places.formattedAddress,places.rating"
        ),
    }
    body = {"textQuery": query, "maxResultCount": min(limit, 10)}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers=headers, json=body,
        )
    if r.status_code >= 300:
        raise HTTPException(502, f"Places error {r.status_code}: {r.text[:300]}")
    out = []
    for p in r.json().get("places", []):
        phone = _e164(p.get("internationalPhoneNumber") or p.get("nationalPhoneNumber") or "")
        if not phone:
            continue  # can't call a place with no number
        out.append({
            "name": (p.get("displayName") or {}).get("text", "Unknown"),
            "number": phone,
            "website": p.get("websiteUri"),
            "address": p.get("formattedAddress", ""),
            "rating": p.get("rating"),
        })
    return out


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
    language: str = ""              # e.g. "Spanish" — conduct the call in this language


class ExtractRequest(BaseModel):
    objective: str
    questions: List[str] = []
    transcript: str


class DiscoverRequest(BaseModel):
    query: str                      # e.g. "oil change"
    location: str = ""              # e.g. "Milpitas, CA"
    limit: int = 5


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
        "vapi_key": bool(VAPI_API_KEY),
        "vapi_number": bool(VAPI_PHONE_NUMBER_ID),
        "discover_provider": DISCOVER_PROVIDER,
        "places_key": bool(GOOGLE_PLACES_API_KEY),
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


@app.post("/api/discover")
async def discover(req: DiscoverRequest):
    """Find local businesses to call. Flags the phone-only ones (no website) —
    the ones you literally can't price any other way."""
    query = req.query if not req.location else f"{req.query} near {req.location}"
    if DISCOVER_PROVIDER == "places" and GOOGLE_PLACES_API_KEY:
        places = await places_discover(query, req.limit)
    else:
        places = _MOCK_PLACES[: req.limit]
    # mark the unindexed ones and sort them first — that's our whole thesis
    for p in places:
        p["phone_only"] = not bool(p.get("website"))
    places.sort(key=lambda p: (not p["phone_only"], -(p.get("rating") or 0)))
    return {"businesses": places, "provider": DISCOVER_PROVIDER}


@app.post("/api/call")
async def call(req: CallRequest):
    """Start a single phone call. Returns a call_id to poll."""
    provider = (req.provider or CALL_PROVIDER).lower()
    task = req.objective
    if req.questions:
        task += " Make sure to find out: " + "; ".join(req.questions) + "."

    if provider == "mock":
        cid = mock_start(req.number, task)
    elif provider == "vapi":
        cid = await vapi_start(req.number, task, req.first_line, req.language)
    else:
        cid = await bland_start(req.number, task, req.first_line, req.record)

    return {"call_id": cid, "provider": provider, "label": req.label or req.number}


@app.get("/api/call/{provider}/{call_id}")
async def call_status(provider: str, call_id: str):
    """Poll a call's status + live transcript."""
    if provider == "mock" or call_id.startswith("mock_"):
        return mock_status(call_id)
    if provider == "vapi":
        return await vapi_status(call_id)
    return await bland_status(call_id)


@app.post("/api/vapi/webhook")
async def vapi_webhook(payload: dict):
    """Vapi posts live events here. We accumulate the transcript per call so the
    dashboard can stream it word-by-word while the call is still happening."""
    msg = payload.get("message") or {}
    call_id = (msg.get("call") or payload.get("call") or {}).get("id")
    if not call_id:
        return {"ok": True}
    live = _VAPI_LIVE.setdefault(call_id, {"turns": [], "done": False})
    mtype = msg.get("type")

    if mtype == "transcript" and msg.get("transcriptType") == "final":
        text = (msg.get("transcript") or "").strip()
        if text:
            role = (msg.get("role") or "").lower()
            live["turns"].append({
                "speaker": "assistant" if role in ("assistant", "bot") else "caller",
                "text": text,
            })
    elif mtype == "conversation-update":
        # rebuild the running transcript from the full message list each update
        turns = []
        for m in (msg.get("messages") or msg.get("conversation") or []):
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            text = (m.get("message") or m.get("content") or "").strip()
            if text:
                turns.append({
                    "speaker": "assistant" if role in ("assistant", "bot") else "caller",
                    "text": text,
                })
        if turns:
            live["turns"] = turns
    elif mtype == "status-update" and msg.get("status") == "ended":
        live["done"] = True
    elif mtype == "end-of-call-report":
        live["done"] = True
    return {"ok": True}


@app.post("/api/extract")
async def extract(req: ExtractRequest):
    """Read a finished transcript into a structured result for the table."""
    q = "; ".join(req.questions) if req.questions else "the key outcome"
    prompt = (
        f"Objective of the call: {req.objective}\n"
        f"Questions we wanted answered: {q}\n"
        f"Full transcript (may be in another language):\n{req.transcript}\n\n"
        "Summarise the outcome FOR THE USER, always in English even if the call "
        "was in another language. Respond as JSON: {"
        '"success": true/false, '
        '"summary": "2-3 sentence recap of how the call went", '
        '"result": "the single bottom-line outcome in one short line, e.g. '
        '\'Booked for 7pm under Alex\' or \'$45, available tomorrow 9am\'", '
        '"answers": {"<question topic>": "<answer or \'unknown\'>"}, '
        '"price": "price if mentioned else null", '
        '"availability": "availability if mentioned else null"'
        "}"
    )
    result = await gemini_json(prompt)
    if not result:
        result = {"success": True, "summary": "Call completed.",
                  "result": "Call completed.", "answers": {},
                  "price": None, "availability": None}
    return result


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
