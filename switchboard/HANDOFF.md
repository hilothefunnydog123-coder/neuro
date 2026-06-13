# Switchboard — Project Handoff / Memory Dump

> Paste this whole file into a new Claude chat to bring it fully up to speed on
> the project. It captures the concept, architecture, code layout, config,
> deployment, known gotchas, and what's left to build.

---

## 1. What Switchboard is

An AI agent that **makes real phone calls for you**. You type (or speak) a task,
it turns it into a calling objective, **dials the number(s)** — in parallel for
"call-arounds" — has a real two-way conversation, then reads the transcripts
back as a ranked comparison + plain-English summary.

**Two modes:**
- **Do it for me** — one task → one call (book, ask, cancel).
- **Call around** — one task → many calls in parallel → ranked results table →
  one-tap "book the best one."

### The pitch / thesis (use this with judges)
> "The phone is the last unindexed marketplace. Half the local economy — the
> mechanic, the clinic, the mom-and-pop shop — has **no website and no online
> price**. The only way to get a quote is to call, and nobody calls five places.
> Switchboard calls them all at once and hands you the cheapest. It's a search
> engine for everything the internet can't reach."

Built for a hackathon ($10k prize). Local context: Milpitas, CA.

---

## 2. The differentiators (why it's not "just another AI caller")

Most AI callers are **B2B** (businesses calling their customers). Switchboard is
**consumer-side** and adds three things almost nobody else has:
1. **Parallel + comparison** — calls many places at once, returns a ranked table.
2. **Auto-discovery of phone-only businesses** — finds local shops and surfaces
   the ones with *no website* first (the "unindexed" thesis, shown not told).
3. **Outcome over conversation** — ends on a decision (ranked result + book the
   winner), not "look, it can talk."

Plus shipped wow-features: **multilingual calling**, **voice input**,
**live streaming transcript**, **English call summaries**.

---

## 3. Tech stack

- **Backend:** FastAPI (Python 3.11), `uvicorn`, `httpx`.
- **Brains:** Google **Gemini** (`gemini-2.0-flash`) — plans the call + extracts
  structured results from transcripts.
- **Voice:** **Vapi** (vapi.ai) — places the actual call (GPT-4o model + OpenAI
  voice + Deepgram transcriber under the hood). Swappable provider.
- **Discovery:** Google **Places API** (optional; has an offline mock fallback).
- **Frontend:** single static `index.html` (vanilla JS, no framework), polished
  dark "SaaS" theme (Inter font).
- **Deploy:** Render (Docker), repo on GitHub.

### Provider abstraction (important design choice)
Calls go through a provider layer so it's swappable:
- `mock` — scripted offline call that streams a fake transcript. **Zero keys,
  no limits. This is the on-stage safety net.**
- `vapi` — real calls (current production provider).
- `bland` — earlier provider, still in code (account got banned, abandoned).

Set via `CALL_PROVIDER` env var, or per-request `provider` field.

---

## 4. File layout

```
neuro/                      (GitHub repo: hilothefunnydog123-coder/neuro)
├── Dockerfile              # root build — repointed to run switchboard/
├── render.yaml             # root Render blueprint — repointed to switchboard
├── switchboard/
│   ├── app.py              # FastAPI backend (all endpoints + providers)
│   ├── static/index.html   # the entire dashboard UI
│   ├── requirements.txt
│   ├── render.yaml         # (secondary) blueprint
│   ├── Dockerfile?         # no — only root Dockerfile is used
│   ├── README.md
│   ├── .env.example
│   ├── .gitignore          # ignores .env
│   └── .env                # secrets, NOT committed
└── (old stock-forecaster app files at root — abandoned, ignore them)
```

Branch: **`claude/kind-goldberg-u2ibgf`** (all work is here, not `main`).

---

## 5. API endpoints (in switchboard/app.py)

- `GET  /health` → status + which keys/providers are configured.
- `POST /api/plan` `{task}` → `{objective, questions[], first_line}` (Gemini).
- `POST /api/discover` `{query, location, limit}` → `{businesses[]}` with a
  `phone_only` flag; phone-only sorted first. (Places or mock.)
- `POST /api/call` `{number, objective, questions[], first_line, label,
  language, provider?}` → `{call_id, provider, label}`.
- `GET  /api/call/{provider}/{call_id}` → `{status, transcript[]}` (poll).
- `POST /api/vapi/webhook` → Vapi posts live events here; accumulates the
  transcript per call into an in-memory store for live streaming.
- `POST /api/extract` `{objective, questions[], transcript}` →
  `{success, summary, result, answers, price, availability}` (Gemini; always
  summarizes in English even for foreign-language calls).

### How live transcript works
Vapi is told (`serverUrl` + `serverMessages: ["conversation-update",
"status-update", "end-of-call-report"]`) to POST events to `/api/vapi/webhook`.
We store turns in `_VAPI_LIVE[call_id]`. `vapi_status` returns the live store
while in progress, falls back to Vapi's GET `/call/{id}` at the end.
`PUBLIC_BASE_URL` (or Render's auto `RENDER_EXTERNAL_URL`) must be set for the
webhook URL. **Resilience:** if Vapi rejects the webhook fields, the backend
strips them and retries so the call still goes through.

### Multilingual
`/api/call` takes `language` (e.g. "Spanish"). When non-English: the system
prompt tells the model to speak only that language, sets a Deepgram transcriber
language code, and uses a model-generated first message. Summary still returns
in English. **Spanish is the most reliable; rehearse the demo in Spanish.**

---

## 6. Environment variables

```
# Brains
GEMINI_API_KEY=<secret>            # aistudio.google.com/apikey (free)
GEMINI_MODEL=gemini-2.0-flash

# Voice
CALL_PROVIDER=vapi                 # "vapi" for real calls, "mock" for offline demo
VAPI_API_KEY=<secret PRIVATE key>  # use the PRIVATE/secret key, not public
VAPI_PHONE_NUMBER_ID=92ce992e-3f46-47b2-85c7-e8f2faa7b546
VAPI_VOICE=alloy
VAPI_MODEL=gpt-4o

# Discovery (optional)
GOOGLE_PLACES_API_KEY=<secret>     # leave empty -> uses mock business list
DISCOVER_PROVIDER=mock             # or "places"

# Live transcript (Render sets RENDER_EXTERNAL_URL automatically)
PUBLIC_BASE_URL=<your render url>   # optional override
```

> The Vapi **private** key was shared in chat during setup — REGENERATE it after
> the hackathon. Keep all secrets in Render's dashboard / env group, never in git
> (`.env` is gitignored).

---

## 7. Deployment (Render)

- Repo deploys via the **root Dockerfile**, which now builds `switchboard/` and
  runs `uvicorn app:app`.
- Set the env vars above on the service (or an env group linked to it).
- After pushing, use **Manual Deploy → Clear build cache & deploy** (Render
  caches Docker layers).
- Verify: `<url>/health` should show `"provider":"vapi"`. If it shows `"mock"`,
  `CALL_PROVIDER=vapi` isn't set.

### GitHub access note
Pushing required installing the **Claude GitHub App** with read+write on the
repo (OAuth "Authorized" alone wasn't enough). It's installed now.

---

## 8. Known gotchas (we hit all of these)

1. **Vapi free numbers have a low DAILY outbound-call limit.** Paying Vapi more
   does NOT lift it — the cap is per free number. **Fix: import your own Twilio
   number into Vapi** (Vapi dashboard → Phone Numbers → Import). Twilio trial can
   only call *verified* numbers; upgrade (~$20) to call any number / real
   businesses. This is the main thing to finish for unlimited calls.
2. **Wrong-app deploys:** the leftover root Dockerfile/render.yaml used to deploy
   the old forecaster (`uvicorn src.api:app` → `ModuleNotFoundError: src`). Fixed
   by repointing root build at switchboard. If it recurs, check you're not on a
   stale service; consider deleting old Render services so there's only one.
3. **Silent dark button:** earlier the "Make the call" button hung disabled with
   no message when a call errored. Fixed with try/finally + showing the real
   error. (The error that surfaced was the Vapi daily limit above.)
4. **Vapi `serverMessages`:** `"transcript"` is NOT a valid event type — use
   `"conversation-update"`. Using the wrong one made Vapi reject every call.

---

## 9. Demo script (the winning flow)

1. Hook: "Raise your hand if you'd rather do anything than call 5 places."
2. **Single live call** to a JUDGE's phone — let them talk; the AI improvises a
   tailored response (real `vapi` mode only — mock is scripted/non-interactive).
3. **Call-around** finale: type/speak "cheapest oil change near me" → phone-only
   shops listed first → parallel calls stream live → ranked table → "book the
   best."
4. **Multilingual** beat (in Spanish) for the "unindexed/immigrant economy" point.
5. Safety net: if wifi/credits/limits fail, flip `CALL_PROVIDER=mock` — full flow
   still runs flawlessly.

---

## 10. Status & TODO

**Done & working:** plan/call/extract, mock + vapi providers, discovery + phone-
only ranking, comparison table, book-the-winner, professional UI, multilingual,
voice input, live transcript, English summaries, resilient error handling,
deploys to Render.

**Left to do (priority order):**
1. **Import a Twilio number** to remove the daily call limit (config, not code).
2. (Optional) **"Limit reached → switch to mock mode" friendly UI fallback** so a
   cap error never shows a raw 400 on stage.
3. (Optional, was next on the list) **War-room view + savings counter** — show
   parallel calls firing live with a running "$X saved / Y min saved" tally.
4. (Optional) Google Places live search (set GOOGLE_PLACES_API_KEY +
   DISCOVER_PROVIDER=places); calling real businesses needs Twilio upgraded.

**Pre-demo checklist:** set GEMINI key; one real test call to own phone; test a
Spanish call; confirm `/health` shows vapi; keep mock as fallback.
