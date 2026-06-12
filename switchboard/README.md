# ☎️ Switchboard — the AI that makes your phone calls

You type one task. Switchboard turns it into a calling objective, **dials the
number(s)** — in parallel for call-arounds — has a real conversation, then reads
the transcripts back as a **ranked comparison table**.

> The dread isn't making one call. It's calling *five places* to compare.
> Switchboard makes them all at once.

## Two modes
- **📞 Do it for me** — one task → one call (book, cancel, ask).
- **🔀 Call around** — one task → many calls in parallel → ranked results table.

## Run it (works with ZERO keys in demo mode)
```bash
pip install -r requirements.txt
cp .env.example .env          # CALL_PROVIDER=mock needs no accounts at all
uvicorn app:app --reload --port 8000
# open http://localhost:8000
```
In **mock mode** the app plays a scripted live call — perfect for building the
UI, rehearsing the pitch, and as an on-stage safety net.

## Go live
1. **Gemini** (the brains): free key at https://aistudio.google.com/apikey → `GEMINI_API_KEY`
2. **Bland.ai** (the voice): key + trial credits at https://bland.ai → `BLAND_API_KEY`
3. Set `CALL_PROVIDER=bland` in `.env`. Done.

## How it works
| Piece | Job |
|-------|-----|
| Gemini | vague task → `{objective, questions}`; transcript → structured result |
| Bland.ai | actually dials the phone and talks (swap via `CALL_PROVIDER`) |
| mock | scripted offline call so nothing depends on the venue wifi |

## Endpoints
- `POST /api/plan` — task → objective + questions
- `POST /api/call` — number + plan → `call_id`
- `GET  /api/call/{provider}/{call_id}` — poll status + live transcript
- `POST /api/extract` — transcript → structured result

## ⚖️ Please call responsibly
California is a **two-party-consent** state for recording. Recording defaults to
**off**. Demo on people who consent (judges, teammates). Don't spam businesses.
