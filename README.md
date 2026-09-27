# Simple Safe Agent — Library Agent

A small, complete agent that searches a library catalog, checks book
availability, and borrows books — built around a local LLM (**llama3.2:3b**
served by **Ollama**) doing real function/tool calling. No cloud API key,
no bonus features (no HITL approval, no risk tiers, no allowlists, no MCP
server, no ReAct/Planner/Router pattern) — just the required agent loop,
tools, schemas, permissions, and basic safety.

## 1. Project Overview

The Library Agent takes a natural-language request (e.g. *"Find a book
about machine learning that's available"*), sends it to a local LLM along
with a set of tool definitions, and lets the model decide which tools to
call and in what order. The agent executes each requested tool against a
small in-memory book catalog, feeds the result back to the model, and
repeats until the model gives a final plain-text answer (or a safety limit
is hit).

## 2. Available Tools

| Tool | Description | Who can call it |
|---|---|---|
| `search_book(query: str)` | Searches the catalog by title/author keyword. | member, librarian, admin |
| `check_availability(book_id: int)` | Looks up whether a specific book is available. | member, librarian, admin |
| `borrow_book(book_id: int)` | Borrows a book (marks it unavailable). | librarian, admin **only** |

Each tool has a real Python implementation in `tools.py` and an explicit
Pydantic input schema in `schemas.py` (types, required fields, and
constraints such as "book_id must be a positive integer").

## 3. Agent Loop

```
User Request
    │
    ▼
Agent sends messages + tool definitions to the local LLM (Ollama)
    │
    ▼
LLM responds with either:
    ├─► a tool_call  ──► harness validates input (Pydantic) + checks
    │                    role permission ──► tool executes ──► result
    │                    is appended to the conversation ──► loop repeats
    │
    └─► a final plain-text answer ──► loop ends, answer is returned
```

The loop is capped at `MAX_ITERATIONS = 6` (see `harness.py`) round trips,
so a misbehaving or confused model can never run forever. A single request
can trigger **multiple tools in sequence**, using the result of one call
to decide the next (search → check availability → borrow, for example).

## 4. Permission Rule

Enforced in `harness.py`'s `check_permission()`, **in application code**,
not just in the prompt:

- **member** — may call `search_book` and `check_availability` only.
- **librarian** / **admin** — may additionally call `borrow_book`.

If a `member` role attempts `borrow_book`, the tool is never executed. The
harness returns a controlled result:

```json
{"ok": false, "error": "permission_denied",
 "message": "Role 'member' is not permitted to call 'borrow_book'. Allowed tools for 'member': ['check_availability', 'search_book']."}
```

This result is fed back to the LLM (and shown to the user) — the action is
blocked with a clear message, never silently ignored and never allowed to
happen anyway.

## 5. Safety

- **Input validation** — every tool call's raw arguments are parsed
  through the matching Pydantic schema (`schemas.py`) before execution.
  Empty queries, non-positive `book_id` values, and missing fields are
  rejected with a clear Pydantic error message rather than reaching the
  tool function.
- **Error handling** — `harness.safe_execute()` wraps every tool call in
  `try/except`. Expected failures (e.g. "no book found") return a
  controlled `{"ok": false, "error": ..., "message": ...}` result;
  truly unexpected exceptions are caught too and turned into a generic
  `internal_error` message — the raw exception/stack trace is never
  leaked back to the user.
- **Availability verification** — when a request asks which books are
  available, the application records catalog search results and requires a
  `check_availability` result for each match before returning an availability
  answer. If the model stops after saying it will check, the application runs
  the missing checks through the same permission and error-handling harness;
  the final status summary is built from those tool results.
- **Iteration limit** — `MAX_ITERATIONS = 6` in `harness.py` bounds the
  agent loop in `agent.py`, so the model cannot call tools forever. If
  the limit is reached, the agent stops instead of hanging. Availability
  requests still return a summary based on completed catalog checks; other
  requests report that no final answer was produced.

## 6. Example Run

The transcript below is a scripted demonstration of the tool pipeline; it
is not an Ollama run. Capture a terminal screenshot from your own Ollama run
for the homework submission. The exact wording of live model responses may
vary, while tool results and application-enforced permissions come from the
project code.

### Run 1 — Normal successful multi-tool flow (role = admin)

Request: `"Find the Introduction to Machine Learning book, check its availability, and borrow it if available"`

```
############################################
# RUN 1: Normal multi-tool flow (role=admin)
############################################
[Step 1] Tool call: search_book({'query': 'Introduction to Machine Learning'}) -> {'ok': True, 'result': {'query': 'Introduction to Machine Learning', 'results': [{'book_id': 1, 'title': 'Introduction to Machine Learning', 'author': 'A. Smith'}], 'count': 1}}
[Step 2] Tool call: check_availability({'book_id': 1}) -> {'ok': True, 'result': {'book_id': 1, 'title': 'Introduction to Machine Learning', 'available': True}}
[Step 3] Tool call: borrow_book({'book_id': 1}) -> {'ok': True, 'result': {'book_id': 1, 'success': True, 'message': "'Introduction to Machine Learning' has been borrowed successfully."}}

[Final Answer after 4 step(s)]
I found 'Introduction to Machine Learning' by A. Smith (book_id=1). It was available, so I borrowed it for you successfully.
```

### Run 2 — Blocked action due to permissions (role = member)

Request: `"Please borrow book_id 1 for me"`

```
############################################
# RUN 2: Permission-denied flow (role=member)
############################################
[Step 1] Tool call: borrow_book({'book_id': 1}) -> {'ok': False, 'error': 'permission_denied', 'message': "Role 'member' is not permitted to call 'borrow_book'. Allowed tools for 'member': ['check_availability', 'search_book']."}

[Final Answer after 2 step(s)]
I was not able to borrow that book directly because your role does not have permission. Please ask a librarian or admin to borrow it on your behalf.
```

### Run 3 — Invalid input caught by validation (bonus check, role = admin)

Request: `"Check availability of book -5"`

```
############################################
# RUN 3: Validation error (bad input caught, no crash)
############################################
[Step 1] Tool call: check_availability({'book_id': -5}) -> {'ok': False, 'error': 'invalid_input', 'message': '1 validation error for CheckAvailabilityInput\nbook_id\n  Value error, book_id must be a positive integer [type=value_error, input_value=-5, input_type=int]'}

[Final Answer after 2 step(s)]
That book_id isn't valid, so I couldn't check availability.
```

## Running the Project

### Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- The `llama3.2:3b` model pulled:

```bash
ollama pull llama3.2:3b
ollama serve   # if not already running as a background service
```

- Python dependencies:

```bash
pip install pydantic requests
```

### Run it

```bash
cd my-agent
python main.py --role admin --request "Find a book about machine learning that's available"
python main.py --role member --request "Please borrow book_id 1 for me"
```

Or run it interactively (it will prompt you for a role and a request):

```bash
python main.py
```

Valid roles: `member`, `librarian`, `admin`.

### Notes

- The agent talks to Ollama's local HTTP API at
  `http://localhost:11434/api/chat` (see `OLLAMA_URL` / `MODEL_NAME` at the
  top of `agent.py`) — no internet access or API key is required.
- The book catalog lives in memory (`tools.py`) and resets each time you
  run the program.
