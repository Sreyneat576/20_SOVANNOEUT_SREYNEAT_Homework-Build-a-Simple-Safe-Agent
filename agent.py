"""
Agent loop for the Library Agent.

Implements: User Request -> Agent/LLM -> Tool Call -> Tool Execution ->
Result -> Agent Decides Again -> Final Answer.

This talks to a local model served by Ollama (default: llama3.2:3b) using
Ollama's OpenAI-style function/tool calling support over its HTTP API at
http://localhost:11434. No cloud API key or network access is required;
just `ollama serve` running locally with the model pulled.
"""

import json
import re

import requests
from pydantic import ValidationError

import tools
from schemas import SearchBookInput, CheckAvailabilityInput, BorrowBookInput
from harness import safe_execute, MAX_ITERATIONS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

# Tool registry: maps tool name -> real implementation + its input schema.
TOOL_REGISTRY = {
    "search_book": {
        "func": tools.search_book,
        "schema": SearchBookInput,
        "description": "Search the library catalog by a title or author keyword.",
    },
    "check_availability": {
        "func": tools.check_availability,
        "schema": CheckAvailabilityInput,
        "description": "Check whether a specific book_id is currently available to borrow.",
    },
    "borrow_book": {
        "func": tools.borrow_book,
        "schema": BorrowBookInput,
        "description": "Borrow a specific book_id. Restricted to librarian/admin roles.",
    }
}


def _build_tool_specs() -> list:
    """Convert each Pydantic schema into an Ollama/OpenAI-style tool spec."""
    specs = []
    for name, entry in TOOL_REGISTRY.items():
        json_schema = entry["schema"].model_json_schema()
        json_schema.pop("title", None)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry["description"],
                    "parameters": json_schema,
                },
            }
        )
    return specs


SYSTEM_PROMPT = (
    "You are a Library Agent. Help the user search the library catalog, check "
    "book availability, and borrow books by calling the provided tools. Always "
    "call a tool to get real data rather than guessing at book IDs or "
    "availability. Use the result of one tool call to decide your next tool "
    "call when needed. If the user asks which books are available, first "
    "search the catalog, then call check_availability for every matching "
    "book before answering. Never infer availability from search results or "
    "say that you will check without making the tool call. Only call "
    "borrow_book when the user explicitly asks to borrow a book. Once you have "
    "enough information, reply with a final plain-text answer and do not call "
    "any more tools."
)

_AVAILABILITY_REQUEST = re.compile(
    r"\b(available|availability|in\s+stock|stock\s+status)\b", re.IGNORECASE
)
_BOOK_ID = re.compile(r"\bbook[_\s-]*id\s*[:=#]?\s*(\d+)\b", re.IGNORECASE)


def _availability_summary(candidates: dict, results: dict, search_was_run: bool) -> str:
    """Build an answer using only actual check_availability results."""
    if not candidates and search_was_run:
        return "I couldn't find any matching books in the library catalog."

    lines = []
    for book_id, book in candidates.items():
        title = book.get("title", f"Book {book_id}")
        author = book.get("author")
        label = f"'{title}' by {author}" if author else f"'{title}'"
        check = results.get(book_id)
        if not check:
            lines.append(f"- {label} (book_id={book_id}): availability was not verified.")
        elif not check.get("ok"):
            lines.append(
                f"- {label} (book_id={book_id}): availability could not be checked. "
                f"{check.get('message', 'The tool returned an error.')}"
            )
        else:
            available = check["result"].get("available")
            status = "available" if available else "not available"
            lines.append(f"- {label} (book_id={book_id}): {status}.")

    if not lines:
        return "I couldn't verify book availability from the catalog."
    return "I checked the catalog availability:\n" + "\n".join(lines)


def _enforce_availability_checks(
    agent, book_ids: list, candidates: dict, results: dict, verbose: bool, step: int
) -> list:
    """Run required availability checks through the normal safe tool path."""
    observations = []
    for book_id in book_ids:
        result = agent._execute_tool("check_availability", {"book_id": book_id})
        results[book_id] = result
        if result.get("ok"):
            book_result = result.get("result", {})
            candidates.setdefault(
                book_id,
                {"book_id": book_id, "title": book_result.get("title", f"Book {book_id}")},
            )
        observations.append({"book_id": book_id, **result})
        if verbose:
            print(
                f"[Step {step}] Tool call: "
                f"check_availability({{'book_id': {book_id}}}) -> {result}"
            )
    return observations


class LibraryAgent:
    """Runs the request -> LLM -> tool -> result -> LLM -> ... -> answer loop."""

    def __init__(self, role: str):
        self.role = role
        self.tool_specs = _build_tool_specs()

    def run(self, user_request: str, verbose: bool = True) -> str:
        availability_requested = bool(_AVAILABILITY_REQUEST.search(user_request))
        candidates = {}
        availability_results = {}
        search_was_run = False

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + f" The current user's role is '{self.role}'.",
            },
            {"role": "user", "content": user_request},
        ]

        for step in range(1, MAX_ITERATIONS + 1):
            response = self._call_llm(messages)
            message = response.get("message", {}) or {}
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                final_text = (message.get("content") or "").strip()
                # Safety net: small local models occasionally print a tool
                # call as plain text instead of using Ollama's structured
                # tool_calls field. Recognize that pattern rather than
                # silently treating it as a real final answer.
                fallback_call = self._try_parse_tool_call_from_text(final_text)
                if fallback_call is not None:
                    if verbose:
                        print(
                            "[Note] Model returned a tool call as plain text "
                            "instead of structured tool_calls; recovered it."
                        )
                    tool_calls = [fallback_call]
                else:
                    direct_ids = list(
                        dict.fromkeys(int(value) for value in _BOOK_ID.findall(user_request))
                    )
                    pending_ids = [
                        book_id for book_id in candidates
                        if book_id not in availability_results
                    ]
                    if availability_requested:
                        pending_ids.extend(
                            book_id for book_id in direct_ids
                            if book_id not in availability_results
                            and book_id not in pending_ids
                        )

                    if availability_requested and pending_ids:
                        observations = _enforce_availability_checks(
                            self, pending_ids, candidates, availability_results,
                            verbose, step + 1,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Application-required availability checks have "
                                    "completed. Use these actual results and do not "
                                    "claim a different status: "
                                    + json.dumps(observations)
                                ),
                            }
                        )
                        if step < MAX_ITERATIONS:
                            continue
                        final_text = _availability_summary(
                            candidates, availability_results, search_was_run
                        )
                        if verbose:
                            print(f"\n[Final Answer after {step} step(s)]\n{final_text}")
                        return final_text

                    if availability_requested and not search_was_run and not availability_results:
                        if step < MAX_ITERATIONS:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Before answering this availability request, "
                                        "you must call search_book to look up matching "
                                        "catalog entries. Do not guess or answer yet."
                                    ),
                                }
                            )
                            continue
                        final_text = (
                            "I couldn't verify availability because no catalog "
                            "search was completed."
                        )
                        if verbose:
                            print(f"\n[Final Answer after {step} step(s)]\n{final_text}")
                        return final_text

                    if availability_requested and (search_was_run or availability_results):
                        final_text = _availability_summary(
                            candidates, availability_results, search_was_run
                        )
                    if verbose:
                        print(f"\n[Final Answer after {step} step(s)]\n{final_text}")
                    return final_text

            messages.append(message)

            for call in tool_calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                tool_result = self._execute_tool(name, raw_args)

                if name == "search_book":
                    search_was_run = True
                    if tool_result.get("ok"):
                        for book in tool_result.get("result", {}).get("results", []):
                            book_id = book.get("book_id")
                            if isinstance(book_id, int):
                                candidates[book_id] = book
                elif name == "check_availability":
                    result_data = tool_result.get("result", {})
                    book_id = result_data.get("book_id", raw_args.get("book_id"))
                    if isinstance(book_id, int):
                        availability_results[book_id] = tool_result
                        if tool_result.get("ok"):
                            candidates.setdefault(
                                book_id,
                                {
                                    "book_id": book_id,
                                    "title": result_data.get("title", f"Book {book_id}"),
                                },
                            )

                if verbose:
                    print(f"[Step {step}] Tool call: {name}({raw_args}) -> {tool_result}")

                messages.append({"role": "tool", "content": json.dumps(tool_result)})

        if availability_requested:
            pending_ids = [
                book_id for book_id in candidates
                if book_id not in availability_results
            ]
            pending_ids.extend(
                book_id for value in _BOOK_ID.findall(user_request)
                for book_id in [int(value)]
                if book_id not in availability_results and book_id not in pending_ids
            )
            if pending_ids:
                _enforce_availability_checks(
                    self, pending_ids, candidates, availability_results,
                    verbose, MAX_ITERATIONS + 1,
                )
            final_text = (
                _availability_summary(candidates, availability_results, search_was_run)
                if search_was_run or availability_results
                else "I couldn't verify availability because no catalog search was completed."
            )
            if verbose:
                print(f"\n[Final Answer after the iteration limit]\n{final_text}")
            return final_text

        if verbose:
            print(f"\n[Iteration limit of {MAX_ITERATIONS} reached without a final answer.]")
        return "Iteration limit reached before the agent produced a final answer."

    @staticmethod
    def _try_parse_tool_call_from_text(text: str):
        """
        Try to recognize a tool call that a model printed as plain text
        (e.g. '{"name":"borrow_book","parameters":{"book_id":2}}') instead
        of using Ollama's structured tool_calls field. Returns a dict in
        the same shape as a real tool_call, or None if nothing recognizable
        is found. This is a small-model quirk seen with llama3.2:3b.
        """
        if not text or '"name"' not in text:
            return None

        # Some small models emit slightly malformed JSON (stray escaped
        # quotes). Normalize the common case before parsing.
        cleaned = text.strip().replace('\\"', '"')

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        name = data.get("name")
        args = data.get("parameters") or data.get("arguments") or {}

        if name in TOOL_REGISTRY and isinstance(args, dict):
            return {"function": {"name": name, "arguments": args}}
        return None

    def _execute_tool(self, name: str, raw_args: dict) -> dict:
        """Validate raw LLM-provided args against the schema, then run safely."""
        entry = TOOL_REGISTRY.get(name)
        if entry is None:
            return {"ok": False, "error": "unknown_tool", "message": f"Tool '{name}' does not exist."}

        try:
            validated = entry["schema"](**raw_args)
        except ValidationError as exc:
            return {"ok": False, "error": "invalid_input", "message": str(exc)}

        return safe_execute(entry["func"], validated, self.role, name)

    def _call_llm(self, messages: list) -> dict:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": self.tool_specs,
            "stream": False,
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach the local Ollama server at "
                f"{OLLAMA_URL}. Make sure 'ollama serve' is running and that "
                f"'{MODEL_NAME}' has been pulled (ollama pull {MODEL_NAME}). "
                f"Original error: {exc}"
            ) from exc
