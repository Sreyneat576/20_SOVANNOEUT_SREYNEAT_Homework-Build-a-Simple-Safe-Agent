"""
Safety and permission harness for the Library Agent.

This module is where all the "application code" enforcement lives, as
opposed to relying on the LLM's prompt to behave. Even if the model
hallucinates a disallowed tool call, this code blocks it.
"""

# Maximum number of agent <-> LLM <-> tool round trips before we force a stop.
# This is the only "loop safety" mechanism required/used in this project.
MAX_ITERATIONS = 6

# Role -> set of tool names that role is permitted to call.
# A regular "member" cannot borrow_book directly; only "librarian"/"admin" can.
ROLE_PERMISSIONS = {
    "member": {"search_book", "check_availability"},
    "librarian": {"search_book", "check_availability", "borrow_book"},
    "admin": {"search_book", "check_availability", "borrow_book"},
}


class PermissionDeniedError(Exception):
    """Raised when a role attempts a tool it is not allowed to call."""


def check_permission(role: str, tool_name: str) -> None:
    """Raise PermissionDeniedError if `role` may not call `tool_name`."""
    allowed = ROLE_PERMISSIONS.get(role, set())
    if tool_name not in allowed:
        raise PermissionDeniedError(
            f"Role '{role}' is not permitted to call '{tool_name}'. "
            f"Allowed tools for '{role}': {sorted(allowed) if allowed else 'none'}."
        )


def safe_execute(func, validated_input, role: str, tool_name: str) -> dict:
    """
    Run a tool safely:
      1. Enforce role-based permission.
      2. Execute the tool with already-validated (Pydantic) input.
      3. Catch expected errors (ValueError from tools.py) and unexpected
         errors, returning a controlled result instead of raising/crashing
         or leaking a raw traceback back to the user.

    Always returns a dict of the shape:
        {"ok": True,  "result": <tool output>}
        {"ok": False, "error": <short code>, "message": <human-readable>}
    """
    try:
        check_permission(role, tool_name)
    except PermissionDeniedError as exc:
        return {"ok": False, "error": "permission_denied", "message": str(exc)}

    try:
        result = func(**validated_input.model_dump())
        return {"ok": True, "result": result}
    except ValueError as exc:
        # Expected, "controlled" failure (bad id, not found, etc.)
        return {"ok": False, "error": "execution_error", "message": str(exc)}
    except Exception:
        # Unexpected failure: never leak the raw exception/stack trace.
        return {
            "ok": False,
            "error": "internal_error",
            "message": "An unexpected internal error occurred while running the tool.",
        }
