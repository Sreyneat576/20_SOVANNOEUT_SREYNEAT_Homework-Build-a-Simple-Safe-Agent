"""
Entry point for the Simple Safe Agent (Library Agent).

Usage:
    python main.py
    python main.py --role member --request "Find a book about machine learning that's available"

If --role/--request are omitted, you'll be prompted interactively.
"""

import argparse

from agent import LibraryAgent
from harness import ROLE_PERMISSIONS

VALID_ROLES = set(ROLE_PERMISSIONS.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Safe Library Agent")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), help="User role")
    parser.add_argument("--request", help="Natural language request for the agent")
    args = parser.parse_args()

    print("=== Simple Safe Agent: Library Agent (local llama3.2:3b via Ollama) ===")

    role = args.role
    if role is None:
        role = input(f"Enter role {sorted(VALID_ROLES)}: ").strip().lower()
        if role not in VALID_ROLES:
            print(f"Unknown role '{role}', defaulting to 'member'.")
            role = "member"

    request = args.request
    if request is None:
        request = input("Enter your request: ").strip()
    if not request:
        print("Empty request, exiting.")
        return

    agent = LibraryAgent(role=role)
    agent.run(request)


if __name__ == "__main__":
    main()
