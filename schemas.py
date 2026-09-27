"""
Pydantic input schemas for the Library Agent's tools.

Each schema defines the exact shape of arguments a tool accepts, including
types, required fields, and simple constraints. The agent validates every
tool call against these schemas *before* the tool is executed, so bad input
(e.g. an empty query or a non-positive book_id) is rejected with a clear
error instead of crashing the program.
"""

from pydantic import BaseModel, Field, field_validator


class SearchBookInput(BaseModel):
    """Arguments for search_book."""

    query: str = Field(
        ...,
        description="Search text to match against book title or author (must not be empty).",
        min_length=1,
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace only")
        return v.strip()


class CheckAvailabilityInput(BaseModel):
    """Arguments for check_availability."""

    book_id: int = Field(
        ...,
        description="ID of the book to check. Must be a positive integer.",
    )

    @field_validator("book_id")
    @classmethod
    def book_id_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("book_id must be a positive integer")
        return v


class BorrowBookInput(BaseModel):
    """Arguments for borrow_book."""

    book_id: int = Field(
        ...,
        description="ID of the book to borrow. Must be a positive integer.",
    )

    @field_validator("book_id")
    @classmethod
    def book_id_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("book_id must be a positive integer")
        return v
