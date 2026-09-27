"""
Real Python implementations of the Library Agent's tools, backed by a small
in-memory "database". Each function assumes it is only ever called with
already-validated arguments (validation happens in agent.py / harness.py
before these functions run).
"""

# In-memory book catalog: book_id -> book record
BOOKS = {
    1: {"title": "Introduction to Machine Learning", "author": "A. Smith", "available": True},
    2: {"title": "Deep Learning Foundations", "author": "B. Lee", "available": False},
    3: {"title": "Python for Data Science", "author": "C. Wong", "available": True},
    4: {"title": "The Great Gatsby", "author": "F. S. Fitzgerald", "available": True},
    5: {"title": "Machine Learning for Absolute Beginners", "author": "D. Patel", "available": True},
}


def search_book(query: str) -> dict:
    """Search the catalog by title or author substring (case-insensitive)."""
    q = query.lower()
    matches = [
        {"book_id": book_id, "title": rec["title"], "author": rec["author"]}
        for book_id, rec in BOOKS.items()
        if q in rec["title"].lower() or q in rec["author"].lower()
    ]
    return {"query": query, "results": matches, "count": len(matches)}


def check_availability(book_id: int) -> dict:
    """Look up whether a specific book_id is currently available."""
    record = BOOKS.get(book_id)
    if record is None:
        raise ValueError(f"No book found with book_id={book_id}")
    return {
        "book_id": book_id,
        "title": record["title"],
        "available": record["available"],
    }


def borrow_book(book_id: int) -> dict:
    """Borrow a book by book_id (admin/librarian only — enforced in harness.py)."""
    record = BOOKS.get(book_id)
    if record is None:
        raise ValueError(f"No book found with book_id={book_id}")
    if not record["available"]:
        return {
            "book_id": book_id,
            "success": False,
            "message": f"'{record['title']}' is currently not available to borrow.",
        }
    record["available"] = False
    return {
        "book_id": book_id,
        "success": True,
        "message": f"'{record['title']}' has been borrowed successfully.",
    }
