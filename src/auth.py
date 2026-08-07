import os
import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path

# On Cloud Run (and most serverless platforms), the container filesystem is ephemeral and
# NOT shared across instances - every scale event or cold start gets a brand-new, empty disk.
# That's exactly why "no such table: embeddings" (and users vanishing / re-showing the login
# page) started happening after moving off Render: a second instance spun up with none of the
# data the first instance wrote locally. APP_DATA_DIR lets this point at a persistent, shared
# mount (e.g. a Cloud Storage FUSE volume) in production, while defaulting to the project root
# for local development where a single process on a normal disk is fine as-is.
DATA_ROOT = Path(os.getenv("APP_DATA_DIR", str(Path(__file__).parent.parent)))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_ROOT / "studysage.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            password_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            username TEXT PRIMARY KEY,
            level TEXT,
            resume_uploaded INTEGER DEFAULT 0,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            message TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def save_feedback(username, message):
    if not message or not message.strip():
        return False
    conn = _get_conn()
    conn.execute(
        "INSERT INTO feedback (username, message, created_at) VALUES (?, ?, ?)",
        (username, message.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return True


def create_user(username, email, password):
    """Returns (success, message)."""
    if not username or not password:
        return False, "Username and password are required."
    conn = _get_conn()
    existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return False, "Username already taken."
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                 (username, email, password_hash))
    conn.execute("INSERT INTO profiles (username, level, resume_uploaded) VALUES (?, NULL, 0)",
                 (username,))
    conn.commit()
    conn.close()
    return True, "Account created."


def verify_user(username, password):
    """Returns True if credentials are valid."""
    conn = _get_conn()
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())


def get_profile(username):
    conn = _get_conn()
    row = conn.execute("SELECT level, resume_uploaded FROM profiles WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row:
        return {"level": None, "resume_uploaded": False}
    return {"level": row[0], "resume_uploaded": bool(row[1])}


def set_profile_level(username, level):
    conn = _get_conn()
    conn.execute("UPDATE profiles SET level = ? WHERE username = ?", (level, username))
    conn.commit()
    conn.close()


def mark_resume_uploaded(username):
    conn = _get_conn()
    conn.execute("UPDATE profiles SET resume_uploaded = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()
