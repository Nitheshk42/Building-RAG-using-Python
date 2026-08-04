import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "studysage.db"


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
