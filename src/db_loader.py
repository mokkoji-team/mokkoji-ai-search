"""prod DB(MySQL)에서 동아리 + 모집글을 로드해 ClubSearchEngine 포맷으로 반환.

필요 환경변수 (.env):
  MOKKOJI_DB_HOST, MOKKOJI_DB_PORT, MOKKOJI_DB_NAME,
  MOKKOJI_DB_USER, MOKKOJI_DB_PASSWORD
"""

import os
from typing import Optional


def _cfg() -> Optional[dict]:
    host = os.getenv("MOKKOJI_DB_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("MOKKOJI_DB_PORT", 3306)),
        "database": os.getenv("MOKKOJI_DB_NAME", "mokkoji"),
        "user": os.getenv("MOKKOJI_DB_USER"),
        "password": os.getenv("MOKKOJI_DB_PASSWORD"),
    }


def db_available() -> bool:
    return _cfg() is not None


def load_clubs_from_db() -> list[dict]:
    """DB에서 전체 동아리 + 모집글을 읽어 clubs 리스트로 반환."""
    import pymysql
    import pymysql.cursors

    cfg = _cfg()
    if cfg is None:
        raise RuntimeError("DB 환경변수가 설정되지 않았습니다 (MOKKOJI_DB_HOST 등)")

    conn = pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor, charset="utf8mb4")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.name, c.club_category, c.club_affiliation,
                       c.description, u.code AS university_code
                FROM club c
                JOIN university u ON c.university_id = u.id
            """)
            clubs = {row["id"]: {**row, "recruitments": []} for row in cur.fetchall()}

            cur.execute("""
                SELECT club_id, title, content,
                       is_always_recruiting,
                       DATE(recruit_start) AS start_date,
                       DATE(recruit_end)   AS end_date
                FROM recruitment
            """)
            for r in cur.fetchall():
                cid = r["club_id"]
                if cid not in clubs:
                    continue
                clubs[cid]["recruitments"].append({
                    "title": r["title"] or "",
                    "content": r["content"] or "",
                    "is_always_open": bool(r["is_always_recruiting"]),
                    "start_date": str(r["start_date"]) if r["start_date"] else None,
                    "end_date": str(r["end_date"]) if r["end_date"] else None,
                })
    finally:
        conn.close()

    return list(clubs.values())
