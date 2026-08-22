"""prod DB(MySQL)에서 동아리 + 모집글을 로드해 ClubSearchEngine 포맷으로 반환.

필요 환경변수 (.env):
  MOKKOJI_DB_HOST, MOKKOJI_DB_PORT, MOKKOJI_DB_NAME,
  MOKKOJI_DB_USER, MOKKOJI_DB_PASSWORD
"""

import os
import re
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


def load_field_limits() -> dict:
    """DB 컬럼 타입을 읽어 varchar 최대 길이를 반환. text 타입은 None (제약 없음).

    반환 형식: {"club.name": 50, "club.description": None, "recruitment.content": None, ...}
    serve.py 시작 시 호출해 search.configure_field_limits()에 전달한다.
    """
    import pymysql
    import pymysql.cursors

    cfg = _cfg()
    if cfg is None:
        return {}

    def _varchar_len(col_type: str) -> Optional[int]:
        m = re.match(r'varchar\((\d+)\)', col_type)
        return int(m.group(1)) if m else None

    conn = pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor, charset="utf8mb4")
    limits: dict = {}
    try:
        with conn.cursor() as cur:
            for table in ("club", "recruitment"):
                cur.execute(f"SHOW FULL COLUMNS FROM {table}")
                for row in cur.fetchall():
                    limits[f"{table}.{row['Field']}"] = _varchar_len(row["Type"])
    finally:
        conn.close()
    return limits


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
                       c.description, c.instagram,
                       c.created_at, c.updated_at,
                       u.code AS university_code
                FROM club c
                JOIN university u ON c.university_id = u.id
            """)
            clubs = {
                row["id"]: {
                    **row,
                    "created_at": str(row["created_at"]) if row.get("created_at") else None,
                    "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
                    "recruitments": [],
                }
                for row in cur.fetchall()
            }

            cur.execute("""
                SELECT id, club_id, title, content,
                       is_always_recruiting,
                       recruit_start, recruit_end,
                       created_at, updated_at
                FROM recruitment
            """)
            for r in cur.fetchall():
                cid = r["club_id"]
                if cid not in clubs:
                    continue
                clubs[cid]["recruitments"].append({
                    "id": r["id"],
                    "title": r["title"] or "",
                    "content": r["content"] or "",
                    "is_always_open": bool(r["is_always_recruiting"]),
                    "recruit_start": str(r["recruit_start"]) if r["recruit_start"] else None,
                    "recruit_end": str(r["recruit_end"]) if r["recruit_end"] else None,
                    "created_at": str(r["created_at"]) if r["created_at"] else None,
                    "updated_at": str(r["updated_at"]) if r["updated_at"] else None,
                })
    finally:
        conn.close()

    return list(clubs.values())
