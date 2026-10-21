"""Regression check for same-day registration progress estimates."""
from datetime import date, datetime
import os
import tempfile

import server


with tempfile.TemporaryDirectory() as temp_dir:
    old_db = server.DB
    server.DB = os.path.join(temp_dir, "progress-test.sqlite")
    server.init_db()
    server.init_labs()
    today = date.today()
    created = datetime.combine(today, datetime.min.time()).timestamp()
    with server.closing(server.db()) as connection:
        connection.execute(
            "INSERT INTO users(email, name, password_hash, created_at, initial_level, initial_level_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("progress-test@example.com", "Progress Test", "test", created, 48.5, created),
        )
        uid = connection.execute(
            "SELECT id FROM users WHERE email=?", ("progress-test@example.com",)
        ).fetchone()["id"]
        connection.execute(
            "INSERT INTO profiles(user_id, birth_year, sex, height_cm, weight_kg, "
            "quiz_answers, quiz_score, skin_type, exposed, sunscreen, supplement, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, today.year - 30, "F", 165, 60, "0,0", 3, 3, "face_hands",
             "sometimes", "no", created),
        )
        connection.commit()

    with server.closing(server.db()) as connection:
        profile = server.get_profile(uid, connection)
    trajectory = server.trajectory_for(uid, profile)
    assert trajectory["baseline"] == 48.5
    assert trajectory["current"] == 48.5
    assert trajectory["registration_date"] == today.isoformat()
    assert trajectory["points"][0]["date"] == today.isoformat()
    assert trajectory["points"][0]["projected"] is False
    print("[PASS] same-day registration does not decay before account creation")
    server.DB = old_db
