from flask import Blueprint, request, jsonify, current_app
from models import db, User
import jwt, os
from datetime import datetime, timezone, timedelta   # ✅ tz-aware

auth_bp = Blueprint("auth", __name__, url_prefix="/api")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
print("JWT secret used by this process:", repr(JWT_SECRET))

def _generate_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),                    # ← JWT spec says it *must* be a string
        "iat": int(now.timestamp()),            # ← integers are safest
        "exp": int((now + timedelta(hours=8)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"error": "username & password required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "username already taken"}), 409

    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "user created"}), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    user = User.query.filter_by(username=data.get("username")).first()
    if not user or not user.check_password(data.get("password", "")):
        return jsonify({"error": "bad credentials"}), 401
    return jsonify({"token": _generate_token(user)})
def current_user():
    from functools import wraps
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hdr = request.headers.get("Authorization", "")
            if not hdr.startswith("Bearer "):
                return jsonify({"error": "missing token"}), 401
            token = hdr.split()[1]
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                user = User.query.get(payload["sub"])
            except jwt.PyJWTError:
                return jsonify({"error": "invalid token"}), 401
            return fn(user, *args, **kwargs)
        return wrapper
    return decorator
