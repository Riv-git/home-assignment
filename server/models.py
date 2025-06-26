from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
db = SQLAlchemy()

class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    pw_hash  = db.Column(db.String(128), nullable=False)

    def set_password(self, raw_pw: str) -> None:
        self.pw_hash = generate_password_hash(raw_pw)

    def check_password(self, raw_pw: str) -> bool:
        return check_password_hash(self.pw_hash, raw_pw)

class PDFDocument(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"))
    sha256      = db.Column(db.String(64), index=True)
    filename    = db.Column(db.String(200))
    pages       = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    # optional free-form metadata
    meta        = db.Column(db.JSON, default={})

    user = db.relationship("User", backref="documents")
