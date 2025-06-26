import os
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS

from models import db
from routes.base_routes import base_bp
from routes.auth_routes import auth_bp
from routes.pdf_routes  import pdf_bp

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET", "change-me")

    CORS(app, origins="*")

    db.init_app(app)
    with app.app_context():
        db.create_all()

    # blueprints
    app.register_blueprint(base_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(pdf_bp)
    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
