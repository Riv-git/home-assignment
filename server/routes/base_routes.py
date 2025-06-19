from flask import Blueprint

base_bp = Blueprint('base', __name__)

@base_bp.route("/", methods=["GET"])
def index():
    return "Hello, World!"

@base_bp.route("/health", methods=["GET"])
def health():
    return "OK" 