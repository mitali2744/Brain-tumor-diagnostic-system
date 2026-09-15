"""
Flask Application Entry Point
------------------------------
Run:
    python app/run.py
Then open: http://localhost:5000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from flask import Flask
from app.database.db import init_db
from app.routes import bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Config ────────────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = "brain_tumor_secret_key_change_in_production"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{config.DATABASE_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # ── Ensure required directories exist ────────────────────────────────────
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)

    # ── Database ──────────────────────────────────────────────────────────────
    init_db(app)

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(bp)

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 50)
    print("  Brain Tumor Diagnostic System")
    print("  Running at: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
