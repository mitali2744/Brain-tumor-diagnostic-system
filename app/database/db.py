"""
Database Setup and Helper Functions
-------------------------------------
Uses SQLite via Flask-SQLAlchemy.
Stores every scan result: original MRI, segmentation mask,
Grad-CAM overlay, classification label, confidence, and timestamp.
"""

import os
import sys
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ScanResult(db.Model):
    """
    One row per uploaded MRI scan.
    Stores file paths (relative to static/) and prediction outputs.
    """
    __tablename__ = "scan_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # File paths (relative to app/static/)
    original_filename  = db.Column(db.String(256), nullable=False)
    upload_path        = db.Column(db.String(512), nullable=False)   # original MRI
    segmentation_path  = db.Column(db.String(512), nullable=True)    # binary mask
    gradcam_path       = db.Column(db.String(512), nullable=True)    # heatmap overlay
    composite_path     = db.Column(db.String(512), nullable=True)    # 4-panel result

    # Classification output
    predicted_class    = db.Column(db.Integer,  nullable=True)
    predicted_label    = db.Column(db.String(64), nullable=True)
    confidence         = db.Column(db.Float, nullable=True)          # 0-100

    # Per-class probabilities stored as JSON string
    all_probabilities  = db.Column(db.Text, nullable=True)

    # Segmentation metric (if ground truth available)
    dice_score         = db.Column(db.Float, nullable=True)

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        probs = {}
        if self.all_probabilities:
            try:
                probs = json.loads(self.all_probabilities)
            except Exception:
                pass
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "upload_path": self.upload_path,
            "segmentation_path": self.segmentation_path,
            "gradcam_path": self.gradcam_path,
            "composite_path": self.composite_path,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "all_probabilities": probs,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def __repr__(self):
        return f"<ScanResult id={self.id} label={self.predicted_label} conf={self.confidence:.1f}%>"


def init_db(app):
    """Initialize the database with the Flask app context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
