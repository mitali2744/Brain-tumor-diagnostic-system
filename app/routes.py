"""
Flask Routes
-------------
GET  /              → Home / upload page
POST /predict       → Run full pipeline, save to DB, show result
GET  /history       → View all past scan results
GET  /result/<id>   → View a single past result
GET  /about         → About / project info page
POST /delete/<id>   → Delete a scan record
"""

import os
import json
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    current_app,
)
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from app.database.db import db, ScanResult
from app.pipeline import run_pipeline

bp = Blueprint("main", __name__)


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


# ─── Home / Upload ────────────────────────────────────────────────────────────

@bp.route("/", methods=["GET"])
def index():
    models_ready = (
        os.path.exists(config.CLASSIFIER_MODEL_PATH)
        and os.path.exists(config.SEGMENTATION_MODEL_PATH)
    )
    return render_template("index.html", models_ready=models_ready)


# ─── Predict ──────────────────────────────────────────────────────────────────

@bp.route("/predict", methods=["POST"])
def predict():
    if "mri_file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("main.index"))

    file = request.files["mri_file"]

    if file.filename == "":
        flash("Please select a file before uploading.", "warning")
        return redirect(url_for("main.index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PNG, JPG, or JPEG image.", "danger")
        return redirect(url_for("main.index"))

    # Save uploaded file
    filename = secure_filename(file.filename)
    save_path = os.path.join(config.UPLOAD_FOLDER, filename)
    # Avoid overwriting — prefix with a short UUID fragment
    import uuid
    unique_prefix = str(uuid.uuid4())[:8]
    unique_filename = f"{unique_prefix}_{filename}"
    save_path = os.path.join(config.UPLOAD_FOLDER, unique_filename)
    file.save(save_path)

    # Run pipeline
    try:
        result = run_pipeline(save_path, original_filename=file.filename)
    except FileNotFoundError as e:
        flash(str(e), "danger")
        return redirect(url_for("main.index"))
    except Exception as e:
        flash(f"An error occurred during analysis: {str(e)}", "danger")
        current_app.logger.exception("Pipeline error")
        return redirect(url_for("main.index"))

    # Save to database
    scan = ScanResult(
        original_filename=result["original_filename"],
        upload_path=result["upload_path"],
        segmentation_path=result["segmentation_path"],
        gradcam_path=result["gradcam_path"],
        composite_path=result["composite_path"],
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
    )
    db.session.add(scan)
    db.session.commit()

    return redirect(url_for("main.result", scan_id=scan.id))


# ─── Single Result ────────────────────────────────────────────────────────────

@bp.route("/result/<int:scan_id>")
def result(scan_id: int):
    scan = ScanResult.query.get_or_404(scan_id)
    probs = {}
    if scan.all_probabilities:
        try:
            probs = json.loads(scan.all_probabilities)
        except Exception:
            pass
    return render_template("result.html", scan=scan, probs=probs)


# ─── History ──────────────────────────────────────────────────────────────────

@bp.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    scans = ScanResult.query.order_by(ScanResult.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template("history.html", scans=scans)


# ─── Delete Scan ──────────────────────────────────────────────────────────────

@bp.route("/delete/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id: int):
    scan = ScanResult.query.get_or_404(scan_id)

    # Remove associated files from disk
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    for path_attr in ["upload_path", "segmentation_path", "gradcam_path", "composite_path"]:
        rel_path = getattr(scan, path_attr)
        if rel_path:
            full_path = os.path.join(static_dir, rel_path)
            if os.path.exists(full_path):
                os.remove(full_path)

    db.session.delete(scan)
    db.session.commit()
    flash(f"Scan #{scan_id} deleted successfully.", "success")
    return redirect(url_for("main.history"))


# ─── About ────────────────────────────────────────────────────────────────────

@bp.route("/about")
def about():
    return render_template("about.html")
