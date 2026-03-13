"""
API route schemas and helper utilities.
Import this in app.py to register additional blueprints.
"""
from flask import Blueprint, request, jsonify

bp = Blueprint("medxai", __name__, url_prefix="/api/v1")


@bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})


@bp.route("/features", methods=["GET"])
def list_features():
    """Return the ordered feature list and their acceptable ranges."""
    from src.utils.validators import FEATURE_CONSTRAINTS
    return jsonify({
        name: {k: v for k, v in constraints.items() if k != "type"}
        for name, constraints in FEATURE_CONSTRAINTS.items()
    })
