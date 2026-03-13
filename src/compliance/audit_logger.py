"""
Audit Logger — HIPAA-aligned audit trail for all model predictions.
Stores timestamped records in SQLite (default) with full explanation metadata.
No raw PHI is stored — only anonymised patient IDs.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB = os.getenv("DATABASE_URL", "sqlite:///medical_ai.db").replace("sqlite:///", "")


class AuditLogger:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    patient_id    TEXT    NOT NULL,
                    model_version TEXT,
                    prediction    INTEGER,
                    confidence    REAL,
                    features_json TEXT,
                    explanation_json TEXT,
                    guideline_json TEXT,
                    has_flag      INTEGER DEFAULT 0
                )
            """)
            conn.commit()
        logger.info(f"Audit DB initialised at {self.db_path}")

    def log(self, patient_id: str, features, prediction,
            confidence, explanation: dict = None,
            guideline_check: dict = None,
            model_version: str = "decision_tree_v2") -> int:
        """Insert one audit record. Returns the new row ID."""
        has_flag = int(
            guideline_check.get("has_contraindication", False)
            or guideline_check.get("total_flags", 0) > 0
        ) if guideline_check else 0

        conf_val = float(confidence[0]) if hasattr(confidence, "__len__") else float(confidence)
        pred_val = int(prediction[0]) if hasattr(prediction, "__len__") else int(prediction)

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO prediction_log
                    (timestamp, patient_id, model_version, prediction, confidence,
                     features_json, explanation_json, guideline_json, has_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                patient_id,
                model_version,
                pred_val,
                conf_val,
                json.dumps(features.tolist() if hasattr(features, "tolist") else list(features)),
                json.dumps(explanation or {}, default=str),
                json.dumps(guideline_check or {}, default=str),
                has_flag,
            ))
            conn.commit()
            row_id = cursor.lastrowid
        logger.info(f"Audit record {row_id} logged for patient {patient_id}")
        return row_id

    def fetch_recent(self, limit: int = 50) -> list:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, timestamp, patient_id, model_version,
                       prediction, confidence, has_flag
                FROM prediction_log
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        keys = ["id", "timestamp", "patient_id", "model_version",
                "prediction", "confidence", "has_flag"]
        return [dict(zip(keys, r)) for r in rows]

    def compliance_summary(self) -> dict:
        with self._connect() as conn:
            total  = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
            flags  = conn.execute("SELECT COUNT(*) FROM prediction_log WHERE has_flag=1").fetchone()[0]
            avg_cf = conn.execute("SELECT AVG(confidence) FROM prediction_log").fetchone()[0]
        return {
            "total_predictions": total,
            "flagged_predictions": flags,
            "flag_rate": round(flags / total, 4) if total else 0,
            "average_confidence": round(avg_cf or 0, 4),
        }

    def export_csv(self, output_path: str = "logs/audit_export.csv"):
        import csv
        rows = self.fetch_recent(limit=100_000)
        if not rows:
            logger.info("No rows to export.")
            return
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Audit log exported to {output_path}")
