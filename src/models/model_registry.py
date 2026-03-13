"""
Model Registry — tracks all trained model versions and their performance metadata.
"""
import json
import os
from datetime import datetime


class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = registry_path
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "schema_version": "1.0"}

    def _save(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(self, name: str, path: str, metrics: dict, model_type: str) -> dict:
        """Register a new model version, deactivating previous versions of same name."""
        for m in self.registry["models"]:
            if m["name"] == name:
                m["active"] = False

        entry = {
            "name": name,
            "version": self._next_version(name),
            "path": path,
            "model_type": model_type,
            "metrics": metrics,
            "registered_at": datetime.utcnow().isoformat(),
            "active": True,
        }
        self.registry["models"].append(entry)
        self._save()
        return entry

    def _next_version(self, name: str) -> int:
        versions = [m["version"] for m in self.registry["models"] if m["name"] == name]
        return max(versions, default=0) + 1

    def get_active(self, name: str) -> dict | None:
        for m in reversed(self.registry["models"]):
            if m["name"] == name and m["active"]:
                return m
        return None

    def list_all(self) -> list:
        return self.registry["models"]

    def deactivate(self, name: str, version: int):
        for m in self.registry["models"]:
            if m["name"] == name and m["version"] == version:
                m["active"] = False
        self._save()
