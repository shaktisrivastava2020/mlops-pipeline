"""Unit tests for registry version logic (offline, no GCS calls)."""
from __future__ import annotations
import pytest
import registry


class TestVersionLogic:
    def test_next_version_when_empty(self, monkeypatch):
        monkeypatch.setattr(registry, "list_versions", lambda: [])
        assert registry.next_version() == "v0"

    def test_next_version_increments(self, monkeypatch):
        monkeypatch.setattr(registry, "list_versions", lambda: ["v0", "v1", "v2"])
        assert registry.next_version() == "v3"

    def test_next_version_handles_gaps(self, monkeypatch):
        monkeypatch.setattr(registry, "list_versions", lambda: ["v0", "v5"])
        assert registry.next_version() == "v6"
