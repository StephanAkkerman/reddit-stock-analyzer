"""Tests for the subreddit catalogue and credential loading."""

from __future__ import annotations

import pytest

from reddit_stock_analyzer.config import (
    DEFAULT_SUBREDDITS,
    SUBREDDIT_CATEGORIES,
    all_subreddits,
    is_valid_subreddit_name,
    reddit_credentials_from_env,
    subreddits_for,
)


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("wallstreetbets", True),
        ("Daytrading", True),
        ("penny_stocks", True),
        ("ab", False),  # too short
        ("r/stocks", False),  # prefix is not part of the name
        ("stocks!", False),
        ("", False),
        ("x" * 33, False),
    ],
)
def test_is_valid_subreddit_name(name, valid):
    assert is_valid_subreddit_name(name) is valid


class TestCatalogue:
    def test_every_catalogued_name_is_valid(self):
        assert all(is_valid_subreddit_name(n) for n in all_subreddits())

    def test_defaults_are_catalogued(self):
        assert set(DEFAULT_SUBREDDITS) <= set(all_subreddits())

    def test_no_arguments_returns_the_defaults(self):
        assert subreddits_for() == DEFAULT_SUBREDDITS

    def test_categories_are_merged_without_duplicates(self):
        merged = subreddits_for("stocks", "research")
        assert merged.count("stocks") == 1  # listed in both categories
        assert "SecurityAnalysis" in merged

    def test_order_follows_declaration(self):
        assert subreddits_for("retail") == SUBREDDIT_CATEGORIES["retail"]

    def test_unknown_category_raises(self):
        with pytest.raises(KeyError, match="Unknown subreddit category"):
            subreddits_for("crypto", "nonsense")


class TestCredentials:
    def test_none_without_a_client_id(self):
        assert reddit_credentials_from_env() is None

    def test_modern_names(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
        monkeypatch.setenv("REDDIT_USER_AGENT", "agent")
        assert reddit_credentials_from_env() == {
            "client_id": "id",
            "client_secret": "secret",
            "user_agent": "agent",
        }

    def test_legacy_names_still_work(self, monkeypatch):
        monkeypatch.setenv("REDDIT_PERSONAL_USE", "id")
        monkeypatch.setenv("REDDIT_SECRET", "secret")
        creds = reddit_credentials_from_env()
        assert creds["client_id"] == "id"
        assert creds["client_secret"] == "secret"
        assert creds["user_agent"]  # falls back to the package user agent

    def test_username_and_password_are_only_added_as_a_pair(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
        monkeypatch.setenv("REDDIT_USERNAME", "me")
        assert "username" not in reddit_credentials_from_env()

        monkeypatch.setenv("REDDIT_PASSWORD", "pw")
        creds = reddit_credentials_from_env()
        assert creds["username"] == "me"
        assert creds["password"] == "pw"
