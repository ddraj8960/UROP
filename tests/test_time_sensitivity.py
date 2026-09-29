"""
Unit tests for WP1: Time sensitivity detection module.
"""
from __future__ import annotations

import pytest
from hmafact.claims.time_sensitive import detect_time_sensitivity


def test_time_sensitivity_positive_cases():
    cases = [
        ("Who is the current Chief Minister of Tamil Nadu?", True, "current"),
        ("What is the current population of Tokyo?", True, "current"),
        ("Who is currently the CEO of Apple?", True, "current"),
        ("What is the stock price of Tesla today?", True, "current"),
        ("Who is the present prime minister of UK?", True, "current"),
        ("Who is the latest president of France?", True, "current"),
        ("Who is the incumbent governor of California?", True, "current"),
        ("What is the inflation rate as of 2026?", True, "current"),
        ("Who is the CM of Tamilnadu in 2026?", True, "current"),
        ("Who is the CM of Tamil Nadu?", True, "current"),
    ]
    for text, expected_ts, expected_scope in cases:
        is_ts, scope = detect_time_sensitivity(text)
        assert is_ts == expected_ts, f"Failed for '{text}': got is_ts={is_ts}"
        assert scope == expected_scope, f"Failed for '{text}': got scope={scope}"


def test_time_sensitivity_past_cases():
    cases = [
        ("Who was the former Chief Minister of Tamil Nadu?", True, "past"),
        ("Who was the president of USA in 1998?", True, "past"),
        ("Who was the ex-CEO of Twitter?", True, "past"),
        ("Who served as prime minister in 2010?", True, "past"),
        ("What was the capital of West Germany in 1980?", True, "past"),
    ]
    for text, expected_ts, expected_scope in cases:
        is_ts, scope = detect_time_sensitivity(text)
        assert is_ts == expected_ts, f"Failed for '{text}': got is_ts={is_ts}"
        assert scope == expected_scope, f"Failed for '{text}': got scope={scope}"


def test_time_sensitivity_negative_cases():
    cases = [
        ("What is the speed of light?", False, "unspecified"),
        ("Where is the Eiffel Tower located?", False, "unspecified"),
        ("What is the chemical formula of water?", False, "unspecified"),
        ("Who painted the Mona Lisa?", False, "unspecified"),
        ("What is the distance from Earth to Mars?", False, "unspecified"),
    ]
    for text, expected_ts, expected_scope in cases:
        is_ts, scope = detect_time_sensitivity(text)
        assert is_ts == expected_ts, f"Failed for '{text}': got is_ts={is_ts}"
        assert scope == expected_scope, f"Failed for '{text}': got scope={scope}"
