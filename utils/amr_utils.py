"""
utils/amr_utils.py — Shared AMR extraction and preprocessing utilities.
"""
from __future__ import annotations

import re


def extract_thinking(response: str) -> str:
    """Extract thinking process before </think> tag."""
    if "</think>" in response:
        thinking = response.split("</think>", 1)[0]
        thinking = thinking.replace("<think>", "")
        return thinking.strip()

    if "<think>" in response:
        thinking = response.split("<think>", 1)[1]
        return thinking.strip()

    # If no tags, assume thinking is everything except the AMR block
    thinking = re.sub(r"<amr>.*?</amr>", "", response, flags=re.DOTALL)
    return thinking.strip()


def extract_amr(response: str) -> str | None:
    """Extract content between <amr>...</amr> tags."""
    match = re.search(r"<amr>(.*?)</amr>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: capture anything after <amr> if </amr> is missing due to truncation
    if "<amr>" in response:
        return response.split("<amr>", 1)[1].strip()
    return None


def fix_amr_parentheses(amr_str: str) -> str:
    """
    Balances parentheses in an AMR string.
    Appends missing closing parentheses or removes extra ones from the end.
    """
    if not amr_str:
        return amr_str

    open_count = amr_str.count('(')
    close_count = amr_str.count(')')

    if open_count > close_count:
        amr_str += ')' * (open_count - close_count)
    elif close_count > open_count:
        diff = close_count - open_count
        for _ in range(diff):
            idx = amr_str.rfind(')')
            if idx != -1:
                amr_str = amr_str[:idx] + amr_str[idx + 1:]

    return amr_str
