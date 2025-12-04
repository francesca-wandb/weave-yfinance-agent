from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from io import StringIO
from typing import Iterable, List, Tuple


def setup_logging() -> Tuple[logging.Logger, StringIO]:
    """Setup logging for the server."""

    # Create a string buffer to hold the logs
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handler.setLevel(logging.INFO)

    # Attach the handler to the main logger
    logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and h.stream is log_buffer for h in logger.handlers):
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Attach the handler to the Weave logger
    weave_logger = logging.getLogger("weave")
    weave_logger.setLevel(logging.INFO)
    weave_logger.propagate = True

    return logger, log_buffer


# Ignore typical date/time tokens to avoid false numeric matches.
_DATE_TIME_RES: List[re.Pattern[str]] = [
    # ISO-like dates
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # 2025-09-30
    re.compile(r"\b\d{4}/\d{2}/\d{2}\b"),  # 2025/09/30
    # Day/Month/Year or Month/Day/Year with 2- or 4-digit year
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # 09/30/25, 30-09-2025
    # Standalone 4-digit years (treat all as dates for comparison purposes)
    re.compile(r"\b(19|20)\d{2}\b"),  # 1999, 2024, 2030
    # Month name formats
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{2,4})?\b",
        re.IGNORECASE,
    ),  # Sep 30, 2025
    re.compile(
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s+\d{2,4})?\b",
        re.IGNORECASE,
    ),  # 30 Sep 2025
    # Quarter + year (Q3 2025)
    re.compile(r"\bq[1-4]\s+(19|20)\d{2}\b", re.IGNORECASE),
    # Times
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # 14:05 or 14:05:59
]

# Capture optional sign/parentheses, optional currency, sci or std number, optional magnitude suffix.
_NUMBER_RE: re.Pattern[str] = re.compile(
    r"""
    (?P<open>\()?
    (?P<sign>[+\-])?
    (?P<cur>[$€£])?
    (?:
        (?P<sci>\d+(?:\.\d+)?[eE][+\-]?\d+) |
        (?P<std>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    )
    \s*
    (?P<suf>
        [kKmMbBtT] |
        bn | mm |
        thousand | millions? | million | billions? | billion | trillions? | trillion
    )?
    \.?
    (?P<close>\))?
    """,
    re.VERBOSE,
)

# Base factors; plural forms are normalized via rstrip("s") + lower().
_SUFFIX_FACTORS: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mm": 1e6,  # finance shorthand
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,  # finance shorthand
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

# Tolerance accommodates rounding to different decimals
_REL_TOL: float = 1e-1
# Tiny-number fallback
_ABS_TOL: float = 1e-6


def _ensure_output_dir(base_dir: str, sub_dir: str) -> str:
    """Ensure the output directory exists and return its absolute path.

    Args:
        base_dir: The base directory, typically the current working directory.
        sub_dir: The subdirectory name to create under `base_dir`.

    Returns:
        The absolute path to the ensured output directory.
    """
    output_dir = os.path.join(base_dir, sub_dir)
    created = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        created = True

    log_file = os.path.join(output_dir, f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if created:
        logger.info("Created output directory: %s", output_dir)
    else:
        logger.info("Output directory already exists: %s", output_dir)
    logger.info("Logging initialized. Output directory: %s", output_dir)
    return output_dir


def _clean_dates_times(text: str) -> str:
    """Remove date/time substrings so they don't count as numeric values."""
    for rx in _DATE_TIME_RES:
        text = rx.sub(" ", text)
    return text


def _apply_suffix(value: float, suffix: str | None) -> float:
    """Scale by suffix (k/M/B/T/bn/mm/words), accepting plural forms like 'millions'."""
    if not suffix:
        return value
    key = suffix.lower().rstrip("s")
    # Keep single-letter case-insensitive: 'K'/'k', 'M'/'m', etc.
    if key in _SUFFIX_FACTORS:
        return value * _SUFFIX_FACTORS[key]
    if suffix in _SUFFIX_FACTORS:  # single-letter original case
        return value * _SUFFIX_FACTORS[suffix]
    return value


def _parse_match(m: re.Match[str]) -> float:
    """Turn a regex match into a normalized float in base units (suffix applied)."""
    num = m.group("sci") or m.group("std") or "0"
    if m.group("std"):
        num = num.replace(",", "")
    base = float(num)
    val = _apply_suffix(base, m.group("suf"))
    neg = m.group("sign") == "-" or (m.group("open") and m.group("close"))
    return -val if neg else val


def _infer_global_scale(text: str) -> float:
    """
    Infer a global numeric scale from unit hints in the text (billions/millions/thousands).

    Examples that trigger scaling:
      - "Values are reported in USD and shown in billions (USD bn)."
      - "in billions", "USD bn", "(USD bn)"
      - "in millions", "USD mm", "USD m"
      - "in thousands", "USD k"
    """
    s = text.lower()

    # Billions
    if (
        re.search(r"\b(in|shown|reported)\s+in\s+(usd\s*)?billions?\b", s)
        or re.search(r"\b(usd\s*)?bn\b", s)
        or re.search(r"\(\s*usd\s*bn\s*\)", s)
        or re.search(r"\bbillion?\b", s)
        or re.search(r"\bbillions?\b", s)
    ):
        return 1e9

    # Millions
    if (
        re.search(r"\b(in|shown|reported)\s+in\s+(usd\s*)?millions?\b", s)
        or re.search(r"\b(usd\s*)?mm\b", s)
        or re.search(r"\b(usd\s*)?m\b", s)
        or re.search(r"\bmillion?\b", s)
        or re.search(r"\bmillions?\b", s)
    ):
        return 1e6

    # Thousands
    if (
        re.search(r"\b(in|shown|reported)\s+in\s+(usd\s*)?thousands?\b", s)
        or re.search(r"\b(usd\s*)?k\b", s)
        or re.search(r"\bthousands?\b", s)
    ):
        return 1e3

    return 1.0


def _numbers(text: str) -> List[float]:
    """Extract normalized numeric values from arbitrary text."""

    if not text:
        return []
    scale = _infer_global_scale(text)
    cleaned = _clean_dates_times(text)

    vals: List[float] = []
    for m in _NUMBER_RE.finditer(cleaned):
        try:
            # If a '%' immediately follows the numeric token, ignore it
            if m.end() < len(cleaned) and cleaned[m.end()] == "%":
                continue
            # If the token is immediately preceded by '~' or '≈', ignore it
            if m.start() > 0 and cleaned[m.start() - 1] in ("~", "≈", "+", "-", "+$", "-$", "≈ $", "~ $", "≈$", "~$"):
                continue
            # If "about " appears right before the token (within 6 chars), ignore it
            lookback = cleaned[max(0, m.start() - 6) : m.start()].lower()
            if "about " in lookback:
                continue

            v = _parse_match(m)

            # Keep only global-scale tokens w/o per-token suffix
            if not m.group("suf"):
                v *= scale
            vals.append(v)
        except Exception:
            continue
    return vals


def _unique(values: Iterable[float]) -> List[float]:
    """Approximate de-duplication using tolerance buckets to reduce repeated matches."""
    out: List[float] = []
    for v in values:
        if not any(math.isclose(v, x, rel_tol=_REL_TOL, abs_tol=_ABS_TOL) for x in out):
            out.append(v)
    return out


def _fraction_matched(outputs: List[float], contexts: List[float]) -> float:
    """Fraction of output numbers that appear in context within tolerance."""
    if not outputs:
        return 1.0
    if not contexts:
        return 0.0
    ctx = _unique(contexts)
    hits = sum(1 for o in outputs if any(math.isclose(o, c, rel_tol=_REL_TOL, abs_tol=_ABS_TOL) for c in ctx))
    return hits / float(len(outputs))
