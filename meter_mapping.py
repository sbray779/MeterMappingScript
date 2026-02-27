#!/usr/bin/env python3
"""
meter_mapping.py
================
Maps token-usage rows from the PTU ChargeBack KQL summary report
(produced by the CreateChargeBackReport Logic App workflow in
github.com/sbray779/PTUChargeBackWorkflow  branch feature/AddResourceID)
to the meter lines in an Azure Cost + Usage export
(e.g. part_0_0001 (3).csv).

Matching strategy
-----------------
1. **ResourceId join (exact, case-insensitive)**
   The feature/AddResourceID branch adds a ``ResourceId`` column to the
   KQL output.  This is the ARM resource ID of the Cognitive Services
   account (e.g. /subscriptions/.../Microsoft.CognitiveServices/accounts/…).
   The same field exists in the cost report.  This is the primary join key.

2. **Deployment name → Meter name fuzzy match**
   Within the set of cost rows that share the same ResourceId the script
   tries to match the ``DeploymentName`` from the token summary to the
   model-family component embedded in the ``meterName`` field of the cost
   report (e.g. "gpt-4o-mini" → "gpt-4o-mini-0718-Outp-regnl Tokens").
   ``rapidfuzz.fuzz.partial_ratio`` is used so that version suffixes in
   the meter name (e.g. "-0718") do not prevent a match.

3. **Input / Output token mapping**
   Meter names that contain "-Inp-" (or "input"/"prompt") are mapped to
   ``PromptTokens`` from the summary.
   Meter names that contain "-Outp-" (or "output"/"completion") are mapped
   to ``CompletionTokens`` from the summary.
   All other meters (e.g. PTU/Batch, embedding) are mapped to
   ``TotalTokens``.

Usage
-----
    python meter_mapping.py \\
        --summary  path/to/chargeback_summary.csv \\
        --cost     "path/to/part_0_0001 (3).csv" \\
        --output   meter_mapping_output.csv

Optional flags:
    --fuzzy-threshold  FLOAT   Minimum rapidfuzz partial_ratio score (0-100).
                               Default: 60.
    --all-cost-rows            Include ALL cost rows for a Cognitive Services
                               resource even when the meter is not matched to
                               a deployment.  Useful for auditing.

Output CSV columns
------------------
All original summary columns are preserved, then the following are appended:
    matched_meter_name          – meterName from the cost report
    meter_type                  – "input" | "output" | "total" | "unknown"
    token_count_for_meter       – PromptTokens / CompletionTokens / TotalTokens
    cost_usd                    – costInUsd from the cost report
    cost_billing_currency       – costInBillingCurrency from the cost report
    cost_quantity               – quantity from the cost report
    cost_unit_of_measure        – unitOfMeasure from the cost report
    cost_date                   – date from the cost report
    billing_period_start        – billingPeriodStartDate from the cost report
    billing_period_end          – billingPeriodEndDate from the cost report
    cost_resource_id            – ResourceId from the cost report (for verification)
    deployment_match_score      – rapidfuzz score (0-100) for transparency
    cost_match_confidence       – "full_match" | "resource_match_only" |
                                  "no_resource_match"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz, process as rfprocess

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False
    print(
        "WARNING: rapidfuzz is not installed.  Falling back to simple substring "
        "matching.  Install it with:  pip install rapidfuzz",
        file=sys.stderr,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Column name constants
# ─────────────────────────────────────────────────────────────────────────────

# KQL chargeback summary (feature/AddResourceID branch output)
_S_RESOURCE_ID   = "ResourceId"
_S_DEPLOYMENT    = "DeploymentName"
_S_MODEL         = "ModelName"
_S_ACCOUNT       = "AccountName"
_S_SUBSCRIPTION  = "SubscriptionId"
_S_PROMPT        = "PromptTokens"
_S_COMPLETION    = "CompletionTokens"
_S_TOTAL         = "TotalTokens"
_S_CALLS         = "Calls"
_S_PRODUCT       = "ProductId"
_S_LUMA          = "Luma"
_S_WORKSPACE     = "Workspace"
_S_SKU_NAME      = "SkuName"
_S_SKU_CAP       = "SkuCapacity"
_S_BACKEND_ID    = "BackendId"
_S_ENDPOINT      = "Endpoint"
_S_FIRST_SEEN    = "FirstSeen"
_S_LAST_SEEN     = "LastSeen"
_S_REGIONS       = "Regions"
_S_CALLERS       = "CallerIpAddresses"

# Azure Cost + Usage export
_C_RESOURCE_ID   = "ResourceId"
_C_METER_NAME    = "meterName"
_C_METER_CAT     = "meterCategory"
_C_METER_SUB     = "meterSubCategory"
_C_CONSUMED_SVC  = "consumedService"
_C_COST_USD      = "costInUsd"
_C_COST_BILLING  = "costInBillingCurrency"
_C_QUANTITY      = "quantity"
_C_UNIT          = "unitOfMeasure"
_C_DATE          = "date"
_C_BP_START      = "billingPeriodStartDate"
_C_BP_END        = "billingPeriodEndDate"
_C_SUBSCRIPTION  = "SubscriptionId"
_C_RG            = "resourceGroupName"

# Services / categories that indicate AI / Cognitive Services billing rows
_AI_CONSUMED_SERVICES = {"microsoft.cognitiveservices"}
_AI_METER_CATEGORIES  = {"foundry models", "cognitive services", "azure openai"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm_rid(rid: object) -> str:
    """Normalise a resource ID for case-insensitive comparison."""
    if not isinstance(rid, str) or not rid.strip():
        return ""
    return rid.strip().lower().rstrip("/")


def _is_input_meter(meter_name: str) -> bool:
    """True when the meter measures prompt / input tokens."""
    m = meter_name.lower()
    return "-inp-" in m or "input" in m or "prompt" in m


def _is_output_meter(meter_name: str) -> bool:
    """True when the meter measures completion / output tokens."""
    m = meter_name.lower()
    return "-outp-" in m or "output" in m or "completion" in m


_RE_TOKENS      = re.compile(r"\s*[Tt]okens\s*$")
_RE_BATCH_IO    = re.compile(r"(-[Bb]atch)?-[IiOo]n?ut?p-\S+$")
_RE_DIRECTION   = re.compile(r"(-[Bb]atch)?-(?:[Ii]np|[Oo]utp)-\S+$")
_RE_SCOPE       = re.compile(r"-(glbl|regnl)$", re.IGNORECASE)


def _meter_model_key(meter_name: str) -> str:
    """
    Strip known suffixes from a meter name to expose the model-family key
    used for fuzzy matching against a deployment name.

    Examples
    --------
    "gpt-4o-mini-0718-Outp-regnl Tokens"  →  "gpt-4o-mini-0718"
    "gpt-4o-mini-0718-Batch-Inp-glbl Tokens" → "gpt-4o-mini-0718"
    "text-embedding-3-large-glbl Tokens"   →  "text-embedding-3-large"
    "embedding-ada-glbl Tokens"            →  "embedding-ada"
    """
    key = _RE_TOKENS.sub("", meter_name)
    key = _RE_DIRECTION.sub("", key)
    key = _RE_SCOPE.sub("", key)
    return key.strip()


def _fuzzy_match_meters(
    deployment: str,
    candidate_meters: list[str],
    threshold: float,
) -> list[tuple[str, float]]:
    """
    Return a list of (meter_name, score) pairs from *candidate_meters* whose
    stripped model key fuzzy-matches *deployment* above *threshold*.

    Prioritises rapidfuzz when available; falls back to substring match.
    """
    dep = deployment.lower().strip()
    # Group meters by their stripped key so we do one comparison per key
    key_to_meters: dict[str, list[str]] = {}
    for mn in candidate_meters:
        k = _meter_model_key(mn).lower()
        key_to_meters.setdefault(k, []).append(mn)

    matched: list[tuple[str, float]] = []

    if _HAS_RAPIDFUZZ:
        results = rfprocess.extract(
            dep,
            list(key_to_meters.keys()),
            scorer=fuzz.partial_ratio,
            limit=None,
            score_cutoff=threshold,
        )
        for key, score, _ in results:
            for mn in key_to_meters[key]:
                matched.append((mn, float(score)))
    else:
        # Simple substring fallback
        for key, meters in key_to_meters.items():
            if dep in key or key in dep:
                for mn in meters:
                    matched.append((mn, 100.0))

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cost_report(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Azure Cost + Usage CSV.

    Returns
    -------
    full_df   – complete DataFrame (all services)
    ai_df     – filtered to Cognitive Services / AI rows only
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Ensure expected columns exist (handle case differences gracefully)
    for col in [_C_RESOURCE_ID, _C_METER_NAME, _C_CONSUMED_SVC, _C_METER_CAT]:
        if col not in df.columns:
            # Try case-insensitive lookup
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                df.rename(columns={matches[0]: col}, inplace=True)
            else:
                df[col] = ""

    df["_rid_norm"] = df[_C_RESOURCE_ID].apply(_norm_rid)

    mask = (
        df[_C_CONSUMED_SVC].str.lower().str.strip().isin(_AI_CONSUMED_SERVICES)
        | df[_C_METER_CAT].str.lower().str.strip().isin(_AI_METER_CATEGORIES)
    )
    ai_df = df[mask].copy()
    return df, ai_df


def load_summary_report(path: str) -> pd.DataFrame:
    """Load the KQL chargeback summary CSV."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if _S_RESOURCE_ID not in df.columns:
        raise ValueError(
            f"Column '{_S_RESOURCE_ID}' not found in summary CSV.\n"
            "This column is added by the feature/AddResourceID branch of "
            "github.com/sbray779/PTUChargeBackWorkflow.\n"
            f"Available columns: {list(df.columns)}"
        )
    df["_rid_norm"] = df[_S_RESOURCE_ID].apply(_norm_rid)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core mapping logic
# ─────────────────────────────────────────────────────────────────────────────

def build_mapping(
    summary_df: pd.DataFrame,
    ai_cost_df: pd.DataFrame,
    fuzzy_threshold: float = 60.0,
    include_all_cost_rows: bool = False,
) -> pd.DataFrame:
    """
    Produce a flat DataFrame that marries every token-summary row to its
    corresponding cost-report meter rows.

    Parameters
    ----------
    summary_df          – loaded KQL chargeback summary
    ai_cost_df          – AI/Cognitive Services rows from the cost report
    fuzzy_threshold     – minimum rapidfuzz score to accept a meter match
    include_all_cost_rows – when True, unmatched cost rows for a ResourceId
                            are still emitted (with confidence "resource_match_only")

    Returns
    -------
    DataFrame with one output row per (summary row × matched cost meter row).
    """
    # Index AI cost rows by normalised resource ID
    cost_by_rid: dict[str, pd.DataFrame] = {}
    for rid_norm, group in ai_cost_df.groupby("_rid_norm"):
        cost_by_rid[rid_norm] = group

    rows: list[dict] = []

    for _, srow in summary_df.iterrows():
        rid         = srow["_rid_norm"]
        deployment  = str(srow.get(_S_DEPLOYMENT, "")).strip()
        prompt_tok  = _safe_float(srow.get(_S_PROMPT))
        compl_tok   = _safe_float(srow.get(_S_COMPLETION))
        total_tok   = _safe_float(srow.get(_S_TOTAL))
        base         = srow.to_dict()

        # ── Step 1: ResourceId lookup ──────────────────────────────────────
        if rid not in cost_by_rid:
            rows.append({
                **base,
                "matched_meter_name":       None,
                "meter_type":               None,
                "token_count_for_meter":    None,
                "cost_usd":                 0.0,
                "cost_billing_currency":    0.0,
                "cost_quantity":            None,
                "cost_unit_of_measure":     None,
                "cost_date":                None,
                "billing_period_start":     None,
                "billing_period_end":       None,
                "cost_resource_id":         None,
                "deployment_match_score":   None,
                "cost_match_confidence":    "no_resource_match",
            })
            continue

        resource_rows   = cost_by_rid[rid]
        candidate_meters = resource_rows[_C_METER_NAME].dropna().unique().tolist()

        # ── Step 2: Fuzzy deployment → meter match ─────────────────────────
        meter_scores = _fuzzy_match_meters(deployment, candidate_meters, fuzzy_threshold)

        if not meter_scores and include_all_cost_rows:
            # Emit all cost rows for this resource with low-confidence label
            for _, crow in resource_rows.iterrows():
                rows.append({**base, **_cost_cols(crow), "meter_type": "unknown",
                             "token_count_for_meter": total_tok,
                             "deployment_match_score": 0,
                             "cost_match_confidence": "resource_match_only"})
            continue

        if not meter_scores:
            rows.append({
                **base,
                "matched_meter_name":       None,
                "meter_type":               None,
                "token_count_for_meter":    total_tok,
                "cost_usd":                 0.0,
                "cost_billing_currency":    0.0,
                "cost_quantity":            None,
                "cost_unit_of_measure":     None,
                "cost_date":                None,
                "billing_period_start":     None,
                "billing_period_end":       None,
                "cost_resource_id":         None,
                "deployment_match_score":   0,
                "cost_match_confidence":    "no_meter_match",
            })
            continue

        # ── Step 3: Input / Output assignment ──────────────────────────────
        matched_names = {mn for mn, _ in meter_scores}
        score_map     = {mn: sc for mn, sc in meter_scores}

        for mn in matched_names:
            meter_subset = resource_rows[resource_rows[_C_METER_NAME] == mn]
            for _, crow in meter_subset.iterrows():
                if _is_input_meter(mn):
                    mtype = "input"
                    tok   = prompt_tok
                elif _is_output_meter(mn):
                    mtype = "output"
                    tok   = compl_tok
                else:
                    mtype = "total"
                    tok   = total_tok

                rows.append({
                    **base,
                    **_cost_cols(crow),
                    "meter_type":            mtype,
                    "token_count_for_meter": tok,
                    "deployment_match_score": score_map.get(mn, 0),
                    "cost_match_confidence": "full_match",
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _cost_cols(crow: pd.Series) -> dict:
    """Extract the relevant cost-report columns into a flat dict."""
    return {
        "matched_meter_name":    crow.get(_C_METER_NAME),
        "cost_usd":              _safe_float(crow.get(_C_COST_USD)),
        "cost_billing_currency": _safe_float(crow.get(_C_COST_BILLING)),
        "cost_quantity":         crow.get(_C_QUANTITY),
        "cost_unit_of_measure":  crow.get(_C_UNIT),
        "cost_date":             crow.get(_C_DATE),
        "billing_period_start":  crow.get(_C_BP_START),
        "billing_period_end":    crow.get(_C_BP_END),
        "cost_resource_id":      crow.get(_C_RESOURCE_ID),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map PTU ChargeBack token-usage summary rows to Azure Cost + Usage meter lines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--summary",
        required=True,
        metavar="PATH",
        help="Path to the KQL chargeback token-summary CSV (from CreateChargeBackReport).",
    )
    parser.add_argument(
        "--cost",
        required=True,
        metavar="PATH",
        help="Path to the Azure Cost + Usage CSV (e.g. 'part_0_0001 (3).csv').",
    )
    parser.add_argument(
        "--output",
        default="meter_mapping_output.csv",
        metavar="PATH",
        help="Output CSV path.  Default: meter_mapping_output.csv",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=60.0,
        metavar="SCORE",
        help=(
            "Minimum rapidfuzz partial_ratio score (0–100) to accept a "
            "deployment-name → meter-name match.  Default: 60."
        ),
    )
    parser.add_argument(
        "--all-cost-rows",
        action="store_true",
        help=(
            "Emit all cost rows for a matched ResourceId even when the "
            "meter name cannot be fuzzy-matched to a deployment.  "
            "Useful for auditing / gap analysis."
        ),
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading cost report:   {args.cost}")
    if not Path(args.cost).exists():
        sys.exit(f"ERROR: Cost report not found: {args.cost}")
    _, ai_cost_df = load_cost_report(args.cost)
    print(f"  AI/Cognitive rows:   {len(ai_cost_df)}")
    if ai_cost_df.empty:
        print(
            "  WARNING: No Cognitive Services / OpenAI rows found.\n"
            "  Check that consumedService == 'Microsoft.CognitiveServices'\n"
            "  or meterCategory matches one of: Foundry Models, Cognitive Services, Azure OpenAI."
        )

    print(f"\nLoading token summary: {args.summary}")
    if not Path(args.summary).exists():
        sys.exit(f"ERROR: Summary report not found: {args.summary}")
    summary_df = load_summary_report(args.summary)
    print(f"  Token summary rows:  {len(summary_df)}")

    # ── Run mapping ───────────────────────────────────────────────────────────
    print(f"\nRunning meter mapping  (fuzzy threshold = {args.fuzzy_threshold}) …")
    result_df = build_mapping(
        summary_df,
        ai_cost_df,
        fuzzy_threshold=args.fuzzy_threshold,
        include_all_cost_rows=args.all_cost_rows,
    )

    # Drop internal normalisation helper columns
    result_df.drop(columns=["_rid_norm"], errors="ignore", inplace=True)

    result_df.to_csv(args.output, index=False)
    print(f"\nOutput written to:    {args.output}")
    print(f"Total output rows:    {len(result_df)}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    if not result_df.empty and "cost_match_confidence" in result_df.columns:
        print("\nMatch confidence breakdown:")
        print(result_df["cost_match_confidence"].value_counts().to_string())

        total_cost = result_df.loc[
            result_df["cost_match_confidence"] == "full_match", "cost_usd"
        ].sum()
        print(f"\nTotal mapped cost (USD): ${total_cost:,.4f}")

    if not result_df.empty and "meter_type" in result_df.columns:
        full = result_df[result_df["cost_match_confidence"] == "full_match"]
        if not full.empty:
            print("\nMeter type breakdown (full_match rows):")
            print(full["meter_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
