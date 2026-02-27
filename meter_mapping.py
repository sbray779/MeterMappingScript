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

2. **Model name → Meter name fuzzy match**
   Within the set of cost rows that share the same ResourceId the script
   tries to match the ``ModelName`` from the token summary to the
   model-family component embedded in the ``meterName`` field of the cost
   report (e.g. "gpt-4o-mini" → "gpt-4o-mini-0718-Outp-regnl Tokens").
   ``rapidfuzz.fuzz.partial_ratio`` is used so that version suffixes in
   the meter name (e.g. "-0718") do not prevent a match.
   ``DeploymentName`` is retained in the output for reference only.

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
    input_tokens                 – PromptTokens for this row
    output_tokens                – CompletionTokens for this row
    cost_usd                    – costInUsd from the cost report
    cost_billing_currency       – costInBillingCurrency from the cost report
    cost_quantity               – quantity from the cost report
    cost_unit_of_measure        – unitOfMeasure from the cost report
    cost_date                   – date from the cost report
    billing_period_start        – billingPeriodStartDate from the cost report
    billing_period_end          – billingPeriodEndDate from the cost report
    cost_resource_id            – ResourceId from the cost report (for verification)
    model_match_score           – rapidfuzz score (0-100) for the ModelName → meterName match
    cost_match_confidence       – "full_match" | "resource_match_only" |
                                  "no_resource_match"
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Optional

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

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    _HAS_AZURE_STORAGE = True
except ImportError:  # pragma: no cover
    _HAS_AZURE_STORAGE = False

# ─────────────────────────────────────────────────────────────────────────────
# Column name constants
# ─────────────────────────────────────────────────────────────────────────────

# KQL chargeback summary (feature/AddResourceID branch output)
# The column may appear as "ResourceId" or "ResourceID" depending on the branch;
# _norm_col() is used at load time to canonicalise it.
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


# ── SKU name → meter scope filtering ─────────────────────────────────────────
#
# Azure OpenAI SkuName values (as returned by the ARM API / chargeback report)
# and the meter-name scope tokens they map to in the Cost + Usage export:
#
#  SkuName                    Meter contains          Does NOT contain
#  ─────────────────────────────────────────────────────────────────────
#  Standard                   -regnl                  -glbl, -Batch-
#  GlobalStandard             -glbl                   -Batch-
#  GlobalBatch                -Batch-  AND  -glbl      —
#  DataZoneStandard           -dzn                    —
#  DataZoneBatch              -Batch-  AND  -dzn       —
#  ProvisionedManaged         (no scope suffix / PTU meters)
#  GlobalProvisionedManaged   (no scope suffix / PTU meters)
#
# Reference: https://learn.microsoft.com/azure/ai-services/openai/quotas-limits
#            and Azure OpenAI deployment types documentation.

_RE_SCOPE_GLBL  = re.compile(r"-glbl",  re.IGNORECASE)
_RE_SCOPE_REGNL = re.compile(r"-regnl", re.IGNORECASE)
_RE_SCOPE_DZN   = re.compile(r"-dzn",   re.IGNORECASE)
_RE_BATCH_TOKEN = re.compile(r"-Batch-", re.IGNORECASE)


def _filter_meters_by_sku(meters: list[str], sku_name: str) -> list[str]:
    """
    Return the subset of *meters* whose scope component matches the deployment
    *sku_name*.

    If the SKU is unknown or no meters survive the filter, the original list
    is returned unchanged (fail-open) so that the fuzzy match still runs.
    """
    sku = sku_name.strip().lower() if sku_name else ""

    if sku == "standard":
        # Regional pay-as-you-go → must contain -regnl, must NOT be batch
        filtered = [
            m for m in meters
            if _RE_SCOPE_REGNL.search(m) and not _RE_BATCH_TOKEN.search(m)
        ]
    elif sku == "globalstandard":
        # Global pay-as-you-go → must contain -glbl, must NOT be batch
        filtered = [
            m for m in meters
            if _RE_SCOPE_GLBL.search(m) and not _RE_BATCH_TOKEN.search(m)
        ]
    elif sku == "globalbatch":
        # Global batch → must contain both -Batch- and -glbl
        filtered = [
            m for m in meters
            if _RE_BATCH_TOKEN.search(m) and _RE_SCOPE_GLBL.search(m)
        ]
    elif sku == "datazonestandard":
        # Data-zone pay-as-you-go → must contain -dzn, must NOT be batch
        filtered = [
            m for m in meters
            if _RE_SCOPE_DZN.search(m) and not _RE_BATCH_TOKEN.search(m)
        ]
    elif sku == "datazonebatch":
        # Data-zone batch → must contain both -Batch- and -dzn
        filtered = [
            m for m in meters
            if _RE_BATCH_TOKEN.search(m) and _RE_SCOPE_DZN.search(m)
        ]
    elif sku in ("provisionedmanaged", "globalprovisionedmanaged"):
        # PTU meters typically lack the Inp-/Outp- direction component;
        # exclude metered (regnl/glbl) and batch meters.
        filtered = [
            m for m in meters
            if not _RE_SCOPE_REGNL.search(m)
            and not _RE_SCOPE_GLBL.search(m)
            and not _RE_SCOPE_DZN.search(m)
            and not _RE_BATCH_TOKEN.search(m)
        ]
    else:
        # Unknown SKU – return everything and let the caller decide
        filtered = meters

    # Fail-open: if the filter removed everything, revert to the full list
    return filtered if filtered else meters


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

# ─────────────────────────────────────────────────────────────────────────────
# Azure Blob Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_blob_service_client(storage_url: str) -> "BlobServiceClient":
    """
    Return a BlobServiceClient for *storage_url*.

    Authentication precedence
    -------------------------
    1. If the URL already embeds a SAS token (contains ``?sv=`` or ``?se=``),
       use it as-is (anonymous / SAS auth).
    2. Otherwise fall back to ``DefaultAzureCredential`` (works with managed
       identity, az login, environment variables, workload identity, etc.).
    """
    if not _HAS_AZURE_STORAGE:
        raise RuntimeError(
            "azure-storage-blob and azure-identity are required for storage access.\n"
            "Install them with:  pip install azure-storage-blob azure-identity"
        )
    if "?" in storage_url:  # SAS token already embedded
        return BlobServiceClient(account_url=storage_url)
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=storage_url, credential=credential)


def _read_blob_csv(client: "BlobServiceClient", container: str, blob_name: str) -> pd.DataFrame:
    """Download a single blob and parse it as a CSV DataFrame."""
    print(f"  Downloading blob: {container}/{blob_name}")
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    data = blob_client.download_blob().readall()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def load_token_chunks_from_storage(
    storage_url: str,
    container: str,
    prefix: str = "",
) -> pd.DataFrame:
    """
    List all blobs in *container* (optionally filtered by *prefix*), download
    every CSV blob, and return the concatenated DataFrame.

    The 24-hour chargeback period is expected to be split into 12 × 2-hour
    chunk files.  All matching blobs are loaded regardless of filename ordering;
    the caller concatenates them into a single summary DataFrame.
    """
    client = _get_blob_service_client(storage_url)
    container_client = client.get_container_client(container)

    blobs = [
        b.name
        for b in container_client.list_blobs(name_starts_with=prefix or None)
        if b.name.lower().endswith(".csv")
    ]

    if not blobs:
        raise FileNotFoundError(
            f"No CSV blobs found in container '{container}' "
            f"(storage: {storage_url}, prefix: '{prefix or '*'}')."
        )

    print(f"  Found {len(blobs)} chunk CSV blob(s) in '{container}' (prefix='{prefix}')")
    frames: list[pd.DataFrame] = []
    for blob_name in sorted(blobs):
        df_chunk = _read_blob_csv(client, container, blob_name)
        frames.append(df_chunk)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined token-usage rows: {len(combined)}")
    return combined


def load_cost_report_from_storage(
    storage_url: str,
    container: str,
    blob_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the Azure Cost + Usage CSV from Blob Storage and return the
    same (full_df, ai_df) tuple as :func:`load_cost_report`.

    If *blob_name* is ``None``, the container is listed and the first CSV blob
    found (alphabetically) is used.  Raises ``FileNotFoundError`` if the
    container contains no CSV blobs.
    """
    client = _get_blob_service_client(storage_url)
    if blob_name is None:
        container_client = client.get_container_client(container)
        csv_blobs = sorted(
            b.name
            for b in container_client.list_blobs()
            if b.name.lower().endswith(".csv")
        )
        if not csv_blobs:
            raise FileNotFoundError(
                f"No CSV blobs found in cost container '{container}' ({storage_url})."
            )
        blob_name = csv_blobs[0]
        print(f"  Auto-detected cost blob: {blob_name}")
    df = _read_blob_csv(client, container, blob_name)
    return _process_cost_dataframe(df)


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
    return _process_cost_dataframe(df)


def _process_cost_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Common post-processing for a cost-report DataFrame loaded from any source.

    Returns
    -------
    full_df   – complete DataFrame (all services)
    ai_df     – filtered to Cognitive Services / AI rows only
    """
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


def _prepare_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common post-processing for a token-usage summary DataFrame loaded from
    any source (local file or Blob Storage).

    Normalises the ResourceId column name (the chunk files use ``ResourceID``
    with a capital D; we canonicalise it to ``ResourceId``).
    """
    df.columns = df.columns.str.strip()

    # Canonicalise ResourceID → ResourceId (case-insensitive)
    if _S_RESOURCE_ID not in df.columns:
        matches = [c for c in df.columns if c.lower() == _S_RESOURCE_ID.lower()]
        if matches:
            df.rename(columns={matches[0]: _S_RESOURCE_ID}, inplace=True)
        else:
            raise ValueError(
                f"Column '{_S_RESOURCE_ID}' (or 'ResourceID') not found in summary CSV.\n"
                "This column is added by the feature/AddResourceID branch of "
                "github.com/sbray779/PTUChargeBackWorkflow.\n"
                f"Available columns: {list(df.columns)}"
            )

    df["_rid_norm"] = df[_S_RESOURCE_ID].apply(_norm_rid)
    return df


def load_summary_report(path: str) -> pd.DataFrame:
    """Load the KQL chargeback summary CSV from a local file path."""
    df = pd.read_csv(path, low_memory=False)
    return _prepare_summary_dataframe(df)


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
    Produce a flat DataFrame: one output row per summary row, with all
    original token-usage columns preserved and two extra columns appended:
      input_meter_name  – best matching input meter from the cost report
      output_meter_name – best matching output meter from the cost report
    """
    # Index AI cost rows by normalised resource ID
    cost_by_rid: dict[str, pd.DataFrame] = {}
    for rid_norm, group in ai_cost_df.groupby("_rid_norm"):
        cost_by_rid[rid_norm] = group

    rows: list[dict] = []

    for _, srow in summary_df.iterrows():
        rid        = srow["_rid_norm"]
        model_name = str(srow.get(_S_MODEL, "")).strip()
        base       = srow.to_dict()

        # ── Step 1: ResourceId lookup ──────────────────────────────────────
        if rid not in cost_by_rid:
            rows.append({**base,
                         "input_meter_name":  None,
                         "output_meter_name": None,
                         "cost_match_confidence": "no_resource_match"})
            continue

        resource_rows    = cost_by_rid[rid]
        sku_name         = str(srow.get(_S_SKU_NAME, "")).strip()
        all_meters       = resource_rows[_C_METER_NAME].dropna().unique().tolist()
        candidate_meters = _filter_meters_by_sku(all_meters, sku_name)

        # ── Step 2: Fuzzy model name → meter match ────────────────────────
        meter_scores = _fuzzy_match_meters(model_name, candidate_meters, fuzzy_threshold)

        if not meter_scores:
            rows.append({**base,
                         "input_meter_name":  None,
                         "output_meter_name": None,
                         "cost_match_confidence": "no_meter_match"})
            continue

        score_map     = {mn: sc for mn, sc in meter_scores}
        matched_names = list(score_map)

        # ── Step 3: Pick best input and best output meter ──────────────────
        inp_candidates = [(mn, score_map[mn]) for mn in matched_names if _is_input_meter(mn)]
        out_candidates = [(mn, score_map[mn]) for mn in matched_names if _is_output_meter(mn)]
        oth_candidates = [(mn, score_map[mn]) for mn in matched_names
                          if not _is_input_meter(mn) and not _is_output_meter(mn)]

        best_input  = max(inp_candidates, key=lambda x: x[1])[0] if inp_candidates else None
        best_output = max(out_candidates, key=lambda x: x[1])[0] if out_candidates else None
        # PTU/embedding: no directional meters — put best match in input slot
        if best_input is None and best_output is None and oth_candidates:
            best_input = max(oth_candidates, key=lambda x: x[1])[0]

        rows.append({**base,
                     "input_meter_name":      best_input,
                     "output_meter_name":     best_output,
                     "cost_match_confidence": "full_match"})

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

    # ── Local-file source (original behaviour) ────────────────────────────────
    local_grp = parser.add_argument_group(
        "local file sources",
        "Load input data from local CSV files (use instead of storage options).",
    )
    local_grp.add_argument(
        "--summary",
        metavar="PATH",
        nargs="+",
        help="One or more KQL chargeback token-summary CSVs (from CreateChargeBackReport). All files are concatenated before mapping.",
    )
    local_grp.add_argument(
        "--cost",
        metavar="PATH",
        help="Path to the Azure Cost + Usage CSV (e.g. 'part_0_0001 (3).csv').",
    )

    # ── Azure Blob Storage source for token-usage chunks ──────────────────────
    tok_grp = parser.add_argument_group(
        "token-usage storage",
        (
            "Pull token-usage data from Azure Blob Storage.  The container is expected "
            "to hold 12 x 2-hour CSV chunk files covering a 24-hour period.  "
            "Authenticate via a SAS token embedded in the URL or DefaultAzureCredential."
        ),
    )
    tok_grp.add_argument(
        "--token-storage-url",
        metavar="URL",
        help=(
            "Azure Storage account URL for the token-usage chunk files, e.g. "
            "'https://myaccount.blob.core.windows.net' or a SAS URL that includes '?sv=…'."
        ),
    )
    tok_grp.add_argument(
        "--token-container",
        metavar="CONTAINER",
        help="Container name that holds the 12 x 2-hour chunk CSV blobs.",
    )
    tok_grp.add_argument(
        "--token-prefix",
        metavar="PREFIX",
        default="",
        help=(
            "Optional blob-name prefix used to filter chunk files within the container "
            "(e.g. 'chargeBack-chunk-').  Defaults to listing all CSV blobs."
        ),
    )

    # ── Azure Blob Storage source for cost + usage CSV ────────────────────────
    cost_grp = parser.add_argument_group(
        "cost-usage storage",
        (
            "Pull the Azure Cost + Usage CSV from Azure Blob Storage.  "
            "Authenticate via a SAS token embedded in the URL or DefaultAzureCredential."
        ),
    )
    cost_grp.add_argument(
        "--cost-storage-url",
        metavar="URL",
        help=(
            "Azure Storage account URL for the cost + usage CSV, e.g. "
            "'https://myaccount.blob.core.windows.net' or a SAS URL."
        ),
    )
    cost_grp.add_argument(
        "--cost-container",
        metavar="CONTAINER",
        help="Container name that holds the cost + usage CSV blob.",
    )
    cost_grp.add_argument(
        "--cost-blob",
        metavar="BLOB",
        help="Blob name of the cost + usage CSV within the container. If omitted, the first CSV blob in the container is used automatically.",
    )

    # ── Common options ────────────────────────────────────────────────────────
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

    # ── Validate source arguments ─────────────────────────────────────────────
    use_token_storage = bool(args.token_storage_url)
    use_cost_storage  = bool(args.cost_storage_url)

    if use_token_storage:
        if not args.token_container:
            sys.exit("ERROR: --token-container is required when --token-storage-url is set.")
    else:
        if not args.summary:
            sys.exit(
                "ERROR: Provide either --summary (local file) "
                "or --token-storage-url / --token-container (Azure Storage)."
            )

    if use_cost_storage:
        if not args.cost_container:
            sys.exit("ERROR: --cost-container is required when --cost-storage-url is set.")
        # --cost-blob is optional; auto-detected from container when omitted
    else:
        if not args.cost:
            sys.exit(
                "ERROR: Provide either --cost (local file) "
                "or --cost-storage-url / --cost-container / --cost-blob (Azure Storage)."
            )

    # ── Load cost data ────────────────────────────────────────────────────────
    if use_cost_storage:
        print(
            f"Loading cost report from storage: {args.cost_storage_url} "
            f"/ {args.cost_container}" + (f" / {args.cost_blob}" if args.cost_blob else " (auto-detecting blob)")
        )
        _, ai_cost_df = load_cost_report_from_storage(
            args.cost_storage_url, args.cost_container, args.cost_blob or None
        )
    else:
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

    # ── Load token-usage data ─────────────────────────────────────────────────
    if use_token_storage:
        print(
            f"\nLoading token-usage chunks from storage: {args.token_storage_url} "
            f"/ {args.token_container} (prefix='{args.token_prefix}')"
        )
        raw_df   = load_token_chunks_from_storage(
            args.token_storage_url, args.token_container, args.token_prefix
        )
        summary_df = _prepare_summary_dataframe(raw_df)
    else:
        print(f"\nLoading token summary files: {', '.join(args.summary)}")
        frames = []
        for p in args.summary:
            if not Path(p).exists():
                sys.exit(f"ERROR: Summary report not found: {p}")
            frames.append(load_summary_report(p))
        summary_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

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


if __name__ == "__main__":
    main()
