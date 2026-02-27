# MeterMapping

Marries the PTU ChargeBack token-usage summary (produced by
[sbray779/PTUChargeBackWorkflow](https://github.com/sbray779/PTUChargeBackWorkflow)
branch `feature/AddResourceID`) to meter lines in an Azure Cost + Usage export.

---

## Why this exists

The `CreateChargeBackReport` Logic App produces a daily CSV (split into 12 × 2-hour
chunk files covering a 24-hour period) that summarises **how many tokens each APIM
product / workspace consumed per Azure OpenAI deployment**.  The report includes the
ARM `ResourceId` of the Cognitive Services account (`feature/AddResourceID` branch).

A separate **Azure Cost + Usage** export from the billing portal contains meter names
broken out by model and input/output direction.

This script joins the two data sets so you can:

* Identify the **input and output meter names** for each token-usage row.
* Narrow meter candidates by **SKU scope** (regional, global, data-zone, batch, PTU).
* Feed the enriched rows into downstream chargeback / cost-allocation pipelines.

---

## How the matching works

| Step | Key | Description |
|------|-----|-------------|
| 1 | **ResourceId** (exact, case-insensitive) | Links a token-summary row to all cost rows for the same Cognitive Services account. |
| 2 | **SKU scope filtering** | Candidate meters are pre-filtered by `SkuName` before fuzzy matching (see table below). |
| 3 | **ModelName → Meter name** (fuzzy) | `rapidfuzz.fuzz.partial_ratio` maps `ModelName` (e.g. `gpt-4o-mini`) to the model component of the meter name (e.g. `gpt-4o-mini-0718-Inp-regnl Tokens`). Version suffixes like `-0718` are tolerated. |
| 4 | **Input / Output selection** | The best-scoring meter containing `-Inp-` becomes `input_meter_name`; the best `-Outp-` meter becomes `output_meter_name`. Both appear on the same output row. |

### How the fuzzy matching works

The fuzzy match uses **`rapidfuzz.fuzz.partial_ratio`**, which checks whether
the query string appears as a contiguous substring of the target, allowing for
minor character differences.  This is important because:

* Meter names include a version suffix the token-usage report does not have
  (e.g. `gpt-4o-mini` vs `gpt-4o-mini-0718-Inp-regnl Tokens`).
* `partial_ratio` scores the *best alignment window*, so `gpt-4o-mini` scores
  100 against `gpt-4o-mini-0718-Inp-regnl Tokens` even though the meter name
  is much longer.

Before scoring, each meter name is stripped of its version, scope, and
direction components to produce a *model key* (e.g. `gpt-4o-mini`) that is
compared against `ModelName` from the token-usage report.

**Threshold** — only meters scoring ≥ `--fuzzy-threshold` (default `60`) are
considered a match.  Lower the threshold if legitimate models are not matching;
raise it if unrelated meters are being selected.

**Tie-breaking** — when multiple meters pass the threshold, the one with the
highest score wins.  Separate winners are chosen for the `-Inp-` (input) and
`-Outp-` (output) directions, so both can appear on the same output row.

### SKU → meter scope mapping

| `SkuName` | Meter scope matched |
|-----------|---------------------|
| `Standard` | `-regnl` (regional), no `-Batch-` |
| `GlobalStandard` | `-glbl` (global), no `-Batch-` |
| `GlobalBatch` | `-Batch-` + `-glbl` |
| `DataZoneStandard` | `-dzn` (data-zone), no `-Batch-` |
| `DataZoneBatch` | `-Batch-` + `-dzn` |
| `ProvisionedManaged` / `GlobalProvisionedManaged` | PTU meters (no scope suffix) |

### Meter name anatomy

```
gpt-4o-mini-0718  -  Inp  -  regnl  Tokens
│                    │        │
│                    │        └─ scope: regnl | glbl | dzn
│                    └────────── direction: Inp (input) or Outp (output)
└────────────────────────────── model + version key
```

For PTU and embedding models the direction component may be absent; the best match
is placed in `input_meter_name`.

---

## Inputs

### 1. Token-summary chunk files

Produced by the `CreateChargeBackReport` Logic App workflow
(**`feature/AddResourceID` branch required** for the `ResourceId` column).
The 24-hour report is split into **12 × 2-hour CSV chunk files**.

Supply them via local paths **or** an Azure Blob Storage container — see [Usage](#usage).

Expected columns (order-independent):

| Column | Description |
|--------|-------------|
| `ResourceId` | ARM resource ID of the Cognitive Services account |
| `ModelName` | Model name (used for fuzzy meter matching) |
| `DeploymentName` | Azure OpenAI deployment name |
| `AccountName` | Cognitive Services account name |
| `SubscriptionId` | Azure subscription |
| `SkuName` | SKU type — drives meter scope filtering |
| `SkuCapacity` | Provisioned capacity (PTU / TPM) |
| `PromptTokens` | Total prompt/input tokens |
| `CompletionTokens` | Total completion/output tokens |
| `TotalTokens` | Sum of prompt + completion tokens |
| `Calls` | Number of API calls |
| `ProductId` | APIM product ID |
| `Luma` | 6-digit org code |
| `Workspace` | Workspace name |
| `BackendId` | APIM backend ID |
| `Endpoint` | Azure OpenAI endpoint URL |
| `FirstSeen` | Earliest request timestamp |
| `LastSeen` | Latest request timestamp |
| `Regions` | Azure regions observed |
| `CallerIpAddresses` | Caller IP addresses |

### 2. Azure Cost + Usage export

Download from **Azure Portal → Cost Management → Exports**.
Supply via a local file path **or** an Azure Blob Storage container — see [Usage](#usage).

Relevant columns used by this script:

| Column | Description |
|--------|-------------|
| `ResourceId` | ARM resource ID |
| `meterName` | E.g. `gpt-4o-mini-0718-Outp-regnl Tokens` |
| `meterCategory` | E.g. `Foundry Models` |
| `consumedService` | E.g. `Microsoft.CognitiveServices` |

---

## Output columns

All original token-summary columns are preserved.  Two columns are appended:

| Column | Description |
|--------|-------------|
| `input_meter_name` | Best-matched input meter name from the cost report (`-Inp-`) |
| `output_meter_name` | Best-matched output meter name from the cost report (`-Outp-`) |
| `cost_match_confidence` | `full_match` / `no_resource_match` / `no_meter_match` |

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Local files

```bash
# Single chunk file
python meter_mapping.py \
    --summary  chargeBack-chunk-04-03h-05h.csv \
    --cost     "part_0_0001 (3).csv" \
    --output   meter_mapping_output.csv

# Multiple chunk files (all concatenated before mapping)
python meter_mapping.py \
    --summary  chargeBack-chunk-04-03h-05h.csv \
               chargeBack-chunk-12-19h-21h.csv \
    --cost     "part_0_0001 (3).csv" \
    --output   meter_mapping_output.csv
```

### Azure Blob Storage

```bash
python meter_mapping.py \
    --token-storage-url  https://myaccount.blob.core.windows.net \
    --token-container    chargeback-chunks \
    --token-prefix       chargeBack-chunk- \
    --cost-storage-url   https://myaccount.blob.core.windows.net \
    --cost-container     cost-exports \
    --output             meter_mapping_output.csv
```

Authentication uses a SAS token embedded in the URL (if the URL contains `?sv=`)
or `DefaultAzureCredential` otherwise (supports `az login`, managed identity, etc.).

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--summary PATH [PATH …]` | | One or more local token-summary chunk CSVs. All are concatenated before mapping. |
| `--cost PATH` | | Local Azure Cost + Usage CSV. |
| `--token-storage-url URL` | | Storage account URL for token-usage chunk blobs. |
| `--token-container NAME` | | Container holding the chunk CSV blobs. |
| `--token-prefix PREFIX` | `""` | Optional blob prefix to filter chunk files (e.g. `chargeBack-chunk-`). |
| `--cost-storage-url URL` | | Storage account URL for the cost + usage CSV. |
| `--cost-container NAME` | | Container holding the cost CSV blob. |
| `--cost-blob NAME` | *(auto)* | Exact blob name of the cost CSV. Omit to auto-detect the first CSV in the container. |
| `--output PATH` | `meter_mapping_output.csv` | Output CSV path. |
| `--fuzzy-threshold SCORE` | `60.0` | Minimum rapidfuzz score (0–100) for a meter-name match. Lower = more permissive. |

---

## Interpreting the output

| `cost_match_confidence` | Meaning |
|------------------------|---------|
| `full_match` | ResourceId and ModelName both matched. Meter names are reliable. |
| `no_resource_match` | The Cognitive Services account from the token summary does not appear in the cost report. Possible causes: different billing period, different subscription export, or `ResourceId` missing from the summary (feature branch not deployed). |
| `no_meter_match` | ResourceId matched but no meter name fuzzy-matched the model name. Try lowering `--fuzzy-threshold` or verify the `ModelName` value. |

---

## Repository structure

```
MeterMapping/
├── meter_mapping.py          # Main Python script
├── requirements.txt          # Python dependencies
├── sample_summary.csv        # Minimal example chargeback summary CSV
├── .gitignore
└── README.md
```

---

## Dependency on PTUChargeBackWorkflow

This tool consumes the CSV output of the `CreateChargeBackReport` Logic App
workflow.  Specifically it requires the **`feature/AddResourceID`** branch
because that branch adds the `ResourceId` column to the KQL `project` clause,
enabling a reliable, exact join to the cost report.

Without `ResourceId`, the only available join keys are `AccountName` (which is
not globally unique) or endpoint URL matching—both more error-prone.

Related repository: https://github.com/sbray779/PTUChargeBackWorkflow/tree/feature/AddResourceID
