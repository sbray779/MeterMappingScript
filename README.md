# MeterMapping

Marries the PTU ChargeBack token-usage summary (produced by
[sbray779/PTUChargeBackWorkflow](https://github.com/sbray779/PTUChargeBackWorkflow)
branch `feature/AddResourceID`) to meter lines in an Azure Cost + Usage export.

---

## Why this exists

The `CreateChargeBackReport` Logic App produces a daily CSV that summarises
**how many tokens each APIM product / workspace consumed per Azure OpenAI
deployment**.  The report includes the ARM `ResourceId` of the Cognitive
Services account (`feature/AddResourceID` branch).

A separate **Azure Cost + Usage** export (`part_0_0001 (3).csv`) from the
billing portal contains the actual **dollar cost** broken out by meter—with
one row per *model × input/output direction × day*.

This script joins the two files so you can:

* See the **cost per department / product** (chargeback).
* Allocate **input-token cost** and **output-token cost** separately.
* Audit which deployments drove the most spend.

---

## How the matching works

| Step | Key | Description |
|------|-----|-------------|
| 1 | **ResourceId** (exact, case-insensitive) | Links a token-summary row to all cost rows for the same Cognitive Services account. This is the primary join key. |
| 2 | **Deployment name → Meter name** (fuzzy) | Within the matched cost rows, `rapidfuzz.fuzz.partial_ratio` maps the deployment name (e.g. `gpt-4o-mini`) to the model component of the meter name (e.g. `gpt-4o-mini-0718-Outp-regnl Tokens`).  Version suffixes like `-0718` are tolerated. |
| 3 | **Input / Output split** | Meter names containing `-Inp-` → `PromptTokens`; `-Outp-` → `CompletionTokens`; otherwise → `TotalTokens`. |

### Meter name anatomy

```
gpt-4o-mini-0718  -  Inp  -  regnl  Tokens
│                    │        │
│                    │        └─ scope: regnl (regional) or glbl (global)
│                    └────────── direction: Inp (input) or Outp (output)
└────────────────────────────── model + version key
```

For embeddings and PTU batch models the direction component may be absent; those
rows are mapped to `TotalTokens`.

---

## Inputs

### 1. Token-summary CSV  (`--summary`)

Produced by the `CreateChargeBackReport` Logic App workflow
(**`feature/AddResourceID` branch required** for the `ResourceId` column).

Expected columns (order-independent):

| Column | Description |
|--------|-------------|
| `ResourceId` | ARM resource ID of the Cognitive Services account |
| `DeploymentName` | Azure OpenAI deployment name |
| `ModelName` | Model name from `CognitiveServicesInventory_CL` |
| `AccountName` | Cognitive Services account name |
| `SubscriptionId` | Azure subscription |
| `PromptTokens` | Total prompt/input tokens |
| `CompletionTokens` | Total completion/output tokens |
| `TotalTokens` | Sum of prompt + completion tokens |
| `Calls` | Number of API calls |
| `ProductId` | APIM product ID |
| `Luma` | 6-digit org code |
| `Workspace` | Workspace name |
| `SkuName` | SKU name |
| `SkuCapacity` | Provisioned capacity (PTU / TPM) |
| `BackendId` | APIM backend ID |
| `Endpoint` | Azure OpenAI endpoint URL |
| `FirstSeen` | Earliest request timestamp |
| `LastSeen` | Latest request timestamp |
| `Regions` | Semicolon-delimited Azure regions |
| `CallerIpAddresses` | Semicolon-delimited caller IPs |

A minimal sample is provided in [`sample_summary.csv`](sample_summary.csv).

### 2. Azure Cost + Usage export  (`--cost`)

Download from **Azure Portal → Cost Management → Exports** (or from the
billing portal).  The file used during development is
`part_0_0001 (3).csv`.

Relevant columns used by this script:

| Column | Description |
|--------|-------------|
| `ResourceId` | ARM resource ID |
| `meterName` | E.g. `gpt-4o-mini-0718-Outp-regnl Tokens` |
| `meterCategory` | E.g. `Foundry Models` |
| `meterSubCategory` | E.g. `Azure OpenAI` |
| `consumedService` | E.g. `Microsoft.CognitiveServices` |
| `costInUsd` | Cost in USD |
| `costInBillingCurrency` | Cost in billing currency |
| `quantity` | Metered quantity |
| `unitOfMeasure` | E.g. `1K` (tokens) |
| `date` | Billing date |
| `billingPeriodStartDate` | Billing period start |
| `billingPeriodEndDate` | Billing period end |

---

## Output columns

All original token-summary columns are preserved, then the following are appended:

| Column | Description |
|--------|-------------|
| `matched_meter_name` | `meterName` from the cost report |
| `meter_type` | `input` / `output` / `total` / `unknown` |
| `token_count_for_meter` | Tokens attributed to this meter |
| `cost_usd` | `costInUsd` from the cost report |
| `cost_billing_currency` | `costInBillingCurrency` from the cost report |
| `cost_quantity` | `quantity` from the cost report |
| `cost_unit_of_measure` | `unitOfMeasure` from the cost report |
| `cost_date` | `date` from the cost report |
| `billing_period_start` | Billing period start date |
| `billing_period_end` | Billing period end date |
| `cost_resource_id` | `ResourceId` from the cost report (verification) |
| `deployment_match_score` | rapidfuzz score 0–100 |
| `cost_match_confidence` | `full_match` / `resource_match_only` / `no_resource_match` / `no_meter_match` |

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

```bash
python meter_mapping.py \
    --summary  path/to/chargeback_summary.csv \
    --cost     "path/to/part_0_0001 (3).csv" \
    --output   meter_mapping_output.csv
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--summary PATH` | *(required)* | KQL chargeback summary CSV |
| `--cost PATH` | *(required)* | Azure Cost + Usage CSV |
| `--output PATH` | `meter_mapping_output.csv` | Output CSV |
| `--fuzzy-threshold SCORE` | `60.0` | Minimum rapidfuzz score (0–100) for a meter-name match.  Lower = more permissive. |
| `--all-cost-rows` | off | Emit all cost rows for a matched ResourceId even when the deployment name cannot be fuzzy-matched to a meter.  Good for gap analysis. |

### Quick test with included sample

```bash
python meter_mapping.py \
    --summary  sample_summary.csv \
    --cost     "part_0_0001 (3).csv" \
    --output   test_output.csv
```

---

## Interpreting the output

| `cost_match_confidence` | Meaning |
|------------------------|---------|
| `full_match` | Both ResourceId and deployment name matched. Cost allocation is reliable. |
| `no_resource_match` | The Cognitive Services account from the token summary does not appear in the cost report.  Possible causes: different billing period, different subscription export, or the ResourceId is missing from the summary (feature branch not deployed). |
| `no_meter_match` | ResourceId matched but no meter name fuzzy-matched the deployment name.  Try lowering `--fuzzy-threshold` or check for typos in the deployment name. |
| `resource_match_only` | Only emitted when `--all-cost-rows` is set.  ResourceId matched but meter name was not matched to a specific deployment. |

---

## Repository structure

```
MeterMapping/
├── meter_mapping.py          # Main Python script
├── requirements.txt          # Python dependencies
├── sample_summary.csv        # Minimal example chargeback summary CSV
├── part_0_0001 (3).csv       # Azure Cost + Usage export (source data, not committed)
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
