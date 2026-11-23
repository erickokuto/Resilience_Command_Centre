# Adaptive model refresh: a single uploaded dataset

## What changed

The dashboard retains its historical, frozen-model views and adds **Adaptive model refresh**. Use this tab when new completed procurement orders are available and you want to re-estimate lead-time and late-delivery predictions from a consolidated CSV.

1. Prepare **one** CSV in the same schema as `data/All_Orders.csv` (download it from the new tab). The file can contain both **completed orders** with `receipt_date`, `actual_lead_days` and `late_flag` filled in, and **open orders** with all three outcomes blank. Do not upload separate training, validation and testing CSV files.
2. Choose training and validation percentages (each 10–80%). Testing is calculated automatically as `100 – training – validation`; it must also be 10–80%.
3. Select a **chronological** split for an evolving operational dataset (recommended) or a **random** split for an exploratory demonstration. The percentages apply only to completed (labelled) orders. At least 60 completed orders are required, and each partition needs at least 10.
4. Click **Train / refresh**. The application performs original schema and order-time checks, fits new Ridge regression and logistic classification pipelines using training rows only, estimates 80%/90% symmetric residual-based intervals from validation rows, reports hold-out test metrics, and scores every row, including open orders.
5. Review the order-level adaptive risk queue, suggested review priorities, and download the full refreshed CSV.

## Important boundaries

- A new upload **replaces the candidate model in that user's current Shiny session**. It does not automatically merge with earlier uploads or save a permanent model. For incremental updates, **append new records to your original consolidated CSV and re-upload that full file**; use stable unique `po_id` + `item_id` identifiers.
- Existing historical evidence, unsupervised supplier segments, original procurement policies and simulated cost/savings scenarios are not retrained. The new `indicative_next_step` field is **human review advice**, not an optimised purchasing action or a realised saving.
- The dashboard's existing operational tabs still use frozen models. The new tab gives independent predictive comparisons without silently changing the historical research results.
- Do not upload unauthorised or confidential organisational data to a public hosting service. File contents are handled on the application server during processing. Model refresh and output CSVs are session-scoped and are not committed to GitHub.
- Chronological splitting orders labelled records by `order_date`. If several orders share the same date, the boundary can split a date across partitions; for rigorous forward-looking research, use distinct time windows and confirm dates, target availability and possible temporal leakage before interpreting performance.
- Predictions for rows in the training subset are in-sample and should not be used as a performance estimate. Use validation/test metrics to assess generalisation. Validation-derived symmetric intervals are approximations, not guaranteed coverage under drift.

## Running locally in RStudio (Windows)

Open the folder as an RStudio project, then install the three R packages if needed:

```r
install.packages(c('shiny','DT','ggplot2'))
```

Create an isolated Python environment in RStudio's **Terminal** (not R Console):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run in the RStudio Console:

```r
Sys.setenv(SUPPLY_PYTHON = normalizePath('.venv/Scripts/python.exe', winslash='/'))
shiny::runApp('.')
```

For Linux/macOS adapt the virtual-environment path. Install and launch in each dashboard folder separately.

## Confirming the backend without Shiny

From each dashboard folder:

```powershell
.\.venv\Scripts\python.exe backend/adaptive_training.py data/All_Orders.csv adaptive_test_outputs --train 70 --validation 15 --method chronological
```

Check these generated session/test files: `adaptive_metrics.csv`, `adaptive_predictions.csv` and `adaptive_summary.json`. Keep output directories out of public Git repositories. Do not push any confidential data.
