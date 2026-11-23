# Resilience_Command_Centre — Supply Chain Resilience Command Centre

Standalone R Shiny application with the same frozen Python analytical models as Analytics_Workflow. Designed for procurement planners, using synthetic German manufacturing records. No SAP connection or live operational action. The bundled demonstration contains 1,000 scored orders and 12,000 order-policy-scenario rows, generated and verified with the backend.

## Windows / RStudio quick start
Extract the entire ZIP into a directory. Install R, RStudio and Python 3.13 (64-bit). From a PowerShell terminal in the extracted Resilience_Command_Centre directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the RStudio console (adjust the path):

```r
setwd('D:/Projects/Resilience_Command_Centre')
install.packages(c('shiny','DT','ggplot2'))
Sys.setenv(SUPPLY_PYTHON=normalizePath('.venv/Scripts/python.exe',winslash='/'))
shiny::runApp('.')
```

On Linux/macOS use `python3.13 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt` and set SUPPLY_PYTHON to the absolute `.venv/bin/python` path. The bundled demonstration is read locally; new uploads need the Python environment. Use only bundled trusted joblib models.

## Operations
- Start with the demonstration. Filter by plant, transport and criticality; set thresholds (review must be below high risk), probability version and nominal interval level.
- Upload a compatible CSV (1–1,000 rows) and click Validate and analyse upload. The same fitted models are applied, and four procurement policies are evaluated under three scenarios. Failed uploads clear current results; restore demonstration or upload a corrected dataset. No automatic retraining.
- Review alerts, supplier exposure, inventory coverage and proposed actions. Scenario results update for the filtered batch. Stress tests hold base-case policies fixed, not reoptimised under stress; all new-batch outcomes are simulated, even for labelled rows.
- Human review records are session-only, cleared on new batch or restore. Export the review log before switching batches or closing the app. Approve for planning does not execute any purchase or SAP operation. Export tables using the buttons.
- Data contract: 40 required fields from the 46-field schema. Outcome columns may be absent on open orders; three audit columns optional. Dates ISO YYYY-MM-DD, same units and category definitions. Historical summaries must genuinely predate the order date; schema checks cannot prove provenance. See data/Data_Dictionary.csv. This is an order-creation model, not live shipment tracking.

## Direct backend
```powershell
.\.venv\Scripts\python.exe backend/command.py data/Scoring.csv results_demo
```
Outputs: results.csv, scenarios.csv, quality.csv and intermediate files. Scenario output includes every order, four policies, three stress levels; results contain base-case recommendations. Prescriptive_Analytics assumptions include 14 days max immediate emergency buffer, 20% emergency acquisition premium, independent orders, alternative supplier quotes and simulated demand. Relevant costs exclude shared base purchasing costs. Costs and savings are not observed financial outcomes. Scenario costs are expected costs, and stockout probabilities are average simulated order-level exposures, not probabilities of an enterprise-wide stockout.

## Verification / outstanding work
Python: replayed all 1,000 reference orders and matched actions, predictions, uncertainty bounds, costs, savings and stockout exposure. Created 12,000 unique scenario rows. A 12-row upload produced 144 scenario rows; missing supplier column was rejected. R is not installed in the authoring environment; app.R passed a delimiter balance check only, not an R parser, runtime or browser acceptance test. No online deployment or live SAP BTP/IBP integration. Use approved data and secure authenticated deployment before actual procurement use. Shared inventory, capacity, budget constraints and human review persistence would be necessary extensions. Analytics_Workflow remains a separate technical workflow application.

## Using the dashboard
Use data/Scoring.csv for open-order forecasts. Use Test.csv for observed outcomes in the Analytics Workflow. Do not upload support tables or prediction exports. The user guide in the reviewer package explains all tabs.
