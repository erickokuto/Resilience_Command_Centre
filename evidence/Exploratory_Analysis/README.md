# Exploratory_Analysis | Data quality and exploratory data analysis
Synthetic German manufacturing capstone. Source: original Supply_Chain_CSV files. No real ERP data. No source records modified. 

## Results
- 10,000 purchase order lines, 46 columns, 120 suppliers, six plants and eight materials; 7,175 route-day signals.
- 31 structural/consistency checks passed (see quality_checks.csv for exact list). All 3,000 missing cells are intentional outcomes on 1,000 scoring rows. Import types differ between labelled and scoring files; stable schema required.
- Training only: 6,000 orders, mean 21.00 days, median 15, SD 16.38, range 2–91, IQR 10–26; 3,275 late (54.58%). Mean actual minus planned 1.89 days. Sea average 48.00 days, 75.02% late; road 12.66 days, 48.65% late. Strike-flagged orders 80.00% late vs 53.41% otherwise. These are associations within synthetic data, not causal findings.
- 553 training lead times exceed the global IQR upper fence of 50 days. Retain them: sea routes are structurally longer. Use route-aware diagnostics rather than deleting extreme values blindly.
- 230 training records have no prior completed supplier receipts; retain the provided smoothed priors and history count. Reconstructed historical receipt counts match all records; smoothing formulas themselves not independently verified.
- 54.28% of training order snapshots have stock coverage shorter than planned lead time; 60.10% shorter than actual lead time (retrospective proxy, not measured stockouts). Inventory is not a ledger.
- Planned lead time correlates with actual lead time (0.922); previous supplier mean (0.861); congestion (0.857). Congestion is structurally zero on non-sea routes, so examine within-mode relationships and avoid causal claims.

## Further analytical work
Aggregate training-only supplier histories; standardise numeric features before PCA/K-means; assess cluster stability, avoid using outcome-derived contemporaneous information for prediction. Fit preprocessing and models only on training data; use validation for tuning, test once after locking, and scoring for deployment. The previously published split-level outcome summaries mean the test set is not completely blind; retain it for final evaluation without tuning against it, and seek a fresh holdout if strict independence is required.

## Run
Python (executed here): `python Exploratory_Analysis.py data output` with pandas, numpy, matplotlib. R companion (not executed here because R is unavailable): `Rscript Exploratory_Analysis.R data output_R` with tidyverse and lubridate. The `data` directory should contain the original CSV files. The R companion reproduces core checks and charts, not every Python audit. Outputs in tables/ and charts/ are from the executed Python script. 
