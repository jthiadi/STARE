# Mistake detection on the IndustReal test set

32 videos, 2.50 h. Ground truth from the official PSR labels; counts are shown so small rows are not over-read.

## oracle: GT actions + GT state

| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | Mean lead (s) | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| skipped | 10 | 10 | 100% (10/10) | 100% (10/10) | 0.1 | 2/10 | 43.6 | 0.0 |
| order | 4 | 4 | 100% (4/4) | 100% (4/4) | 0.7 | 4/4 | 39.2 | 0.0 |
| install | 18 | 20 | 90% (18/20) | 100% (18/18) | 0.6 | 4/18 | 32.0 | 0.8 |
| all | 32 | 34 | 94% (32/34) | 100% (32/32) | 0.4 | 10/32 | 37.2 | 0.8 |

Early warnings: 21.2 / h. "Cannot verify": 0.0 / h.

## baseline: official IndustReal PSR on ASD + the same mistake rules

| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | Mean lead (s) | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| skipped | 10 | 12 | 8% (1/12) | 10% (1/10) | 0.0 | 0/10 | - | 4.4 |
| order | 4 | 7 | 43% (3/7) | 75% (3/4) | 7.9 | 0/4 | - | 1.6 |
| install | 18 | 0 | - (0/0) | 0% (0/18) | - | 0/18 | - | 0.0 |
| all | 32 | 19 | 21% (4/19) | 12% (4/32) | 5.9 | 0/32 | - | 6.0 |

Early warnings: 0.0 / h. "Cannot verify": 0.0 / h.

## predicted actions + ASD

| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | Mean lead (s) | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| skipped | 10 | 7 | 43% (3/7) | 30% (3/10) | 1.6 | 2/10 | 43.3 | 1.6 |
| order | 4 | 3 | 100% (3/3) | 75% (3/4) | 5.9 | 4/4 | 35.6 | 0.0 |
| install | 18 | 13 | 15% (2/13) | 11% (2/18) | 1.8 | 4/18 | 35.6 | 4.4 |
| all | 32 | 23 | 35% (8/23) | 25% (8/32) | 3.3 | 10/32 | 37.2 | 6.0 |

Early warnings: 19.2 / h. "Cannot verify": 7.2 / h.

## pred_asdv14

| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | Mean lead (s) | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| skipped | 10 | 13 | 15% (2/13) | 20% (2/10) | 2.7 | 2/10 | 43.3 | 4.4 |
| order | 4 | 5 | 80% (4/5) | 100% (4/4) | 5.7 | 4/4 | 35.6 | 0.4 |
| install | 18 | 11 | 27% (3/11) | 17% (3/18) | 2.4 | 4/18 | 35.6 | 3.2 |
| all | 32 | 29 | 31% (9/29) | 28% (9/32) | 3.9 | 10/32 | 37.2 | 8.0 |

Early warnings: 20.4 / h. "Cannot verify": 4.8 / h.

## pred_v14

| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | Mean lead (s) | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| skipped | 10 | 5 | 40% (2/5) | 20% (2/10) | 2.4 | 0/10 | - | 1.2 |
| order | 4 | 5 | 80% (4/5) | 100% (4/4) | 6.5 | 0/4 | - | 0.4 |
| install | 18 | 7 | 43% (3/7) | 17% (3/18) | 7.7 | 0/18 | - | 1.6 |
| all | 32 | 17 | 53% (9/17) | 28% (9/32) | 5.9 | 0/32 | - | 3.2 |

Early warnings: 0.0 / h. "Cannot verify": 0.0 / h.

_Generated 2026-11-10 00:48 by `python -m assist.evaluation.eval`._