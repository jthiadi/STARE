# Procedure step recognition (IndustReal PSR protocol, test set, release ASD)

| Method | F1 | POS | Delay (s) | With errors: F1 | POS | Delay (s) | No errors: F1 |
|---|---|---|---|---|---|---|---|
| *IndustReal B3 (WACV'24), as published* | *0.883* | *0.797* | *22.4* | *0.816* | *0.731* | *20.4* | |
| *IndustReal, as reported in STORM-PSR (CVIU'25)* | *0.891* | *0.797* | *21.0* |  |  |  | |
| *STORM-PSR (CVIU'25), as published* | *0.901* | *0.812* | *15.5* |  |  |  | |
| Official IndustReal PSR on release ASD | 0.878 | 0.815 | 16.9 | 0.807 | 0.731 | 14.8 | 0.941 |
| Ours: engine, oracle (upper bound) | 0.964 | 0.954 | 0.7 | 0.923 | 0.902 | 0.7 | 1.000 |
| Ours: earlier live system (TCN actions, ASD) | 0.892 | 0.795 | 15.9 | 0.804 | 0.704 | 13.6 | 0.969 |
| pred_asdv14 | 0.920 | 0.848 | 7.0 | 0.854 | 0.765 | 8.0 | 0.978 |
