# Low-travel magnetic magnitude statistics

`mag_norm` is the exact scalar consumed by `GetMagTravelRefPoint` (`mag/norm/corr/lpf`). `corrected_mag` is reconstructed as the norm of full-rate filtered XYZ after subtracting the interpolated nuisance body and world fields, matching `MagNuisanceFullRateCorrection`.

All ranges use strict ground-truth bounds and all finite samples; standard deviations use NumPy's population convention (`ddof=0`).

## Cohort macro averages

Each cell is the mean of the per-log mean and the mean of the per-log within-range standard deviation.

| cohort | logs | mag_norm 0-5 mean / std | corrected 0-5 mean / std | mag_norm 0-25 mean / std | corrected 0-25 mean / std |
| --- | ---: | ---: | ---: | ---: | ---: |
| harry | 14 | 982.0 / 86.2 | 1060.0 / 36.4 | 1094.3 / 98.0 | 1138.3 / 59.5 |
| jamaal | 6 | 985.4 / 43.8 | 996.1 / 20.1 | 1035.4 / 70.2 | 1042.6 / 40.0 |
| stumpjumper-front-pod-v2 | 11 | 399.1 / 88.0 | 492.0 / 55.0 | 526.7 / 145.9 | 577.5 / 96.7 |
| stumpjumper-front-pod-v1 | 7 | 673.5 / 121.8 | 654.7 / 67.9 | 729.4 / 145.7 | 715.0 / 88.0 |

## Per-log table

| cohort | log | N 0-5 | mag_norm 0-5 mean | mag_norm 0-5 std | corrected 0-5 mean | corrected 0-5 std | N 0-25 | mag_norm 0-25 mean | mag_norm 0-25 std | corrected 0-25 mean | corrected 0-25 std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harry | log-0113 | 116 | 929.3 | 124.5 | 1015.8 | 26.6 | 3729 | 1041.5 | 103.7 | 1123.8 | 47.2 |
| harry | log-0115 | 76 | 914.4 | 152.3 | 949.7 | 105.9 | 1407 | 1100.4 | 152.8 | 1108.9 | 142.0 |
| harry | log-0116 | 568 | 1059.3 | 52.0 | 1146.6 | 37.2 | 8909 | 1122.0 | 65.1 | 1203.0 | 39.1 |
| harry | log-0117 | 219 | 632.2 | 229.0 | 1003.1 | 42.5 | 1855 | 998.0 | 201.4 | 1130.4 | 75.0 |
| harry | log-0118 | 1334 | 954.3 | 36.2 | 1158.0 | 35.4 | 4867 | 1102.4 | 106.0 | 1180.2 | 53.8 |
| harry | log-0119 | 3940 | 1102.6 | 55.3 | 1094.7 | 17.0 | 6937 | 1113.4 | 60.7 | 1111.0 | 36.4 |
| harry | log-0120 | 82 | 1044.3 | 60.8 | 1066.0 | 14.6 | 1193 | 1095.8 | 65.2 | 1111.4 | 42.9 |
| harry | log-0122 | 177 | 1006.6 | 104.0 | 1041.1 | 46.1 | 9348 | 1128.4 | 83.1 | 1135.7 | 45.6 |
| harry | log-0125 | 109 | 954.7 | 138.7 | 1002.3 | 45.9 | 3882 | 1090.9 | 111.5 | 1107.8 | 57.7 |
| harry | log-0126 | 471 | 1049.3 | 15.0 | 1059.7 | 8.4 | 2991 | 1110.1 | 46.4 | 1130.7 | 42.9 |
| harry | log-0127 | 4935 | 985.8 | 64.0 | 1019.4 | 37.5 | 12831 | 1102.9 | 112.7 | 1115.7 | 88.0 |
| harry | log-0129 | 1736 | 1003.4 | 34.7 | 1104.1 | 24.8 | 9380 | 1065.3 | 98.1 | 1133.7 | 52.6 |
| harry | log-0132 | 225 | 1052.7 | 64.2 | 1142.1 | 41.2 | 5107 | 1117.0 | 86.2 | 1217.1 | 61.6 |
| harry | log-0133 | 26 | 1059.3 | 75.2 | 1037.8 | 25.8 | 1807 | 1132.7 | 78.6 | 1127.1 | 47.7 |
| jamaal | log-0078-valid | 158 | 991.2 | 46.8 | 1034.1 | 18.1 | 7811 | 1038.9 | 61.3 | 1107.2 | 25.1 |
| jamaal | log-0079 | 2281 | 989.0 | 15.6 | 1018.4 | 10.1 | 13560 | 1092.2 | 62.3 | 1087.6 | 45.6 |
| jamaal | log-0080-valid | 5720 | 1002.5 | 43.3 | 973.7 | 18.1 | 14580 | 1023.1 | 69.5 | 998.4 | 43.0 |
| jamaal | log-0081 | 1332 | 980.8 | 72.1 | 987.7 | 26.6 | 27569 | 1054.1 | 75.8 | 1054.4 | 40.0 |
| jamaal | log-0098 | 11341 | 954.1 | 19.6 | 979.2 | 12.1 | 16311 | 985.3 | 62.7 | 994.4 | 36.0 |
| jamaal | log-0099 | 628 | 994.8 | 65.4 | 983.5 | 35.7 | 4926 | 1018.7 | 89.5 | 1013.4 | 50.0 |
| stumpjumper-front-pod-v2 | log-0046 | 7745 | 400.9 | 40.8 | 725.1 | 59.0 | 20785 | 538.8 | 127.6 | 737.7 | 66.9 |
| stumpjumper-front-pod-v2 | log-0048 | 13732 | 379.1 | 99.1 | 547.8 | 35.6 | 71169 | 493.5 | 120.8 | 585.1 | 51.3 |
| stumpjumper-front-pod-v2 | log-0049 | 1320 | 509.6 | 116.2 | 450.2 | 23.9 | 21362 | 549.3 | 102.5 | 549.0 | 75.8 |
| stumpjumper-front-pod-v2 | log-0054 | 5425 | 392.5 | 65.2 | 541.0 | 52.6 | 40110 | 570.5 | 155.9 | 648.8 | 96.3 |
| stumpjumper-front-pod-v2 | log-0055 | 8721 | 329.7 | 94.0 | 337.0 | 65.5 | 23716 | 504.0 | 182.1 | 492.2 | 157.2 |
| stumpjumper-front-pod-v2 | log-0056 | 2728 | 386.3 | 66.0 | 516.3 | 66.6 | 35764 | 598.7 | 169.9 | 626.1 | 83.9 |
| stumpjumper-front-pod-v2 | log-0058 | 2946 | 366.5 | 86.7 | 458.5 | 65.9 | 8561 | 509.0 | 151.2 | 550.8 | 110.7 |
| stumpjumper-front-pod-v2 | log-0063 | 11754 | 323.6 | 91.2 | 371.4 | 44.4 | 29269 | 454.0 | 149.2 | 456.4 | 92.4 |
| stumpjumper-front-pod-v2 | log-0071_183 | 4698 | 403.1 | 69.1 | 502.6 | 38.3 | 25632 | 499.2 | 117.0 | 571.0 | 92.8 |
| stumpjumper-front-pod-v2 | log-0072_184 | 17450 | 563.3 | 188.6 | 528.6 | 92.6 | 26339 | 578.3 | 173.0 | 556.2 | 108.1 |
| stumpjumper-front-pod-v2 | log-0073_185 | 795 | 335.6 | 51.1 | 433.8 | 60.6 | 4220 | 498.6 | 156.2 | 579.7 | 128.5 |
| stumpjumper-front-pod-v1 | log103 | 7687 | 649.2 | 205.6 | 657.1 | 53.1 | 23154 | 677.9 | 189.6 | 683.7 | 61.1 |
| stumpjumper-front-pod-v1 | log104 | 2851 | 897.8 | 91.5 | 696.3 | 85.9 | 15464 | 827.6 | 120.0 | 735.8 | 70.0 |
| stumpjumper-front-pod-v1 | log106 | 1813 | 602.1 | 83.2 | 619.6 | 42.8 | 4010 | 644.0 | 98.5 | 666.2 | 77.0 |
| stumpjumper-front-pod-v1 | log107 | 1126 | 595.1 | 95.2 | 549.4 | 68.9 | 3245 | 638.0 | 138.6 | 631.0 | 117.3 |
| stumpjumper-front-pod-v1 | log109 | 680 | 531.7 | 79.1 | 641.5 | 35.6 | 3123 | 685.7 | 150.8 | 746.6 | 96.4 |
| stumpjumper-front-pod-v1 | log110 | 774 | 733.3 | 172.5 | 663.8 | 138.7 | 3462 | 813.9 | 168.0 | 735.4 | 116.8 |
| stumpjumper-front-pod-v1 | log112 | 11426 | 705.6 | 125.7 | 755.0 | 50.6 | 29506 | 818.6 | 154.7 | 806.5 | 77.4 |
