# TP-EAR

## Requirements:

TP-EAR has been developed using pandas=1.5.2, numpy=1.23.1, scikit-learn=1.5.1 and matplotlib=3.7.1.
Testing was performed in Python 3.9.7.

The SSA code was largely lifted from https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
and edits to this were generally minimal to fit our requirements.

## Method:

1. Obtain 6-axis IMU data and save in CSV format.
2. Run apply_algorithms() to determine gait events using our algorithms.
3. Run compare_gait_events_to_gt() to find the difference between each algorithm's events and the ground truth.
4. Run calculate_raw_event_metrics() to calculate the sensitivity and laterality scores.
5. Run generate_tsps() to create TSPs from event data.
6. Run calculate_tsp_metrics() to analyse the time differences for the generated TSPs.

## Further Enquiries

This repository contains the code used to generate the results reported in the paper. Those wishing to use
TP-EAR for their own projects should preprocess their own data and use the code found in processing/TPEAR.py.
Contact tf375@cam.ac.uk with any further questions or suggestions.