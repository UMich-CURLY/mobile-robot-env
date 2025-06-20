bash experiments/run_exp_sif.sh distance_only Utility_DistanceOnly_ExploitationOnly_ObservedOnly.yaml 3 5 "0,1,2,3,4" sample400 false --no_stair_climbing

bash experiments/run_exp_sif.sh new_prediction Utility_RegionCooccurrenceOnly_ExploitationOnly_ObservedOnly_Prediction.yaml 3 5 "0,1,2,3,4" sample400_fail true
