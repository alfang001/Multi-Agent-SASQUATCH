import numpy as np

# Original paper stats
# In order of radius 8.5, 10.0, 12.0, 14.5, 20.0
s1_time_no_break = np.array([54.269, 86.860, 111.623, 162.181, 214.447])
s1_rmse_no_break = np.array([0.360, 0.089, 0.163, 0.198, 0.190])
s1_ghd_no_break = np.array([1.492, 1.089, 2.345, 2.749, 2.600])
s1_time_break = np.array([54.071, 9.986, 11.947, 9.847, 12.504])
s1_rmse_break = np.array([0.365, 0.177, 0.263, 0.311, 0.481])
s1_ghd_break = np.array([1.626, 1.475, 2.048, 2.351, 3.711])
s1_average_time_saved = 1.0 - np.mean(s1_time_break / s1_time_no_break)

# Compute stats for Scenario 1
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print(f"Scenario 1 max/min, range, time no break: {np.max(s1_time_no_break)}, {np.min(s1_time_no_break)}, {np.max(s1_time_no_break) - np.min(s1_time_no_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_time_no_break)}, {np.mean(s1_time_no_break)}")
# print(f"Scenario 1 max/min, range, rmse no break: {np.max(s1_rmse_no_break)}, {np.min(s1_rmse_no_break)}, {np.max(s1_rmse_no_break) - np.min(s1_rmse_no_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_rmse_no_break)}, {np.mean(s1_rmse_no_break)}")
# print(f"Scenario 1 max/min, range, ghd no break: {np.max(s1_ghd_no_break)}, {np.min(s1_ghd_no_break)}, {np.max(s1_ghd_no_break) - np.min(s1_ghd_no_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_ghd_no_break)}, {np.mean(s1_ghd_no_break)}")
# print(f"Scenario 1 max/min, range, time break: {np.max(s1_time_break)}, {np.min(s1_time_break)}, {np.max(s1_time_break) - np.min(s1_time_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_time_break)}, {np.mean(s1_time_break)}")
# print(f"Scenario 1 max/min, range, rmse break: {np.max(s1_rmse_break)}, {np.min(s1_rmse_break)}, {np.max(s1_rmse_break) - np.min(s1_rmse_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_rmse_break)}, {np.mean(s1_rmse_break)}")
# print(f"Scenario 1 max/min, range, ghd break: {np.max(s1_ghd_break)}, {np.min(s1_ghd_break)}, {np.max(s1_ghd_break) - np.min(s1_ghd_break)}")
# print(f"Scenario 1 median and mean: {np.median(s1_ghd_break)}, {np.mean(s1_ghd_break)}")
print(f"Scenario 1 average time saved: {s1_average_time_saved}")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")

# Scenario 2 stats
# In order of radius 8.5, 10.0, 12.0, 14.5, 20.0
s2_time_no_break = np.array([53.416, 88.204, 110.818, 171.711, 217.541])
s2_rmse_no_break = np.array([5.277, 6.523, 6.523, 9.164, 6.996])
s2_ghd_no_break = np.array([19.235, 19.989, 19.989, 25.051, 19.642])
s2_time_break = np.array([54.644, 66.096, 79.932, 117.165, 142.546])
s2_rmse_break = np.array([4.580, 6.365, 4.626, 9.277, 6.938])
s2_ghd_break = np.array([15.193, 20.077, 15.903, 25.060, 19.046])
s2_average_time_saved = 1.0 - np.mean(s2_time_break / s2_time_no_break)

# Compute stats for Scenario 2
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print(f"Scenario 2 max/min, range, time no break: {np.max(s2_time_no_break)}, {np.min(s2_time_no_break)}, {np.max(s2_time_no_break) - np.min(s2_time_no_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_time_no_break)}, {np.mean(s2_time_no_break)}")
# print(f"Scenario 2 max/min, range, rmse no break: {np.max(s2_rmse_no_break)}, {np.min(s2_rmse_no_break)}, {np.max(s2_rmse_no_break) - np.min(s2_rmse_no_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_rmse_no_break)}, {np.mean(s2_rmse_no_break)}")
# print(f"Scenario 2 max/min, range, ghd no break: {np.max(s2_ghd_no_break)}, {np.min(s2_ghd_no_break)}, {np.max(s2_ghd_no_break) - np.min(s2_ghd_no_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_ghd_no_break)}, {np.mean(s2_ghd_no_break)}")
# print(f"Scenario 2 max/min, range, time break: {np.max(s2_time_break)}, {np.min(s2_time_break)}, {np.max(s2_time_break) - np.min(s2_time_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_time_break)}, {np.mean(s2_time_break)}")
# print(f"Scenario 2 max/min, range, rmse break: {np.max(s2_rmse_break)}, {np.min(s2_rmse_break)}, {np.max(s2_rmse_break) - np.min(s2_rmse_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_rmse_break)}, {np.mean(s2_rmse_break)}")
# print(f"Scenario 2 max/min, range, ghd break: {np.max(s2_ghd_break)}, {np.min(s2_ghd_break)}, {np.max(s2_ghd_break) - np.min(s2_ghd_break)}")
# print(f"Scenario 2 median and mean: {np.median(s2_ghd_break)}, {np.mean(s2_ghd_break)}")
print(f"Scenario 2 average time saved: {s2_average_time_saved}")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")

# Scenario 3 stats
# In order of radius 8.5, 10.0, 12.0, 14.5, 20.0
s3_time_no_break = np.array([34.141, 55.560, 69.582, 102.360, 137.020])
s3_rmse_no_break = np.array([0.946, 0.136, 0.305, 0.471, 1.447])
s3_ghd_no_break = np.array([7.904, 0.863, 2.888, 2.743, 4.713])
s3_time_break = np.array([35.248, 11.108, 11.252, 22.523, 34.100])
s3_rmse_break = np.array([0.505, 0.231, 0.322, 0.659, 1.578])
s3_ghd_break = np.array([2.408, 0.932, 2.131, 2.704, 4.964])
s3_average_time_saved = 1.0 - np.mean(s3_time_break / s3_time_no_break)

# Compute stats for Scenario 3
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print(f"Scenario 3 max/min, range, time no break: {np.max(s3_time_no_break)}, {np.min(s3_time_no_break)}, {np.max(s3_time_no_break) - np.min(s3_time_no_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_time_no_break)}, {np.mean(s3_time_no_break)}")
# print(f"Scenario 3 max/min, range, rmse no break: {np.max(s3_rmse_no_break)}, {np.min(s3_rmse_no_break)}, {np.max(s3_rmse_no_break) - np.min(s3_rmse_no_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_rmse_no_break)}, {np.mean(s3_rmse_no_break)}")
# print(f"Scenario 3 max/min, range, ghd no break: {np.max(s3_ghd_no_break)}, {np.min(s3_ghd_no_break)}, {np.max(s3_ghd_no_break) - np.min(s3_ghd_no_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_ghd_no_break)}, {np.mean(s3_ghd_no_break)}")
# print(f"Scenario 3 max/min, range, time break: {np.max(s3_time_break)}, {np.min(s3_time_break)}, {np.max(s3_time_break) - np.min(s3_time_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_time_break)}, {np.mean(s3_time_break)}")
# print(f"Scenario 3 max/min, range, rmse break: {np.max(s3_rmse_break)}, {np.min(s3_rmse_break)}, {np.max(s3_rmse_break) - np.min(s3_rmse_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_rmse_break)}, {np.mean(s3_rmse_break)}")
# print(f"Scenario 3 max/min, range, ghd break: {np.max(s3_ghd_break)}, {np.min(s3_ghd_break)}, {np.max(s3_ghd_break) - np.min(s3_ghd_break)}")
# print(f"Scenario 3 median and mean: {np.median(s3_ghd_break)}, {np.mean(s3_ghd_break)}")
print(f"Scenario 3 average time saved: {s3_average_time_saved}")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")