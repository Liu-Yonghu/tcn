import os
import sys
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict


def read_data(file_path, sensor):
    # read CSV file
    try:
        if sensor.lower() == 'radar':
            df = pd.read_csv(file_path, header=None, names=['angle', 'range', 'magnitude', 'timestamp'])
            df.columns = df.columns.str.strip()
        else:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
        print(f'the shape of this raw dataset: {df.shape}')
        print(df.head(5))
        return df
    except Exception as e:
        print(f'Failed to load data: {e}')


def features_extract(df, sensor):
    # extract the useful fields which we want
    data = []
    if sensor.lower() == 'tof':

        distance_cols = [col for col in df.columns if col.startswith('distance')]
        valid_cols = [col for col in df.columns if col.startswith(".is_valid_range")]
        target_cols = [col for col in df.columns if col.startswith("target_status")]
        timestamp_cols = [col for col in df.columns if col.startswith(".HostTimestamp")]

        # features = df[distance_cols].values.astype(np.float32)
        # signals = df[signal_cols].values.astype(np.float32)  # probably used later
        # valid = df[valid_cols].values.astype(np.float32)
        # timestamp = df[timestamp_cols].values.astype((np.float32))
        #
        # data = np.concatenate([timestamp, features, valid], axis=1)

        data = pd.concat([df[timestamp_cols], df[distance_cols], df[valid_cols], df[target_cols]],  axis=1)

    elif sensor.lower() == 'ultrasound':

        timestamp_cols = [col for col in df.columns if col.startswith("Timestamp")]
        X_cols = [col for col in df.columns if col.startswith('X')]
        Y_cols = [col for col in df.columns if col.startswith("Y")]

        data = pd.concat([df[timestamp_cols], df[X_cols], df[Y_cols]], axis=1)

    elif sensor.lower() == 'radar':

        angle_cols = 'angle'
        range_cols = 'range'
        max_magnitude_cols = 'magnitude'
        timestamp_cols = 'timestamp'


        angles = [col for col in df.columns if col.startswith(angle_cols)]
        ranges = [col for col in df.columns if col.startswith(range_cols)]
        max_magnitudes = [col for col in df.columns if col.startswith(max_magnitude_cols)]
        timestamp = [col for col in df.columns if col.startswith(timestamp_cols)]

        # angles = df[angle_cols].values.reshape(-1, 1)
        # ranges = df[range_cols].values.reshape(-1, 1)
        # max_magnitudes = df[max_magnitude_cols].values.astype(np.float32).reshape(-1, 1)
        # timestamp = df[timestamp_cols].values.astype(np.float32).reshape(-1, 1)
        #
        # data = np.concatenate([timestamp, angles, ranges, max_magnitudes], axis=1)
        # data = pd.concat([df[timestamp_cols], df[angle_cols], df[range_cols], df[max_magnitude_cols]], axis=1)
        data = pd.concat([df[timestamp], df[angles], df[ranges], df[max_magnitudes]], axis=1)

    else:
        print(" no this kind of sensor ")
    return data

def interpolation_data_by_samples(X):
    timestamp_col = X.columns[0]
    distance_cols = X.columns[1:65]
    valid_cols = X.columns[65:129]
    target_cols = X.columns[129:]

    interpolated_rows = []
    for _, row in X.iterrows():
        distances = row[distance_cols].astype(float)

        isvalid_mask = row[valid_cols].astype(bool).to_numpy().ravel()          # (64,)
        target_mask = row[target_cols].isin([4, 5, 6, 9, 10]).to_numpy().ravel()    # (64,)

        valids = np.logical_and(isvalid_mask, target_mask)

        mask = pd.Series(valids, index=distances.index)

        interpolated = distances.mask(~mask).interpolate(method='linear', limit_direction='both')

        new_row = pd.concat([
            pd.Series({timestamp_col: row[timestamp_col]}),
            interpolated,
        ])
        interpolated_rows.append(new_row)

    result = pd.DataFrame(interpolated_rows)
    return result

def interpolation_data_by_field(X):
    timestamp_col = X.columns[0]
    distance_cols = X.columns[1:65]
    valid_cols = X.columns[65:129]
    target_cols = X.columns[129:]

    distance_interp = X[distance_cols].copy()

    isvalid_mask = X[valid_cols].to_numpy().astype(bool)           # (n,64)
    target_mask = X[target_cols].isin([4, 5, 6, 9, 10]).to_numpy()     # (n,64)
    mask = np.logical_and(isvalid_mask, target_mask)
    # interpolate on every column
    for j, col in enumerate(distance_cols):
        col_mask = mask[:, j]
        s = X[col].mask(~col_mask).interpolate(method='linear', limit_direction='both')
        distance_interp[col] = s
    result = pd.concat([X[timestamp_col], distance_interp], axis=1)
    return result, mask

def interpolation_data_by_field_mmWave(X):
    pass


def save_data(data, labels, filename, method="mean"):
    timestamp = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    global_min = features.values.min()
    global_max = features.values.max()
    global_mean = features.values.mean()
    global_std = features.values.std()

    data_norm = (features - global_min) / (global_max - global_min)
    data_std = (features - global_mean) / global_std
    print(f"feature's mean: {global_mean}")
    print(f"feature's std: {global_std}")

    data_mixed = 1 / (1+np.exp(-data_std))

    path = 'clean_data/'
    os.makedirs(path, exist_ok=True)

    timestamp = timestamp.reset_index(drop=True)
    data_norm = data_norm.reset_index(drop=True)
    data_std = data_std.reset_index(drop=True)
    data_mixed = data_mixed.reset_index(drop=True)

    if method == "mean":
        labels_down = mean_labels(labels, len(data))
        labels_down = labels_down.reset_index(drop=True)
        labels_down.to_csv(os.path.join(path, f'label_down.csv'), index=False, header=True)

        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]

        data_mixed["X"] = labels_down["X"]
        data_mixed["Y"] = labels_down["Y"]
    elif method == "downsample":
        indices = np.linspace(0, len(labels) - 1, num=len(data), dtype=int)
        labels_down = labels.iloc[indices].reset_index(drop=True)

        # concat the result
        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]

        data_mixed["X"] = labels_down["X"]
        data_mixed["Y"] = labels_down["Y"]
    else:
        raise ValueError("Unsupported method. Choose from ['mean', 'downsample'].")

    data_norm = data_norm.round(6)
    data_std = data_std.round(6)
    data_mixed = data_mixed.round(6)

    data_norm = pd.concat([timestamp, data_norm], axis=1)
    data_std = pd.concat([timestamp, data_std], axis=1)
    data_mixed = pd.concat([timestamp, data_mixed], axis=1)

    data_norm.to_csv(os.path.join(path, f'norm_{filename}.csv'), index=False, header=True)
    data_std.to_csv(os.path.join(path, f'std_{filename}.csv'), index=False, header=True)
    data_mixed.to_csv(os.path.join(path, f'mixed_{filename}.csv'), index=False, header=True)
    print(f"Saved norm_{filename}, std_{filename} , mixed_{filename} to {path}")
    return pd.DataFrame(data_std), pd.DataFrame(data_norm), pd.DataFrame(data_mixed)



def save_data_mmwave(data, labels, filename, method="mean"):
    timestamp = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    features_min = features.min()
    features_max = features.max()
    features_mean = features.mean()
    features_std = features.std()

    data_norm = ((features - features_min) / (features_max - features_min)).round(6)
    data_std = ((features - features_mean) / features_std).round(6)
    print(f"feature's min: {features_min}")
    print(f"feature's max: {features_max}")
    print(f"feature's mean: {features_mean}")
    print(f"feature's std: {features_std}")

    data_mixed = (1 / (1+np.exp(-data_std))).round(6)

    path = 'clean_data/'
    os.makedirs(path, exist_ok=True)

    timestamp = timestamp.reset_index(drop=True)
    data_norm = data_norm.reset_index(drop=True)
    data_std = data_std.reset_index(drop=True)
    data_mixed = data_mixed.reset_index(drop=True)

    if method == "mean":
        labels_down = mean_labels(labels, len(data))
        labels_down = labels_down.reset_index(drop=True)
        labels_down.to_csv(os.path.join(path, f'labels_down.csv'), index=False, header=True)

        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]

        data_mixed["X"] = labels_down["X"]
        data_mixed["Y"] = labels_down["Y"]

    elif method == "downsample":
        indices = np.linspace(0, len(labels) - 1, num=len(data), dtype=int)
        labels_down = labels.iloc[indices].reset_index(drop=True)

        # concat the result
        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]

        data_mixed["X"] = labels_down["X"]
        data_mixed["Y"] = labels_down["Y"]
    else:
        raise ValueError("Unsupported method. Choose from ['mean', 'downsample'].")

    data_norm = data_norm.round(6)
    data_std = data_std.round(6)
    data_mixed = data_mixed.round(6)



    data_norm = pd.concat([timestamp, data_norm], axis=1)
    data_std = pd.concat([timestamp, data_std], axis=1)
    data_mixed = pd.concat([timestamp, data_mixed], axis=1)

    data_norm.to_csv(os.path.join(path, f'norm_{filename}.csv'), index=False, header=True, float_format="%.6f")
    data_std.to_csv(os.path.join(path, f'std_{filename}.csv'), index=False, header=True, float_format="%.6f")
    data_mixed.to_csv(os.path.join(path, f'mixed_{filename}.csv'), index=False, header=True, float_format="%.6f")
    print(f"Saved norm_{filename}, std_{filename} , mixed_{filename} to {path}")
    return pd.DataFrame(data_std), pd.DataFrame(data_norm), pd.DataFrame(data_mixed)


def mean_labels(labels, target_len):
    n = len(labels)
    indices = np.linspace(0, n, target_len + 1, dtype=int)
    averaged_labels = []

    for i in range(target_len):
        start = indices[i]
        end = indices[i + 1]
        chunk = labels.iloc[start:end]
        averaged = chunk.mean(numeric_only=True)
        averaged_labels.append(averaged)

    print(len(averaged_labels), target_len)
    return pd.DataFrame(averaged_labels).reset_index(drop=True)


def clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3, stuck_thresh=1e-3, stuck_len=16):
    df = df.copy()
    N = len(df)
    df["valid_mask"] = 1

    # ensure it is a number
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    print("===== Step 1: Boundary Check =====")
    out_range = (df["X"] < range_min) | (df["X"] > range_max) | (df["Y"] < range_min) | (df["Y"] > range_max)
    if out_range.any():
        print("Detected boundary anomalies: ")
        print(df.loc[out_range, ["Timestamp", "X", "Y"]])
    df.loc[out_range, "valid_mask"] = 0

    print("===== Step 2: Jump Detection =====")
    dx = df["X"].diff()
    dy = df["Y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    jump_mask = dist > jump_thresh
    if jump_mask.any():
        print("Detected jump anomalies: ")
        print(df.loc[jump_mask, ["Timestamp", "X", "Y"]])
    df.loc[jump_mask, "valid_mask"] = 0

    print("===== Step 3: Stagnation Detection =====")
    for i in range(N - stuck_len):
        segment_x = df["X"].iloc[i:i+stuck_len]
        segment_y = df["Y"].iloc[i:i+stuck_len]
        if segment_x.std() < stuck_thresh and segment_y.std() < stuck_thresh:
            print(f"Detected stagnation anomalies: index {i} ~ {i+stuck_len-1}")
            df.loc[i:i+stuck_len-1, "valid_mask"] = 0

    # 4. replace by NaN
    df.loc[df["valid_mask"] == 0, ["X", "Y"]] = np.nan
    df.loc[df["valid_mask"] == 0, "valid_mask"] = 1

    # 5. interpolation fixing
    df["X"] = df["X"].interpolate(method="linear", limit_direction="both")
    df["Y"] = df["Y"].interpolate(method="linear", limit_direction="both")
    return df


def fix_ultraSound_trajectory():
    df = read_data("./raw_data/exp1_10min_ultrasound.csv", "ultrasound")
    df1 = clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3)
    df1.to_csv("./raw_data/exp1_10min_ultrasound.csv", index=False)

    df = read_data("./raw_data/exp2_30min_ultrasound.csv", "ultrasound")
    df2 = clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3)
    df2.to_csv("./raw_data/exp2_30min_ultrasound.csv", index=False)

    df = read_data("./raw_data/exp3_1h_ultrasound.csv", "ultrasound")
    df3 = clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3)
    df3.to_csv("./raw_data/exp3_1h_ultrasound.csv", index=False)

    df = read_data("./raw_data/exp4_1h_ultrasound.csv", "ultrasound")
    df4 = clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3)
    df4.to_csv("./raw_data/exp4_1h_ultrasound.csv", index=False)

def analysis_tof(exp, data, std_data, norm_data):
    data_collection = {"raw": {}, "clean": {}, "std": {}, "norm": {}}

    timestamp_col = data.columns[0]
    distance_cols = data.columns[1:65]
    valid_cols = data.columns[65:129]
    target_cols = data.columns[129:]

    isvalid_mask = data[valid_cols].to_numpy().astype(bool)  # (n,64)
    target_mask = data[target_cols].isin([4, 5, 6, 9, 10]).to_numpy()  # (n,64)
    mask = np.logical_and(isvalid_mask, target_mask)

    D = data[distance_cols]
    data_collection["raw"]["mask"] = np.sum(mask)
    data_collection["raw"]["invalid_number"] = np.sum(~mask)
    data_collection["raw"]["total_number"] = D.size
    data_collection["raw"]["valid_ratio"] = float(np.sum(mask) / D.size)
    data_collection["raw"]["G_min"] = D.values.min()
    data_collection["raw"]["G_max"] = D.values.max()
    data_collection["raw"]["G_mean"] = D.values.mean()
    data_collection["raw"]["G_std"] = D.values.std()

    col_std = D.std(axis=0, skipna=True).to_numpy()
    data_collection["raw"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["raw"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = D.std(axis=1, skipna=True).to_numpy()
    data_collection["raw"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["raw"]["median_spatial_std"] = float(np.median(spatial_std))

    results, _ = interpolation_data_by_field(data)
    print(results)

    features = results.iloc[:, 1:]

    data_collection["clean"]["mask"] = np.nan
    data_collection["clean"]["invalid_number"] = np.nan
    data_collection["clean"]["total_number"] = np.nan
    data_collection["clean"]["valid_ratio"] = np.nan
    data_collection["clean"]["G_min"] = features.values.min()
    data_collection["clean"]["G_max"] = features.values.max()
    data_collection["clean"]["G_mean"] = features.values.mean()
    data_collection["clean"]["G_std"] = features.values.std()

    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["clean"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["clean"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["clean"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["clean"]["median_spatial_std"] = float(np.median(spatial_std))

    # z-socore normal
    features = std_data.iloc[:, 1:65]

    data_collection["std"]["mask"] = np.nan
    data_collection["std"]["invalid_number"] = np.nan
    data_collection["std"]["total_number"] = np.nan
    data_collection["std"]["valid_ratio"] = np.nan
    data_collection["std"]["G_min"] = features.values.min()
    data_collection["std"]["G_max"] = features.values.max()
    data_collection["std"]["G_mean"] = features.values.mean()
    data_collection["std"]["G_std"] = features.values.std()

    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["std"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["std"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["std"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["std"]["median_spatial_std"] = float(np.median(spatial_std))

    #minmax normal
    features = norm_data.iloc[:, 1:65]

    data_collection["norm"]["mask"] = np.nan
    data_collection["norm"]["invalid_number"] = np.nan
    data_collection["norm"]["total_number"] = np.nan
    data_collection["norm"]["valid_ratio"] = np.nan
    data_collection["norm"]["G_min"] = features.values.min()
    data_collection["norm"]["G_max"] = features.values.max()
    data_collection["norm"]["G_mean"] = features.values.mean()
    data_collection["norm"]["G_std"] = features.values.std()

    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["norm"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["norm"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["norm"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["norm"]["median_spatial_std"] = float(np.median(spatial_std))

    out_path = os.path.join("clean_data", "analysis.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    need_header = (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0)
    with open(out_path, "a+", encoding="utf-8") as results:
        if need_header:
            results.write(
                '"type","exp","mask","invalid_number","total_number","valid_ratio",'
                '"Global_min","Global_max","G_mean","Global_std",'
                '"avg_col_std","zero_var_cols","mean_spatial_std","median_spatial_std"\n'
            )
        for section, m in data_collection.items():
            line = (
                f'{section},{exp},{m.get("mask")},{m.get("invalid_number")},'
                f'{m.get("total_number")},{m.get("valid_ratio"):.5f},'
                f'{m.get("G_min"):.5f},{m.get("G_max"):.5f},{m.get("G_mean"):.5f},{m.get("G_std"):.5f},'
                f'{m.get("avg_col_std"):.5f},{m.get("zero_var_cols"):.5f},{m.get("mean_spatial_std"):.5f},{m.get("median_spatial_std"):.5f}\n'
            )
            results.write(line)


def filter_radar_data(raw_features, max_range=4.25, min_mag=300, max_angle=45,
                      range_jump_threshold=1, angle_jump_threshold=10):
    df = raw_features.copy()
    # ---- Step 1: Basic range/magnitude/angle filtering ----
    valid_mask = (
        (df["range"] > 0) &
        (df["range"] < max_range) &
        (df["magnitude"] > min_mag) &
        (df["angle"].abs() < max_angle)
    )
    df["valid_mask"] = valid_mask.astype(int)

    # ---- Step 2: Compute jump difference ----
    df["range_diff"] = df["range"].diff().abs()
    df["angle_diff"] = df["angle"].diff().abs()

    # ---- Step 3: Continuity filtering ----
    continuity_mask = (
        (df["range_diff"].fillna(0) < range_jump_threshold) &
        (df["angle_diff"].fillna(0) < angle_jump_threshold)
    )
    df["valid_mask"] &= continuity_mask.astype(int)

    print(f"The original number of data: {len(df)}")
    print(f"The number of filtered data: {df['valid_mask'].sum()}")
    print(f"The percentage: {df['valid_mask'].sum()/len(df)*100:.2f}%")

    # ---- Step 4: Replace invalid samples with NaN ----
    df.loc[df["valid_mask"] == 0, ["angle", "range", "magnitude"]] = np.nan


    # ---- Step 5: Linear interpolation for continuity ----
    df["angle"] = df["angle"].interpolate(method="linear", limit_direction="both").round(6)
    df["range"] = df["range"].interpolate(method="linear", limit_direction="both").round(6)
    df["magnitude"] = df["magnitude"].interpolate(method="linear", limit_direction="both").round(6)

    # ---- Step 6: Clean up helper columns ----
    df = df[['timestamp', 'angle', 'range', 'magnitude']].copy()

    return df


def analysis_mmwave(exp, raw_data, clean_data, std_data, norm_data):
    data_collection = {"raw": {}, "clean": {}, "std": {}, "norm": {}}

    D = raw_data.iloc[:, 1:]
    print(D.head())
    data_collection["raw"]["mask"] = np.nan
    data_collection["raw"]["invalid_number"] = np.nan
    data_collection["raw"]["total_number"] = D.size
    data_collection["raw"]["valid_ratio"] = np.nan
    data_collection["raw"]["angle_min"] = D.iloc[:, 0].min()
    data_collection["raw"]["angle_max"] = D.iloc[:, 0].max()
    data_collection["raw"]["angle_mean"] = D.iloc[:, 0].mean()
    data_collection["raw"]["angle_std"] = D.iloc[:, 0].std()
    data_collection["raw"]["range_min"] = D.iloc[:, 1].min()
    data_collection["raw"]["range_max"] = D.iloc[:, 1].max()
    data_collection["raw"]["range_mean"] = D.iloc[:, 1].mean()
    data_collection["raw"]["range_std"] = D.iloc[:, 1].std()
    data_collection["raw"]["magnitude_min"] = D.iloc[:, 2].min()
    data_collection["raw"]["magnitude_max"] = D.iloc[:, 2].max()
    data_collection["raw"]["magnitude_mean"] = D.iloc[:, 2].mean()
    data_collection["raw"]["magnitude_std"] = D.iloc[:, 2].std()


    col_std = D.std(axis=0, skipna=True).to_numpy()
    data_collection["raw"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["raw"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = D.std(axis=1, skipna=True).to_numpy()
    data_collection["raw"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["raw"]["median_spatial_std"] = float(np.median(spatial_std))

    features = clean_data.iloc[:, 1:]

    data_collection["clean"]["mask"] = np.nan
    data_collection["clean"]["invalid_number"] = np.nan
    data_collection["clean"]["total_number"] = features.size
    data_collection["clean"]["valid_ratio"] = np.nan
    data_collection["clean"]["angle_min"] = features.iloc[:, 0].min()
    data_collection["clean"]["angle_max"] = features.iloc[:, 0].max()
    data_collection["clean"]["angle_mean"] = features.iloc[:, 0].mean()
    data_collection["clean"]["angle_std"] = features.iloc[:, 0].std()
    data_collection["clean"]["range_min"] = features.iloc[:, 1].min()
    data_collection["clean"]["range_max"] = features.iloc[:, 1].max()
    data_collection["clean"]["range_mean"] = features.iloc[:, 1].mean()
    data_collection["clean"]["range_std"] = features.iloc[:, 1].std()
    data_collection["clean"]["magnitude_min"] = features.iloc[:, 2].min()
    data_collection["clean"]["magnitude_max"] = features.iloc[:, 2].max()
    data_collection["clean"]["magnitude_mean"] = features.iloc[:, 2].mean()
    data_collection["clean"]["magnitude_std"] = features.iloc[:, 2].std()


    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["clean"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["clean"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["clean"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["clean"]["median_spatial_std"] = float(np.median(spatial_std))

    # z-socore normal
    features = std_data.iloc[:, 1:]

    data_collection["std"]["mask"] = np.nan
    data_collection["std"]["invalid_number"] = np.nan
    data_collection["std"]["total_number"] = np.nan
    data_collection["std"]["valid_ratio"] = np.nan
    data_collection["std"]["angle_min"] = features.iloc[:, 0].min()
    data_collection["std"]["angle_max"] = features.iloc[:, 0].max()
    data_collection["std"]["angle_mean"] = features.iloc[:, 0].mean()
    data_collection["std"]["angle_std"] = features.iloc[:, 0].std()
    data_collection["std"]["range_min"] = features.iloc[:, 1].min()
    data_collection["std"]["range_max"] = features.iloc[:, 1].max()
    data_collection["std"]["range_mean"] = features.iloc[:, 1].mean()
    data_collection["std"]["range_std"] = features.iloc[:, 1].std()
    data_collection["std"]["magnitude_min"] = features.iloc[:, 2].min()
    data_collection["std"]["magnitude_max"] = features.iloc[:, 2].max()
    data_collection["std"]["magnitude_mean"] = features.iloc[:, 2].mean()
    data_collection["std"]["magnitude_std"] = features.iloc[:, 2].std()


    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["std"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["std"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["std"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["std"]["median_spatial_std"] = float(np.median(spatial_std))

    # minmax normal
    features = norm_data.iloc[:, 1:]

    data_collection["norm"]["mask"] = np.nan
    data_collection["norm"]["invalid_number"] = np.nan
    data_collection["norm"]["total_number"] = np.nan
    data_collection["norm"]["valid_ratio"] = np.nan
    data_collection["norm"]["angle_min"] = features.iloc[:, 0].min()
    data_collection["norm"]["angle_max"] = features.iloc[:, 0].max()
    data_collection["norm"]["angle_mean"] = features.iloc[:, 0].mean()
    data_collection["norm"]["angle_std"] = features.iloc[:, 0].std()
    data_collection["norm"]["range_min"] = features.iloc[:, 1].min()
    data_collection["norm"]["range_max"] = features.iloc[:, 1].max()
    data_collection["norm"]["range_mean"] = features.iloc[:, 1].mean()
    data_collection["norm"]["range_std"] = features.iloc[:, 1].std()
    data_collection["norm"]["magnitude_min"] = features.iloc[:, 2].min()
    data_collection["norm"]["magnitude_max"] = features.iloc[:, 2].max()
    data_collection["norm"]["magnitude_mean"] = features.iloc[:, 2].mean()
    data_collection["norm"]["magnitude_std"] = features.iloc[:, 2].std()

    col_std = features.std(axis=0, skipna=True).to_numpy()
    data_collection["norm"]["avg_col_std"] = float(np.mean(col_std))
    data_collection["norm"]["zero_var_cols"] = int(np.sum(col_std < 1e-12))

    spatial_std = features.std(axis=1, skipna=True).to_numpy()
    data_collection["norm"]["mean_spatial_std"] = float(np.mean(spatial_std))
    data_collection["norm"]["median_spatial_std"] = float(np.median(spatial_std))

    out_path = os.path.join("clean_data", "analysis_mmwave.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    need_header = (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0)
    with open(out_path, "a+", encoding="utf-8") as results:
        if need_header:
            results.write(
                '"type","exp","mask","invalid_number","total_number","valid_ratio",'
                '"angle_min","angle_max","angle_mean","angle_std",'
                '"range_min","range_max","range_mean","range_std",'
                '"magnitude_min","magnitude_max","magnitude_mean","magnitude_std",'
                '"avg_col_std","zero_var_cols","mean_spatial_std","median_spatial_std"\n'
            )
        for section, m in data_collection.items():
            line = (
                f'{section},{exp},{m.get("mask")},{m.get("invalid_number")},'
                f'{m.get("total_number")},{m.get("valid_ratio"):.5f},'
                f'{m.get("angle_min"):.5f},{m.get("angle_max"):.5f},{m.get("angle_mean"):.5f},{m.get("angle_std"):.5f},'
                f'{m.get("range_min"):.5f},{m.get("range_max"):.5f},{m.get("range_mean"):.5f},{m.get("range_std"):.5f},'
                f'{m.get("magnitude_min"):.5f},{m.get("magnitude_max"):.5f},{m.get("magnitude_mean"):.5f},{m.get("magnitude_std"):.5f},'
                f'{m.get("avg_col_std"):.5f},{m.get("zero_var_cols"):.5f},{m.get("mean_spatial_std"):.5f},{m.get("median_spatial_std"):.5f}\n'
            )
            results.write(line)


def cleaning_tof(exp):
    # default
    file_path = "./raw_data/data_VL53L7CH__AIKit__ZONE_8x8__20241029_115843.csv"
    sensor = "ToF"
    if exp == 1:
        file_path = "./raw_data/data_VL53L7CH__AIKit__ZONE_8x8__20241029_115843.csv"
        sensor = "ToF"
    elif exp == 2:
        file_path = "./raw_data/data_VL53L7CH__AIKit__ZONE_8x8__20241029_122238.csv"
        sensor = "ToF"
    elif exp == 3:
        file_path = "./raw_data/data_VL53L7CH__AIKit__ZONE_8x8__20241029_145832.csv"
        sensor = "ToF"
    elif exp == 4:
        file_path = "./raw_data/data_VL53L7CH__AIKit__ZONE_8x8__20241029_164900.csv"
        sensor = "ToF"

    df = read_data(file_path, sensor)
    raw_features = features_extract(df, sensor)
    print(raw_features.shape)
    print("before interpolation:")

    print(raw_features.head(10))

    # to check the extracted features
    raw_features.to_csv('./raw_data/raw.csv', index=False, header=True)

    result, _ = interpolation_data_by_field(raw_features)
    result.to_csv('./raw_data/features.csv', index=False, header=True)
    print(result.shape)
    print("after interpolation:")
    print(result.head(10))

    #read ultrasound data as label
    file_path = "./raw_data/exp1_10min_ultrasound.csv"
    sensor = "ultrasound"
    filename = "TOFEXP1"

    if exp == 1:
        file_path = "./raw_data/exp1_10min_ultrasound.csv"
        sensor = "ultrasound"
        filename = "TOFEXP1"
    elif exp == 2:
        file_path = "./raw_data/exp2_30min_ultrasound.csv"
        sensor = "ultrasound"
        filename = "TOFEXP2"
    elif exp == 3:
        file_path = "./raw_data/exp3_1h_ultrasound.csv"
        sensor = "ultrasound"
        filename = "TOFEXP3"
    elif exp == 4:
        file_path = "./raw_data/exp4_1h_ultrasound.csv"
        sensor = "ultrasound"
        filename = "TOFEXP4"

    df = read_data(file_path, sensor)

    labels = features_extract(df, sensor)

    print(f"The shape of label{labels.shape}")
    print(labels.head(10))

    # to check the extracted labels
    labels.to_csv('./raw_data/labels.csv', index=False, header=True)
    std_features, norm_features, _ = save_data(result, labels, filename, "mean")

    analysis_tof(exp, raw_features, std_features, norm_features)


def cleaning_mmwave(exp):
    # default
    file_path = "./raw_data/radar-walking-talha-exp1-10min.csv"
    sensor = "radar"
    if exp == 1:
        file_path = "./raw_data/radar-walking-talha-exp1-10min.csv"
        sensor = "radar"
    elif exp == 2:
        file_path = "./raw_data/radar-walking-talha-exp2-30min.csv"
        sensor = "radar"
    elif exp == 3:
        file_path = "./raw_data/radar-walking-talha-exp3-1h.csv"
        sensor = "radar"
    elif exp == 4:
        file_path = "./raw_data/radar-walking-talha-exp4-1h.csv"
        sensor = "radar"

    df = read_data(file_path, sensor)
    raw_features = features_extract(df, sensor)

    # to check the extracted features
    raw_features.to_csv('./raw_data/raw.csv', index=False, header=True)
    print(raw_features.head())

    results = filter_radar_data(raw_features)
    results.to_csv('./raw_data/features.csv', index=False, header=True)
    print(results.shape)

    #read ultrasound data as label
    file_path = "./raw_data/exp1_10min_ultrasound.csv"
    sensor = "ultrasound"
    filename = "mmWaveEXP1"

    if exp == 1:
        file_path = "./raw_data/exp1_10min_ultrasound.csv"
        sensor = "ultrasound"
        filename = "mmWaveEXP1"
    elif exp == 2:
        file_path = "./raw_data/exp2_30min_ultrasound.csv"
        sensor = "ultrasound"
        filename = "mmWaveEXP2"
    elif exp == 3:
        file_path = "./raw_data/exp3_1h_ultrasound.csv"
        sensor = "ultrasound"
        filename = "mmWaveEXP3"
    elif exp == 4:
        file_path = "./raw_data/exp4_1h_ultrasound.csv"
        sensor = "ultrasound"
        filename = "mmWaveEXP4"

    df = read_data(file_path, sensor)

    labels = features_extract(df, sensor)

    print(f"The shape of label:{labels.shape}")

    # to check the extracted labels
    labels.to_csv('./raw_data/labels.csv', index=False, header=True)
    std_features, norm_features, _ = save_data_mmwave(results, labels, filename, "mean")
    analysis_mmwave(exp, raw_features, results, std_features, norm_features)

    print(f"\n=== EXP{exp} Summary ===")
    for col in ['angle', 'range', 'magnitude']:
        mean_before = raw_features[col].mean()
        std_before = raw_features[col].std()
        mean_after = results[col].mean()
        std_after = results[col].std()
        min_after = results[col].min()
        max_after = results[col].max()
        print(f"{col.capitalize():<10} mean/std before filter: {mean_before:.3f}, {std_before:.3f}")
        print(f"{col.capitalize():<10} mean/std after  filter: {mean_after:.3f}, {std_after:.3f}")
        print(f"{col.capitalize():<10} min/max after filter: {min_after:.3f} ~ {max_after:.3f}\n")

    print(f"Total samples kept: {len(results)}/{len(raw_features)} = {len(results) / len(raw_features) * 100:.1f}%")


if __name__ == '__main__':

    # Check ultrasound trajectory
    #fix_ultraSound_trajectory()

    #main function to clean data
    # cleaning_tof(1)
    # cleaning_tof(2)
    # cleaning_tof(3)
    # cleaning_tof(4)
    # #
    # # # read as experiment data
    # df = pd.read_csv('clean_data/std_TOFEXP1.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP1.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_TOFEXP2.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP2.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_TOFEXP3.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP3.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_TOFEXP4.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP4.csv', index=False, header=False)
    #
    # df = pd.read_csv('clean_data/norm_TOFEXP1.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/norm_TOFEXP1.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/norm_TOFEXP2.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/norm_TOFEXP2.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/norm_TOFEXP3.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/norm_TOFEXP3.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/norm_TOFEXP4.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/norm_TOFEXP4.csv', index=False, header=False)
    #
    # df = pd.read_csv('clean_data/mixed_TOFEXP1.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_TOFEXP1.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_TOFEXP2.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_TOFEXP2.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_TOFEXP3.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_TOFEXP3.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_TOFEXP4.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_TOFEXP4.csv', index=False, header=False)
    #
    #
    # # #clean mmWave radar
    cleaning_mmwave(1)
    cleaning_mmwave(2)
    cleaning_mmwave(3)
    cleaning_mmwave(4)
    #
    # # # read as experiment data
    # df = pd.read_csv('clean_data/std_mmWaveEXP1.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_mmWaveEXP1.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_mmWaveEXP2.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_mmWaveEXP2.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_mmWaveEXP3.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_mmWaveEXP3.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/std_mmWaveEXP4.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/std_mmWaveEXP4.csv', index=False, header=False)
    #
    df = pd.read_csv('clean_data/norm_mmWaveEXP1.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/norm_mmWaveEXP1.csv', index=False, header=False)
    df = pd.read_csv('clean_data/norm_mmWaveEXP2.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/norm_mmWaveEXP2.csv', index=False, header=False)
    df = pd.read_csv('clean_data/norm_mmWaveEXP3.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/norm_mmWaveEXP3.csv', index=False, header=False)
    df = pd.read_csv('clean_data/norm_mmWaveEXP4.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/norm_mmWaveEXP4.csv', index=False, header=False)
    #
    # df = pd.read_csv('clean_data/mixed_mmWaveEXP1.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_mmWaveEXP1.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_mmWaveEXP2.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_mmWaveEXP2.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_mmWaveEXP3.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_mmWaveEXP3.csv', index=False, header=False)
    # df = pd.read_csv('clean_data/mixed_mmWaveEXP4.csv')
    # data = df[df.columns[1:]].to_csv('../exp_data/mixed_mmWaveEXP4.csv', index=False, header=False)


    # 全面比较
    # def comprehensive_compare(df1, df2):
    #     # 1. 形状比较
    #     print("形状比较:")
    #     print(f"df1 shape: {df1.shape}")
    #     print(f"df2 shape: {df2.shape}")
    #
    #     # 2. 列名比较
    #     print("\n列名比较:")
    #     print("df1 columns:", df1.columns.tolist())
    #     print("df2 columns:", df2.columns.tolist())
    #
    #     # 3. 数值差异
    #     # print("\n数值差异:")
    #     # diff = df1.compare(df2)
    #     # print(diff.to_numpy().sum())
    #
    #     # 4. 统计描述
    #     print("\n描述性统计:")
    #     print("df1 describe:")
    #     print(df1.describe())
    #     print("\ndf2 describe:")
    #     print(df2.describe())
    #     #
    #     # # 5. 绝对误差
    #     # print("\n绝对误差:")
    #     # abs_diff = np.abs(df1 - df2)
    #     # print(abs_diff.to_numpy().sum())
    #
    #
    # comprehensive_compare(df1, df2)















