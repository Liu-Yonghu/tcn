import os
import pandas as pd
import numpy as np
def read_data(file_path, sensor):
    # read CSV file
    try:
        if sensor.lower() == 'radar':
            df = pd.read_csv(file_path, header=None, names=['angle', 'range', 'magnitude', 'timestamp'])
        else:
            df = pd.read_csv(file_path)
        print(f'the size of this raw dataset: {df.shape}')
        return df
    except Exception as e:
        print(f'Failed to load data: {e}')


def features_extract(df, sensor):
    #extract the the useful fields which we want
    data = []
    if sensor.lower() == 'tof':

        distance_cols = [col for col in df.columns if col.startswith('distance')]
        signal_cols = [col for col in df.columns if col.startswith("signal_per_spad")]
        valid_cols = [col for col in df.columns if col.startswith(".is_valid_range")]
        timestamp_cols = [col for col in df.columns if col.startswith(".HostTimestamp")]

        # features = df[distance_cols].values.astype(np.float32)
        # signals = df[signal_cols].values.astype(np.float32)  # probably used later
        # valid = df[valid_cols].values.astype(np.float32)
        # timestamp = df[timestamp_cols].values.astype((np.float32))
        #
        # data = np.concatenate([timestamp, features, valid], axis=1)

        data = pd.concat([df[timestamp_cols], df[distance_cols], df[valid_cols]], axis=1)

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
        data = pd.concat([df[timestamp_cols], df[angle_cols], df[range_cols], df[max_magnitude_cols]], axis=1)

    else:
        print(" no this kind of sensor ")
    return data

def interpolation_data_by_samples(X):
    timestamp_col = X.columns[0]
    distance_cols = X.columns[1:65]
    valid_cols = X.columns[65:]

    interpolated_rows = []
    for idx, row in X.iterrows():
        distances = row[distance_cols].astype(float)
        valids = row[valid_cols].astype(bool)

        # 构建 Series 并插值
        valids.index = distances.index
        interpolated = distances.mask(~valids).interpolate(method='linear', limit_direction='both')

        # 合并结果
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
    valid_cols = X.columns[65:]

    distance_interp = X[distance_cols].copy()
    valid_mask = X[valid_cols].astype(bool)

    # interpolate on every column
    for col in distance_cols:
        valid_col = valid_cols[distance_cols.get_loc(col)]
        mask = valid_mask[valid_col]
        distance_interp[col] = X[col].mask(~mask).interpolate(method='linear', limit_direction='both')

    result = pd.concat([X[timestamp_col], distance_interp], axis=1)
    return result

def save_data(data, labels, filename, method = "interpolated"):
    timestamp = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    data_norm = (features - features.min()) / (features.max() - features.min())
    data_std = (features - features.mean()) / features.std()

    path = 'clean_data/'
    os.makedirs(path, exist_ok=True)

    data_norm = pd.concat([timestamp, data_norm], axis=1)
    data_std = pd.concat([timestamp, data_std], axis=1)

    if method == "mean":

        labels_down = mean_labels(labels, len(data))

        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]


    elif method == "downsample":
        indices = np.linspace(0, len(labels) - 1, num=len(data), dtype=int)
        labels_down = labels.iloc[indices].reset_index(drop=True)
        data_norm = data_norm.reset_index(drop=True)
        data_std = data_std.reset_index(drop=True)

        # concat the result
        data_norm["X"] = labels_down["X"]
        data_norm["Y"] = labels_down["Y"]

        data_std["X"] = labels_down["X"]
        data_std["Y"] = labels_down["Y"]
    else:
        raise ValueError("Unsupported method. Choose from ['mean', 'downsample'].")

    data_norm.to_csv(os.path.join(path, f'norm_{filename}.csv'), index=False, header=True)
    data_std.to_csv(os.path.join(path, f'std_{filename}.csv'), index=False, header=True)
    print(f"Saved norm_{filename} and std_{filename} to {path}")

def mean_labels(labels, target_len):
    n = len(labels)
    step = n / target_len
    averaged_labels = []

    for i in range(target_len):
        start = int(i * step)
        end = int((i + 1) * step)
        chunk = labels.iloc[start:end]
        averaged = chunk.mean(numeric_only=True)
        averaged_labels.append(averaged)

    print(len(averaged_labels), target_len)
    return pd.DataFrame(averaged_labels).reset_index(drop=True)

def cleaning(exp):
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

    # to check the extracted features
    raw_features.to_csv('./raw_data/raw.csv', index=False, header=True)

    result = interpolation_data_by_field(raw_features)
    result.to_csv('./raw_data/features.csv', index=False, header=True)
    print(result.shape)

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

    print(labels.shape)

    # to check the extracted labels
    labels.to_csv('./raw_data/labels.csv', index=False, header=True)
    save_data(result, labels, filename, "mean")

def clean_trajectory_general(df, range_min=0, range_max=3, jump_thresh=0.3, stuck_thresh=1e-3, stuck_len=16):
    df = df.copy()
    N = len(df)
    df["valid_mask"] = 1

    #ensure it is a number
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



if __name__ == '__main__':

    # Check ultrasound trajectory
    # fix_ultraSound_trajectory()

    # main function to clean data
    cleaning(1)
    cleaning(2)
    cleaning(3)
    cleaning(4)

    df = pd.read_csv('clean_data/std_TOFEXP1.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP1.csv', index=False, header=False)
    df = pd.read_csv('clean_data/std_TOFEXP2.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP2.csv', index=False, header=False)
    df = pd.read_csv('clean_data/std_TOFEXP3.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP3.csv', index=False, header=False)
    df = pd.read_csv('clean_data/std_TOFEXP4.csv')
    data = df[df.columns[1:]].to_csv('../exp_data/std_TOFEXP4.csv', index=False, header=False)














