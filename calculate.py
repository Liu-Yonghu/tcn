import glob, os
import scipy.stats as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib

def compute_feature_label_stats(csv_path, save_plot=True, n_neighbors=5):
    df = pd.read_csv(csv_path, header=None)
    feature_cols = df.columns[:-2]
    x_col, y_col = df.columns[-2], df.columns[-1]

    # --- Pearson correlation ---
    corr_x = df[feature_cols].corrwith(df[x_col])
    corr_y = df[feature_cols].corrwith(df[y_col])

    # --- Mutual information ---
    X_features = df[feature_cols].values
    Yx = df[x_col].values
    Yy = df[y_col].values

    mi_x = mutual_info_regression(X_features, Yx, discrete_features=False, n_neighbors=n_neighbors, random_state=42)
    mi_y = mutual_info_regression(X_features, Yy, discrete_features=False, n_neighbors=n_neighbors, random_state=42)

    # --- Combine results ---
    stats_df = pd.DataFrame({
        "feature": feature_cols,
        "pearson_x": corr_x.values,
        "pearson_y": corr_y.values,
        "mi_x": mi_x,
        "mi_y": mi_y
    })

    # --- Print summary ---
    print(f"\n{os.path.basename(csv_path)} summary:")
    print(f"Mean |r(x)| = {stats_df['pearson_x'].abs().mean():.4f}, Std = {stats_df['pearson_x'].std():.4f}")
    print(f"Mean |r(y)| = {stats_df['pearson_y'].abs().mean():.4f}, Std = {stats_df['pearson_y'].std():.4f}")
    print(f"Mean MI(x)  = {stats_df['mi_x'].mean():.4f}, Std = {stats_df['mi_x'].std():.4f}")
    print(f"Mean MI(y)  = {stats_df['mi_y'].mean():.4f}, Std = {stats_df['mi_y'].std():.4f}")

    # Identify strongest and weakest features
    max_px = stats_df.loc[stats_df['pearson_x'].abs().idxmax()]
    max_py = stats_df.loc[stats_df['pearson_y'].abs().idxmax()]
    max_mx = stats_df.loc[stats_df['mi_x'].idxmax()]
    max_my = stats_df.loc[stats_df['mi_y'].idxmax()]

    print(f"\n Strongest linear correlation:")
    print(f"  X → feature {max_px['feature']} : r = {max_px['pearson_x']:.3f}")
    print(f"  Y → feature {max_py['feature']} : r = {max_py['pearson_y']:.3f}")
    print(f"\n Strongest nonlinear dependency (Mutual Information):")
    print(f"  X → feature {max_mx['feature']} : MI = {max_mx['mi_x']:.5f}")
    print(f"  Y → feature {max_my['feature']} : MI = {max_my['mi_y']:.5f}")

    # --- Visualization ---
    if save_plot:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]

        # Pearson heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            stats_df[["pearson_x", "pearson_y"]].T,
            cmap="coolwarm",
            center=0,
            vmin=-0.6, vmax=0.6,
            cbar_kws={"label": "Pearson r"},
            annot=True
        )
        plt.yticks([0.5, 1.5], ["X", "Y"], rotation=0)
        # plt.xlabel("Feature index")
        plt.xticks([0.5, 1.5, 2.5], ["Angle", "Range", "Magnitude"], rotation=0)
        plt.title(f"Pearson Correlation ({base_name})")
        plt.tight_layout()
        plt.savefig(f"{base_name}_pearson_heatmap.png", dpi=300)
        plt.close()

        # Mutual information heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            stats_df[["mi_x", "mi_y"]].T,
            cmap="YlGnBu",
            cbar_kws={"label": "Mutual Information"},
            annot=True
        )
        plt.yticks([0.5, 1.5], ["X", "Y"], rotation=0)
        # plt.xlabel("Distance 0-63")
        plt.xticks([0.5, 1.5, 2.5], ["Angle", "Range", "Magnitude"], rotation=0)
        plt.title(f"Mutual Information ({base_name})")
        plt.tight_layout()
        plt.savefig(f"{base_name}_mi_heatmap.png", dpi=300)
        plt.close()

        print(f"Saved heatmaps: {base_name}_pearson_heatmap.png, {base_name}_mi_heatmap.png")

    return stats_df


def compute_feature_label_spearman(csv_path, save_plot=True):
    df = pd.read_csv(csv_path, header=None)
    feature_cols = df.columns[:-2]  # last x,y
    x_col, y_col = df.columns[-2], df.columns[-1]

    corr_x = []
    corr_y = []

    # calculate spearman correlation
    for i in feature_cols:
        r_x, _ = spearmanr(df.iloc[:, i], df[x_col])
        r_y, _ = spearmanr(df.iloc[:, i], df[y_col])
        corr_x.append(r_x)
        corr_y.append(r_y)

    corr_df = pd.DataFrame({
        "feature_index": feature_cols,
        "spearman_with_x": corr_x,
        "spearman_with_y": corr_y
    })

    # print info
    print(f"\n{os.path.basename(csv_path)} Spearman correlation summary:")
    print(f"Mean |ρ(x)|: {corr_df['spearman_with_x'].abs().mean():.3f}, Std: {corr_df['spearman_with_x'].std():.3f}")
    print(f"Mean |ρ(y)|: {corr_df['spearman_with_y'].abs().mean():.3f}, Std: {corr_df['spearman_with_y'].std():.3f}")

    # min max
    max_x = corr_df.loc[corr_df['spearman_with_x'].idxmax()]
    min_x = corr_df.loc[corr_df['spearman_with_x'].idxmin()]
    max_y = corr_df.loc[corr_df['spearman_with_y'].idxmax()]
    min_y = corr_df.loc[corr_df['spearman_with_y'].idxmin()]

    print(f"\n X correlation:")
    print(f"  Max ρ: {max_x['spearman_with_x']:.3f}  (feature index {max_x['feature_index']})")
    print(f"  Min ρ: {min_x['spearman_with_x']:.3f}  (feature index {min_x['feature_index']})")

    print(f"\n Y correlation:")
    print(f"  Max ρ: {max_y['spearman_with_y']:.3f}  (feature index {max_y['feature_index']})")
    print(f"  Min ρ: {min_y['spearman_with_y']:.3f}  (feature index {min_y['feature_index']})")

    if save_plot:
        heat_df = corr_df[["spearman_with_x", "spearman_with_y"]].T

        plt.figure(figsize=(8, 4))
        ax = sns.heatmap(
            heat_df,
            cmap="coolwarm",
            center=0,
            annot=False,
            fmt=".3f",
            cbar_kws={"label": "Spearman ρ"},
            vmin=-0.6,
            vmax=0.6,
        )

        # ax.set_xticks([0.5, 1.5, 2.5])
        # ax.set_xticklabels(["Angle", "Range", "Magnitude"], rotation=0)

        ax.set_xlabel("Distance 0-63")
        # ax.set_ylabel("Label")
        ax.set_yticklabels(["X", "Y"], rotation=0)
        plt.title(f"Spearman Correlation ({os.path.basename(csv_path)})")
        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(csv_path)[0]}_spearman_heatmap.png", dpi=300)
        plt.close()

        print(f"Saved heatmap: {os.path.splitext(csv_path)[0]}_spearman_heatmap.png")

    return corr_df


def check_data_validity(csv_path, exp_name="EXP1", save_dir="data_check"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== Checking {exp_name} data validity ===")

    # read data
    df = pd.read_csv(csv_path, header=None)
    n_cols = df.shape[1]

    # name：f1, f2, ... , X, Y
    feature_cols = [f"f{i}" for i in range(1, n_cols - 1)]
    label_cols = ["X", "Y"]
    df.columns = feature_cols + label_cols

    print(f"Loaded {len(df)} samples, {len(feature_cols)} features + 2 labels (X,Y)\n")

    # description
    desc = df.describe().T
    desc.to_csv(f"{save_dir}/{exp_name}_describe.csv")
    print(desc, "\n")

    # features distribution
    for col in feature_cols:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f"{exp_name} - {col} Distribution")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{exp_name}_{col}_hist.png", dpi=200)
        plt.close()

    # trajectory and speed
    plt.figure(figsize=(6, 6))
    plt.plot(df['X'], df['Y'], '-', alpha=0.7)
    plt.scatter(df['X'].iloc[0], df['Y'].iloc[0], c='g', label='Start')
    plt.scatter(df['X'].iloc[-1], df['Y'].iloc[-1], c='r', label='End')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{exp_name} - Trajectory")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(f"{save_dir}/{exp_name}_trajectory.png", dpi=200)
    plt.close()

    vx = np.diff(df['X'])
    vy = np.diff(df['Y'])
    speed = np.sqrt(vx ** 2 + vy ** 2)*4
    plt.figure(figsize=(6, 3))
    plt.plot(speed, color='purple')
    plt.title(f"{exp_name} - Speed over time")
    plt.xlabel("Frame")
    plt.ylabel("Speed (m/s)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{exp_name}_speed.png", dpi=200)
    plt.close()

    # === 6. mutual info + R² ===
    X = df[feature_cols]
    mi_x = mutual_info_regression(X, df['X'])
    mi_y = mutual_info_regression(X, df['Y'])
    model_x = LinearRegression().fit(X, df['X'])
    model_y = LinearRegression().fit(X, df['Y'])
    r2_x = r2_score(df['X'], model_x.predict(X))
    r2_y = r2_score(df['Y'], model_y.predict(X))

    # summary
    summary = f"""=== {exp_name} Data Quality Summary ===
    File: {csv_path}

    Features: {feature_cols}
    Labels: X, Y


    [Mutual Information]
    X: {mi_x}
    Y: {mi_y}

    [Linear Regression R²]
    R²(X) = {r2_x:.3f}
    R²(Y) = {r2_y:.3f}
    """
    print(summary)
    with open(f"{save_dir}/{exp_name}_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Results saved to: {save_dir}/")


def analyze_feature_label_corr(csv_path, shift_range=(-16, 16), skiprows=0, show_plot=True):
    data = pd.read_csv(csv_path).to_numpy()
    if isinstance(data, str):
        df = pd.read_csv(data, header=None, skiprows=skiprows)
        data = df.to_numpy()
    else:
        data = np.asarray(data)

    n_feat = data.shape[1] - 2
    feature_names = [f"f{i}" for i in range(1, n_feat + 1)]
    label_names = ["X", "Y"]

    results = []

    # Iterate through different shifts
    for shift in range(shift_range[0], shift_range[1] + 1):
        if shift > 0:
            mm = data[shift:, :-2]
            gt = data[:len(mm), -2:]
        elif shift < 0:
            mm = data[:shift, :-2]
            gt = data[-shift:, -2:]
        else:
            mm = data[:, :-2]
            gt = data[:, -2:]

        corr_x, corr_y = [], []
        for i in range(n_feat):
            corr_x.append(pearsonr(mm[:, i], gt[:, 0])[0])
            corr_y.append(pearsonr(mm[:, i], gt[:, 1])[0])

        mean_x = np.nanmean(np.abs(corr_x))
        mean_y = np.nanmean(np.abs(corr_y))
        results.append((shift, mean_x, mean_y))

    # find best shift
    res_df = pd.DataFrame(results, columns=["shift", "mean|r(X)|", "mean|r(Y)|"])
    best_shift_x = res_df.iloc[res_df["mean|r(X)|"].idxmax()]["shift"]
    best_shift_y = res_df.iloc[res_df["mean|r(Y)|"].idxmax()]["shift"]

    print("=== shift-correlation analysis ===")
    print(res_df)
    print(f"\nbest time shift：X = {best_shift_x:+f} frame, Y = {best_shift_y:+f} frame")

    # plot shift curve
    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(res_df["shift"], res_df["mean|r(X)|"], label="Mean |r(X)|", color='b')
        plt.plot(res_df["shift"], res_df["mean|r(Y)|"], label="Mean |r(Y)|", color='r')
        plt.axvline(best_shift_x, color='b', linestyle='--', alpha=0.7)
        plt.axvline(best_shift_y, color='r', linestyle='--', alpha=0.7)
        plt.xlabel("Frames shift")
        plt.ylabel("Mean |r|")
        plt.ylim(0, 1)
        plt.title("Feature-Label Correlation vs. Time Shift")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(csv_path)[0]}_shift_correlation.png", dpi=300)

    return res_df, best_shift_x, best_shift_y

def analyze_feature_label_corr_multi(csv_files, labels=None, shift_range=(-8, 8), save_path="multi_shift_corr_.png"):
    plt.figure(figsize=(9, 5))
    all_results = []

    for idx, csv_path in enumerate(csv_files):
        data = pd.read_csv(csv_path, header=None).to_numpy()
        n_feat = data.shape[1] - 2

        results = []

        # 遍历不同 shift
        for shift in range(shift_range[0], shift_range[1] + 1):
            if shift > 0:       # feature late
                mm = data[shift:, :-2]
                gt = data[:len(mm), -2:]
            elif shift < 0:     # feature advance
                mm = data[:shift, :-2]
                gt = data[-shift:, -2:]
            else:
                mm = data[:, :-2]
                gt = data[:, -2:]

            corr_x, corr_y = [], []
            for i in range(n_feat):
                r_x, _ = pearsonr(mm[:, i], gt[:, 0])
                r_y, _ = pearsonr(mm[:, i], gt[:, 1])
                corr_x.append(r_x)
                corr_y.append(r_y)

            mean_x = np.nanmean(np.abs(corr_x))
            mean_y = np.nanmean(np.abs(corr_y))
            results.append((shift, mean_x, mean_y))

        res_df = pd.DataFrame(results, columns=["shift", "mean|r(X)|", "mean|r(Y)|"])
        best_shift_x = res_df.iloc[res_df["mean|r(X)|"].idxmax()]["shift"]
        best_shift_y = res_df.iloc[res_df["mean|r(Y)|"].idxmax()]["shift"]

        # save results
        all_results.append((csv_path, best_shift_x, best_shift_y))

        label = labels[idx] if labels else f"EXP{idx+1}"
        plt.plot(res_df["shift"], res_df["mean|r(X)|"],
                 label=f"{label} (X)", lw=1.8)
        plt.plot(res_df["shift"], res_df["mean|r(Y)|"],
                 linestyle="--", lw=1.3, alpha=0.8,
                 label=f"{label} (Y)")

    # save figure
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Time Shift (frames)")
    plt.ylabel("Mean |r|")
    plt.ylim(0,1)

    plt.title("Feature-Label Correlation vs. Time Shift (All Experiments)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=400)
    print(f"\nSaved multi-experiment shift correlation plot: {save_path}")

    return all_results

def mean_folds_val_loss(base_dir):
    base_dir = base_dir or "results/std_TOFEXP1"
    fold_means = []

    # iterate 6 folds
    for fold in range(1, 7):
        csv_path = os.path.join(base_dir, str(fold), "NAS_results.csv")
        df = pd.read_csv(csv_path)

        fold_mean = df['Min val MSE'].mean()
        fold_means.append(fold_mean)
        print(f"Fold {fold} mean: {fold_mean:.4f}")

    fold_means = np.array(fold_means)

    # Grand mean
    grand_mean = fold_means.mean()

    # Standard error
    s = fold_means.std(ddof=1)
    se = s / np.sqrt(len(fold_means))

    # 95% CI
    dfree = len(fold_means) - 1
    t_crit = st.t.ppf(0.975, dfree)
    ci_low = grand_mean - t_crit * se
    ci_high = grand_mean + t_crit * se

    print("\n====== Overall Statistics ======")
    print("Fold means:", fold_means)
    print(f"Grand mean: {grand_mean:.4f}")
    print(f"Standard error: {se:.4f}")
    print(f"95% CI: ({ci_low:.4f}, {ci_high:.4f})")


if __name__ == "__main__":

    # mean_folds_val_loss("E1/norm_mmWaveEXP4")
    # file = r"D:\HuaweiMoveData\Users\liuyo\Desktop\Thesis\ML_models\datasets\preprocessed-IrEXP2.csv"
    #
    # compute_feature_label_spearman(file)

    # file = "exp_data/norm_TOFEXP1.csv"
    #
    # compute_feature_label_correlation(file)
    # features = "data_preprocess/raw_data/radar-walking-talha-exp4-1h.csv"
    # label = "data_preprocess/raw_data/exp4_1h_ultrasound.csv"
    #
    # compute_correlation_downsample(features, label)
    for i in range(4):
        file = f"exp_data/norm_mmWaveEXP{i+1}.csv"
        compute_feature_label_stats(file)
        # compute_feature_label_spearman(file)

    # check_data_validity("exp_data/norm_mmWaveEXP2.csv", exp_name="mmWaveEXP2")


    # analyze_feature_label_corr("exp_data/norm_mmWaveEXP1.csv")

    # csv_list = [
    #     "exp_data/norm_mmWaveEXP1.csv",
    #     "exp_data/norm_mmWaveEXP2.csv",
    #     "exp_data/norm_mmWaveEXP3.csv",
    #     "exp_data/norm_mmWaveEXP4.csv"
    # ]
    # #
    # # csv_list = [
    # #     "exp_data/norm_TOFEXP1.csv",
    # #     "exp_data/norm_TOFEXP2.csv",
    # #     "exp_data/norm_TOFEXP3.csv",
    # #     "exp_data/norm_TOFEXP4.csv"
    # # ]
    #
    # labels = ["EXP1", "EXP2", "EXP3", "EXP4"]
    #
    # analyze_feature_label_corr_multi(csv_list, labels=labels, shift_range=(-12, 12))

