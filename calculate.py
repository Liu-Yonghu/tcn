import pandas as pd
import numpy as np
import scipy.stats as st
import glob, os


def mean_folds_val_loss(base_dir):
    base_dir = base_dir or "results/std_TOFEXP1"
    fold_means = []

    # 遍历6折
    for fold in range(1, 7):
        csv_path = os.path.join(base_dir, str(fold), "NAS_results.csv")
        df = pd.read_csv(csv_path)

        # 假设 validation score 存在 'val_loss' 列
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
    mean_folds_val_loss("results/std_TOFEXP2")
