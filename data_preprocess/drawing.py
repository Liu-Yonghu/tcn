import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from clean_data import read_data, features_extract, interpolation_data_by_field
matplotlib.use('TkAgg')
def animate_depth(frames, save_path, isHandled=None):
    fig, ax = plt.subplots()
    if isHandled == "minMax":
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=0, vmax=1)
    elif isHandled == "zScore":
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=-3, vmax=3)
    else:
        min = np.min(frames)
        max = np.max(frames)
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=min, vmax=max)

    plt.colorbar(im, ax=ax, label="Depth (m)")
    frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    def update(frame_index):
        im.set_array(frames[frame_index])
        frame_text.set_text(f'Frame: {frame_index + 1}')
        return [im, frame_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True, repeat=False)
    ani.save(save_path, writer="ffmpeg", fps=5)
    #plt.show()
    plt.close()
    return ani

def check_one_frame(frame, save_path, isHandled=False):
    # the first frame
    print(frame)
    print("----------------------------------------------------")
    frame_row_major = frame.reshape(8, 8)
    print(frame_row_major)
    print("-------------------------------------------------------")

    fig, ax = plt.subplots()
    if isHandled:
        im = ax.imshow(frame_row_major, cmap="jet", vmin=-3, vmax=3)
    else:
        im = ax.imshow(frame_row_major, cmap="jet", vmin=0, vmax=3)

    ax.set_title("Row-major reshape (default)")
    plt.colorbar(im, ax=ax)
    plt.savefig(save_path, format='jpg')
    plt.close()

def normalize_ground_truth(coords, room_size=(3.0, 3.0)):
    return coords / np.array(room_size)

def animate_trajectory(coords, interval=50, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
    # ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Ground Truth Trajectory Animation")
    ax.grid(True, linestyle="--", alpha=0.5)

    line, = ax.plot([], [], '-', c="blue", alpha=0.7)
    point, = ax.plot([], [], 'ro', markersize=6)  # 当前点
    start = ax.plot(coords[0, 0], coords[0, 1], 'go', markersize=8, label="Start")[0]
    end = ax.plot(coords[-1, 0], coords[-1, 1], 'rx', markersize=8, label="End")[0]
    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(i):
        line.set_data(coords[:i, 0], coords[:i, 1])  # 轨迹线
        point.set_data([coords[i, 0]], [coords[i, 1]])  # 当前点（注意加方括号）
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(coords),
                                  init_func=init, interval=interval, blit=True, repeat=False)

    if save_path:
        ani.save(save_path, writer="ffmpeg" if save_path.endswith(".mp4") else "imagemagick")
        print(f"动画已保存到 {save_path}")

    plt.savefig()
    return ani

def analyze_dataset(file_path, name="dataset", jump_thresh=0.3):
    df = pd.read_csv(file_path, names=["t", "X", "Y", "valid"])
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna()

    # basic info
    N = len(df)
    x_stats = df["X"].agg(["min", "max", "mean", "std"])
    y_stats = df["Y"].agg(["min", "max", "mean", "std"])

    # noise/jump detection
    dx = df["X"].diff()
    dy = df["Y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    jump_ratio = (dist > jump_thresh).mean()

    print(f"===== {name} =====")
    print(f"total frames: {N}")
    print(f"x coordinate: {x_stats.to_dict()}")
    print(f"y coordinate: {y_stats.to_dict()}")
    print(f"mean shift: {dist.mean():.4f}, max shift: {dist.max():.4f}")
    print(f"Jump ratio (> {jump_thresh}): {jump_ratio:.2%}")

    return df, dist

def compare_datasets(file1, file2, name1="Dataset A", name2="Dataset B", jump_thresh=0.3):
    df1, dist1 = analyze_dataset(file1, name1, jump_thresh)
    df2, dist2 = analyze_dataset(file2, name2, jump_thresh)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df1["X"], df1["Y"], "b.", alpha=0.5, label=name1)
    plt.plot(df2["X"], df2["Y"], "r.", alpha=0.5, label=name2)
    plt.legend()
    plt.title("Trajectory Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(dist1.dropna(), bins=50, alpha=0.5, label=name1)
    plt.hist(dist2.dropna(), bins=50, alpha=0.5, label=name2)
    plt.legend()
    plt.title("Step Distance Distribution")

    plt.tight_layout()
    plt.savefig()

def animate_depth_with_mask(frames, masks, save_path, isHandled=None):
    fig, ax = plt.subplots()
    if isHandled == "minMax":
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=0, vmax=1)
    elif isHandled == "zScore":
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=-3, vmax=3)
    else:
        min = np.min(frames)
        max = np.max(frames)
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=min, vmax=max)

    plt.colorbar(im, ax=ax, label="Depth (m)")
    frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # 初始 mask
    mask_rgba = np.zeros((*frames[0].shape, 4))   # RGBA
    mask_rgba[..., 3] = masks[0] * 1.0          # alpha 通道 = 1 表示不透明
    mask_im = ax.imshow(mask_rgba, animated=True, zorder=10)

    def update(frame_index):
        im.set_array(frames[frame_index])

        mask_rgba = np.zeros((*frames[frame_index].shape, 4))
        mask_rgba[..., 3] = masks[frame_index] * 1.0  # alpha=1 → 黑色
        mask_im.set_array(mask_rgba)

        frame_text.set_text(f'Frame: {frame_index + 1}')
        return [im, mask_im, frame_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=200, blit=True, repeat=False)

    ani.save(save_path, writer="ffmpeg", fps=5)
    #plt.show()
    plt.close()
    return ani

def generate_animate_depth_mask(exp):
    # default
    os.makedirs("./animation/", exist_ok=True)
    file_path = ""
    sensor = ""

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

    _, mask = interpolation_data_by_field(raw_features)

    data = pd.read_csv('./clean_data/std_TOFEXP4.csv', usecols=range(1, 65)).to_numpy()
    frames = data.reshape(-1, 8, 8)
    mask = mask.reshape(-1, 8, 8)
    #mask = np.ones((14777, 8, 8), dtype=int)
    isHandled = "zScore"
    animate_depth_with_mask(frames, ~mask, f"./animation/std_TOFEXP{exp}_with_mask.mp4", isHandled)


if __name__ == "__main__":
    os.makedirs("./animation/", exist_ok=True)
    generate_animate_depth_mask(4)

    # data = pd.read_csv('./clean_data/std_TOFEXP4.csv', usecols=range(1, 65)).to_numpy()
    # frames = data.reshape(-1, 8, 8)
    # print(data.shape)

    # data = pd.read_csv('./raw_data/features.csv', usecols=range(1, 65)).to_numpy()
    # frames = (data/1000).reshape(-1, 8, 8)

    #data = pd.read_csv('./raw_data/raw.csv', usecols=range(1, 65)).to_numpy()
    #frames = (data/1000).reshape(-1, 8, 8)

    # data = pd.read_csv('../exp_data/std_TOFEXP1.csv', usecols=range(0, 64)).to_numpy()
    # frames = (data).reshape(-1, 8, 8)
    #isHandled = "zScore" #"minMax"

    #check_one_frame(data[0], "./animation/std_TOFEXP4.jpg", isHandled)
    #animate_depth(frames, "./animation/std_TOFEXP4.mp4", isHandled)


    # labels = pd.read_csv('../exp_data/std_TOFEXP4.csv', usecols=range(64, 66)).to_numpy()
    # print(labels.shape)
    # recordmin = []
    # recordmax = []
    # for idx, row in enumerate(labels):
    #     if (row[0]< 0) or (row[1] < 0):
    #         recordmin.append((idx, row))
    #     if (row[0] > 3) or (row[1] > 3):
    #         recordmax.append((idx, row))
    #
    # print(recordmin)
    # print(recordmax)
    # animate_trajectory(labels, interval=30,)

    #compare_datasets("./raw_data/exp3_1h_ultrasound.csv", "./raw_data/exp4_1h_ultrasound.csv", jump_thresh=0.3)

