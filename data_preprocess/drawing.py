import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
def animate_depth(frames, isHandled=False):
    fig, ax = plt.subplots()
    if isHandled:
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=-3, vmax=3)
    else:
        im = ax.imshow(frames[0], cmap="jet", animated=True, vmin=0, vmax=3)

    plt.colorbar(im, ax=ax, label="Depth (m)")
    frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    def update(frame_index):
        im.set_array(frames[frame_index])
        frame_text.set_text(f'Frame: {frame_index + 1}')
        return [im, frame_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True, repeat=False)
    plt.show()
    return ani

def check_one_frame(frame, isHandled=False):
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

    plt.show()


def normalize_ground_truth(coords, room_size=(3.0, 3.0)):
    return coords / np.array(room_size)

def animate_trajectory(coords, interval=50, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
    ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
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

    plt.show()
    return ani



if __name__ == "__main__":
    # data = pd.read_csv('./clean_data/norm_tof1_.csv', usecols=range(1, 65)).to_numpy()
    # frames = data.reshape(-1, 8, 8)
    # print(data.shape)

    # data = pd.read_csv('./raw_data/features.csv', usecols=range(1, 65)).to_numpy()
    # frames = (data/1000).reshape(-1, 8, 8)

    # data = pd.read_csv('./raw_data/raw.csv', usecols=range(1, 65)).to_numpy()
    # frames = (data/1000).reshape(-1, 8, 8)

    # data = pd.read_csv('../exp_data/std_TOFEXP1.csv', usecols=range(0, 64)).to_numpy()
    # frames = (data).reshape(-1, 8, 8)
    # isHandled = True
    #
    # check_one_frame(data[0], isHandled)
    # animate_depth(frames, isHandled)

    labels = pd.read_csv('../exp_data/std_TOFEXP4.csv', usecols=range(64, 66)).to_numpy()
    animate_trajectory(labels, interval=50)
