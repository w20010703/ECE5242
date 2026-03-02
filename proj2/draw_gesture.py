import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_imu_data(file_paths):
    # 7 columns per project spec: Time, Wx, Wy, Wz, Ax, Ay, Az
    columns = ['Time', 'Wx', 'Wy', 'Wz', 'Ax', 'Ay', 'Az']

    num_files = len(file_paths)
    fig, axes = plt.subplots(num_files, 1, figsize=(12, 5 * num_files))

    if num_files == 1:
        axes = [axes]

    for i, file_path in enumerate(file_paths):
        # Data is tab/whitespace-separated (no commas); use sep=r'\s+' for pandas 2.x
        df = pd.read_csv(
            file_path,
            header=None,
            sep=r'\s+',
            engine='python',
            skip_blank_lines=True,
        )
        if df.shape[1] != 7:
            raise ValueError(f"Expected 7 columns in {file_path}, got {df.shape[1]}")

        df.columns = columns

        for col in columns[1:]:
            axes[i].plot(df['Time'], df[col], label=col)

        axes[i].set_title(f'Gesture Waveform: {os.path.basename(file_path)}')
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('Sensor Value')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# Plot files: set folder_path and optionally limit to specific files
folder_path = 'ECE5242Proj2-train'
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_abs = os.path.join(script_dir, folder_path)
files_to_plot = []
for f in sorted(os.listdir(folder_abs)):
    if f.endswith('.txt'):
        files_to_plot.append(os.path.join(folder_abs, f))
if not files_to_plot:
    raise SystemExit(f"No .txt files found in {folder_abs}")
plot_imu_data(files_to_plot)
