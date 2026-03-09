from pathlib import Path

import numpy as np
from scipy.io import loadmat


WINDOW_SIZE = 150
EXPECTED_ROWS = 960
DATA_KEY = 'all_cir_data'


def split_samples(cir_data, window_size=WINDOW_SIZE):
    total_frames = cir_data.shape[1]
    num_samples = total_frames // window_size

    for sample_index in range(num_samples):
        start_idx = sample_index * window_size
        end_idx = start_idx + window_size
        yield cir_data[:, start_idx:end_idx]


def process_new_test_data(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f'找不到输入目录: {input_dir}')

    mat_files = sorted(input_dir.glob('*.mat'), key=lambda file_path: file_path.name)
    if not mat_files:
        raise FileNotFoundError(f'{input_dir} 中没有找到 .mat 文件。')

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_counter = 0

    for mat_file in mat_files:
        mat_data = loadmat(mat_file)
        if DATA_KEY not in mat_data:
            raise KeyError(f'{mat_file} 中缺少变量 {DATA_KEY}')

        cir_data = np.asarray(mat_data[DATA_KEY])
        if cir_data.ndim != 2:
            raise ValueError(f'{mat_file} 中的 {DATA_KEY} 必须是二维数组。')
        if cir_data.shape[0] != EXPECTED_ROWS:
            raise ValueError(
                f'{mat_file} 中的 {DATA_KEY} 第一维应为 {EXPECTED_ROWS}，实际为 {cir_data.shape[0]}。'
            )

        saved_count = 0
        for sample in split_samples(cir_data):
            output_path = output_dir / f'new_sample_{sample_counter:03d}.npy'
            np.save(output_path, sample)
            sample_counter += 1
            saved_count += 1

        print(f'{mat_file.name} 处理完成，保存 {saved_count} 个样本。')

    print(f'处理完成，共保存 {sample_counter} 个样本到: {output_dir}')


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    process_new_test_data(
        input_dir="C:\Projects\CS-Final\Dataset-process\Raw_Data\\test_data",
        output_dir="C:\Projects\CS-Final\processed_test_data",
    )