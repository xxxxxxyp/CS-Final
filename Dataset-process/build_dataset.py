import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# --- 配置参数 ---
WINDOW_SIZE = 150
DATA_KEY = 'all_cir_data'
SUBJECT_IDS = (1, 2, 3)
EXPECTED_FILES_PER_SUBJECT = 6


def split_samples(cir_data, window_size=WINDOW_SIZE):
    """沿列方向按固定帧数切分数据，丢弃不足一个窗口的尾部数据。"""
    total_frames = cir_data.shape[1]
    num_samples = total_frames // window_size

    if num_samples > 100:
        raise ValueError('单个条件下的切片数量超过 100，无法生成 5 位 GroupID。')

    for sample_index in range(num_samples):
        start_idx = sample_index * window_size
        end_idx = start_idx + window_size
        yield sample_index, cir_data[:, start_idx:end_idx]


def process_dataset(base_dir, output_dir, labels_csv_path):
    """读取 Raw_Data 中的 .mat 文件，切分后保存为 .npy，并生成标签 CSV。"""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    labels_csv_path = Path(labels_csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_rows = []

    print('开始处理数据集...')

    for subject_id in SUBJECT_IDS:
        subject_dir = base_dir / f'subject_{subject_id}'

        if not subject_dir.is_dir():
            raise FileNotFoundError(f'找不到受试人目录: {subject_dir}')

        # 按文件名字母顺序固定读取，确保第 1~6 个文件的映射关系稳定。
        mat_files = sorted(subject_dir.glob('*.mat'), key=lambda file_path: file_path.name)

        if len(mat_files) != EXPECTED_FILES_PER_SUBJECT:
            raise ValueError(
                f'{subject_dir} 中应当恰好包含 {EXPECTED_FILES_PER_SUBJECT} 个 .mat 文件，'
                f'当前找到 {len(mat_files)} 个。'
            )

        for file_index, mat_file in enumerate(mat_files):
            clothing_id = (file_index // 2) + 1
            pose_id = file_index % 2

            mat_data = loadmat(mat_file)
            if DATA_KEY not in mat_data:
                raise KeyError(f'{mat_file} 中缺少变量 {DATA_KEY}')

            cir_data = np.asarray(mat_data[DATA_KEY])
            if cir_data.ndim != 2:
                raise ValueError(f'{mat_file} 中的 {DATA_KEY} 必须是二维数组。')

            saved_count = 0
            for sample_index, sample in split_samples(cir_data):
                group_id = f'{subject_id}{clothing_id}{pose_id}{sample_index:02d}'
                np.save(output_dir / f'{group_id}.npy', sample)
                label_rows.append([group_id, subject_id, clothing_id, pose_id])
                saved_count += 1

            print(
                f'受试人 {subject_id} - 衣服 {clothing_id} - 姿态 {pose_id} '
                f'处理完成，共保存 {saved_count} 个样本。'
            )

    with labels_csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GroupID', 'Subject', 'Clothing', 'Pose'])
        writer.writerows(label_rows)

    print(f'数据处理完成，共生成 {len(label_rows)} 个样本。')
    print(f'.npy 文件已保存至: {output_dir}')
    print(f'标签文件已保存至: {labels_csv_path}')


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    process_dataset(
        base_dir=project_root / 'Raw_Data',
        output_dir=project_root / 'Processed_Data',
        labels_csv_path=project_root / 'dataset_labels.csv',
    )
