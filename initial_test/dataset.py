from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# 以当前文件所在目录为基准，兼容从 initial_test/ 跨目录读取项目根目录数据
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def _resolve_path(path_like: Union[str, Path]) -> Path:
	"""将输入路径解析为绝对路径，优先兼容项目根目录下的数据文件。"""
	path = Path(path_like)
	if path.is_absolute():
		return path

	candidate_paths = [
		Path.cwd() / path,
		CURRENT_DIR / path,
		PROJECT_ROOT / path,
	]
	for candidate in candidate_paths:
		if candidate.exists():
			return candidate.resolve()

	return (PROJECT_ROOT / path).resolve()


class CIRDataset(Dataset):
	"""用于声学感知 CIR 数据的自定义数据集。"""

	def __init__(self, samples: Union[pd.DataFrame, Sequence[Tuple[Union[str, Path], int]]]):
		"""
		参数:
			samples: 包含文件路径和 Subject 标签的 DataFrame 或列表。
					 DataFrame 需要至少包含 file_path 和 Subject 两列。
		"""
		if isinstance(samples, pd.DataFrame):
			required_columns = {"file_path", "Subject"}
			missing_columns = required_columns - set(samples.columns)
			if missing_columns:
				raise ValueError(f"samples 缺少必要列: {sorted(missing_columns)}")
			records_df = samples.loc[:, ["file_path", "Subject"]].copy()
		else:
			records_df = pd.DataFrame(samples, columns=["file_path", "Subject"])

		if records_df.empty:
			raise ValueError("samples 不能为空")

		self.samples: List[Tuple[Path, int]] = []
		for row in records_df.itertuples(index=False):
			file_path = _resolve_path(row.file_path)
			self.samples.append((file_path, int(row.Subject)))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		# 惰性加载单个样本，避免一次性将全部数据读入内存
		file_path, subject = self.samples[index]

		matrix = np.load(file_path)
		if matrix.shape != (960, 150):
			raise ValueError(f"样本形状异常: {file_path} -> {matrix.shape}，期望 (960, 150)")

		# 增加通道维度，转换为 2D-CNN 需要的 (1, 960, 150)
		data_tensor = torch.from_numpy(matrix).float().unsqueeze(0)

		# Subject 原始标签为 1, 2, 3，这里减 1 以适配 CrossEntropyLoss
		label_tensor = torch.tensor(subject - 1, dtype=torch.long)

		return data_tensor, label_tensor


def get_dataloaders(
	csv_path: Union[str, Path],
	data_dir: Union[str, Path],
	batch_size: int,
	test_split: float = 0.2,
):
	"""构建按 Subject 分层抽样的训练集与测试集 DataLoader。"""
	csv_file = _resolve_path(csv_path)
	data_folder = _resolve_path(data_dir)

	labels_df = pd.read_csv(csv_file)
	required_columns = {"GroupID", "Subject"}
	missing_columns = required_columns - set(labels_df.columns)
	if missing_columns:
		raise ValueError(f"CSV 缺少必要列: {sorted(missing_columns)}")

	samples_df = labels_df.loc[:, ["GroupID", "Subject"]].copy()
	samples_df["GroupID"] = samples_df["GroupID"].astype(str)
	samples_df["file_path"] = samples_df["GroupID"].apply(lambda group_id: data_folder / f"{group_id}.npy")

	missing_files = [str(path) for path in samples_df["file_path"] if not Path(path).exists()]
	if missing_files:
		preview = ", ".join(missing_files[:5])
		raise FileNotFoundError(f"存在缺失的数据文件，示例: {preview}")

	train_df, test_df = train_test_split(
		samples_df.loc[:, ["file_path", "Subject"]],
		test_size=test_split,
		random_state=42,
		stratify=samples_df["Subject"],
		shuffle=True,
	)

	train_dataset = CIRDataset(train_df.reset_index(drop=True))
	test_dataset = CIRDataset(test_df.reset_index(drop=True))

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader
