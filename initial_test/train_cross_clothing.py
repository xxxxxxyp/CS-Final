from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import CIRDataset
from model import LightweightCIRCNN


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

CSV_PATH = PROJECT_ROOT / "dataset_labels.csv"
DATA_DIR = PROJECT_ROOT / "Processed_Data"
BEST_MODEL_PATH = CURRENT_DIR / "best_model_cross_clothing.pth"

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 3


def get_device() -> torch.device:
	"""自动选择当前可用的最佳计算设备。"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
		return torch.device("mps")
	return torch.device("cpu")


def train_one_epoch(
	model: nn.Module,
	data_loader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> float:
	"""执行一个完整的训练轮次，并返回平均训练损失。"""
	model.train()
	total_loss = 0.0
	total_samples = 0

	for inputs, targets in data_loader:
		inputs = inputs.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()
		logits = model(inputs)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		batch_size = inputs.size(0)
		total_loss += loss.item() * batch_size
		total_samples += batch_size

	return total_loss / total_samples


def evaluate(
	model: nn.Module,
	data_loader,
	criterion: nn.Module,
	device: torch.device,
) -> Tuple[float, float]:
	"""在验证集/测试集上评估平均损失与准确率。"""
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for inputs, targets in data_loader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			logits = model(inputs)
			loss = criterion(logits, targets)

			predictions = logits.argmax(dim=1)
			batch_size = inputs.size(0)

			total_loss += loss.item() * batch_size
			total_correct += (predictions == targets).sum().item()
			total_samples += batch_size

	avg_loss = total_loss / total_samples
	accuracy = 100.0 * total_correct / total_samples
	return avg_loss, accuracy


def main() -> None:
	device = get_device()
	print(f"Using device: {device}")

	labels_df = pd.read_csv(CSV_PATH)
	required_columns = {"GroupID", "Subject", "Clothing"}
	missing_columns = required_columns - set(labels_df.columns)
	if missing_columns:
		raise ValueError(f"CSV 缺少必要列: {sorted(missing_columns)}")

	# 严格执行跨衣物泛化测试：Clothing==1/2 用于训练，Clothing==3 用于测试
	samples_df = labels_df.loc[:, ["GroupID", "Subject", "Clothing"]].copy()
	samples_df["GroupID"] = samples_df["GroupID"].astype(str)
	samples_df["file_path"] = samples_df["GroupID"].apply(lambda group_id: DATA_DIR / f"{group_id}.npy")

	train_df = samples_df[samples_df["Clothing"].isin([1, 2])].loc[:, ["file_path", "Subject"]]
	test_df = samples_df[samples_df["Clothing"] == 3].loc[:, ["file_path", "Subject"]]

	if train_df.empty:
		raise ValueError("训练集为空，请检查 Clothing == 1/2 的数据是否存在。")
	if test_df.empty:
		raise ValueError("测试集为空，请检查 Clothing == 3 的数据是否存在。")

	missing_files = [str(path) for path in samples_df["file_path"] if not Path(path).exists()]
	if missing_files:
		preview = ", ".join(missing_files[:5])
		raise FileNotFoundError(f"存在缺失的数据文件，示例: {preview}")

	train_dataset = CIRDataset(train_df.reset_index(drop=True))
	test_dataset = CIRDataset(test_df.reset_index(drop=True))

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	model = LightweightCIRCNN(num_classes=NUM_CLASSES).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	best_val_acc = -1.0

	for epoch in range(1, EPOCHS + 1):
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, test_loader, criterion, device)

		print(
			f"Epoch [{epoch}/{EPOCHS}] | "
			f"Train Loss: {train_loss:.3f} | "
			f"Val Loss: {val_loss:.3f} | "
			f"Val Acc: {val_acc:.2f}%"
		)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), BEST_MODEL_PATH)
			print("--> Best model saved!")


if __name__ == "__main__":
	main()
