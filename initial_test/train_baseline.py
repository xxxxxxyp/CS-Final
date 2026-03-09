from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import get_dataloaders
from model import LightweightCIRCNN


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

CSV_PATH = PROJECT_ROOT / "dataset_labels_diff.csv"
DATA_DIR = PROJECT_ROOT / "Processed_Diff_Data"
BEST_MODEL_PATH = CURRENT_DIR / "best_model_diff.pth"

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

	train_loader, test_loader = get_dataloaders(
		csv_path=CSV_PATH,
		data_dir=DATA_DIR,
		batch_size=BATCH_SIZE,
	)

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
