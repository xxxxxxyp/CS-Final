from pathlib import Path

import numpy as np
import torch

from model import LightweightCIRCNN


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODEL_PATH = CURRENT_DIR / 'best_model.pth'
DATA_DIR = PROJECT_ROOT / 'processed_test_data'

NUM_CLASSES = 3
EXPECTED_SHAPE = (960, 150)
TRUE_LABEL = 0


def get_device() -> torch.device:
	"""自动选择当前可用的最佳计算设备。"""
	if torch.cuda.is_available():
		return torch.device('cuda')
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
		return torch.device('mps')
	return torch.device('cpu')


def load_model(device: torch.device) -> LightweightCIRCNN:
	"""加载训练好的最优基线模型，并切换到推理模式。"""
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f'找不到模型权重文件: {MODEL_PATH}')

	model = LightweightCIRCNN(num_classes=NUM_CLASSES).to(device)
	checkpoint = torch.load(MODEL_PATH, map_location=device)

	if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
		state_dict = checkpoint['state_dict']
	else:
		state_dict = checkpoint

	model.load_state_dict(state_dict)
	model.eval()
	return model


def load_input_tensor(npy_path: Path, device: torch.device) -> torch.Tensor:
	"""读取单个 .npy 样本，并扩展为模型需要的 4 维输入。"""
	sample = np.load(npy_path)
	if sample.shape != EXPECTED_SHAPE:
		raise ValueError(
			f'{npy_path} 的形状应为 {EXPECTED_SHAPE}，实际为 {sample.shape}。'
		)

	return torch.from_numpy(sample).float().unsqueeze(0).unsqueeze(0).to(device)


def main() -> None:
	device = get_device()
	print(f'使用设备: {device}')

	if not DATA_DIR.is_dir():
		raise FileNotFoundError(f'找不到数据目录: {DATA_DIR}')

	npy_files = sorted(DATA_DIR.glob('*.npy'), key=lambda file_path: file_path.name)
	if not npy_files:
		raise FileNotFoundError(f'{DATA_DIR} 中没有找到 .npy 文件。')

	model = load_model(device)

	# 已知这批新测试数据全部来自受试者 1，在模型中的正确标签为 0。
	total_samples = 0
	prediction_counts = {class_index: 0 for class_index in range(NUM_CLASSES)}
	correct_predictions = 0

	with torch.inference_mode():
		for npy_file in npy_files:
			input_tensor = load_input_tensor(npy_file, device)
			logits = model(input_tensor)
			predicted_label = logits.argmax(dim=1).item()

			total_samples += 1
			prediction_counts[predicted_label] += 1
			if predicted_label == TRUE_LABEL:
				correct_predictions += 1

	accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

	print(f'测试总样本数: {total_samples}')
	print(f'预测为 0 的数量: {prediction_counts[0]}')
	print(f'预测为 1 的数量: {prediction_counts[1]}')
	print(f'预测为 2 的数量: {prediction_counts[2]}')
	print(f'识别为受试者 0 的准确率: {accuracy:.2%}')


if __name__ == '__main__':
	main()