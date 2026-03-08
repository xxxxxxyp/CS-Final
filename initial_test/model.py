import torch
import torch.nn as nn


class LightweightCIRCNN(nn.Module):
	"""面向小样本声学感知任务的轻量级 2D-CNN。"""

	def __init__(self, num_classes: int = 3, dropout_prob: float = 0.5):
		super().__init__()

		self.features = nn.Sequential(
			self._make_conv_block(1, 8),
			self._make_conv_block(8, 16),
			self._make_conv_block(16, 32),
		)

		# 进一步压缩空间尺寸，控制全连接层参数规模，降低过拟合风险
		self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
		feature_dim = self._infer_feature_dim()

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(feature_dim, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout_prob),
			nn.Linear(64, num_classes),
		)

	@staticmethod
	def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
		"""构建单个卷积块：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d。"""
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
		)

	def _infer_feature_dim(self) -> int:
		"""使用固定输入尺寸推断展平后的特征维度。"""
		with torch.no_grad():
			dummy_input = torch.zeros(1, 1, 960, 150)
			output = self.avg_pool(self.features(dummy_input))
		return output.numel()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 输入形状期望为 (Batch_Size, 1, 960, 150)
		x = self.features(x)
		x = self.avg_pool(x)
		logits = self.classifier(x)
		return logits


if __name__ == '__main__':
	model = LightweightCIRCNN()
	dummy_input = torch.randn(2, 1, 960, 150)
	output = model(dummy_input)

	total_params = sum(parameter.numel() for parameter in model.parameters())

	print(f"输出张量形状: {output.shape}")
	print(f"模型总参数量: {total_params:,}")
