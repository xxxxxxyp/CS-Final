import os
import numpy as np
import scipy.io as sio

# --- 配置参数 ---
base_dir = './Dataset-process/Raw_Data'  # 你的主文件夹路径
window_size = 150        # 每 150 帧作为一个数据点
labels = [1, 2, 3]       # 受试人标签

X_list = []  # 用来存放特征数据
y_list = []  # 用来存放对应的标签

print("开始构建数据集...")

# --- 遍历文件夹与文件 ---
for label in labels:
    folder_path = os.path.join(base_dir, f'subject_{label}')
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"警告: 找不到文件夹 {folder_path}，已跳过。")
        continue

    # 遍历该受试人文件夹下的所有 .mat 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            filepath = os.path.join(folder_path, filename)
            
            # 1. 读取 .mat 文件
            mat_data = sio.loadmat(filepath)
            # 提取保存的变量（确保这里填入的是你在 MATLAB 里保存的变量名）
            cir_data = mat_data['all_cir_data']  # 形状通常为 (zclen, total_frames)
            
            total_frames = cir_data.shape[1]
            
            # 2. 计算可以切出多少个完整的数据块
            num_samples = total_frames // window_size
            
            # 3. 按 150 帧进行切分
            for i in range(num_samples):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                # 取出这段数据作为一个样本，形状为 (zclen, 150)
                sample = cir_data[:, start_idx:end_idx]
                
                X_list.append(sample)
                y_list.append(label)
                
    print(f"受试人 {label} 的数据处理完毕。")

# --- 转换为 Numpy 数组并保存 ---
# X 的最终形状将会是 (总样本数, zclen, 150)
X = np.array(X_list)
# y 的最终形状将会是 (总样本数,)
y = np.array(y_list)

print(f"\n数据集构建成功！")
print(f"特征矩阵 X 形状: {X.shape}")
print(f"标签矩阵 y 形状: {y.shape}")

# 保存为高度压缩的 .npz 格式，方便后续用 PyTorch/TensorFlow 加载
save_path = 'cir_dataset.npz'
np.savez_compressed(save_path, X=X, y=y)
print(f"数据集已保存至: {save_path}")