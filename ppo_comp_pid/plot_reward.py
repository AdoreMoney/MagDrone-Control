import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# 1. 设置全局样式
plt.style.use(['science', 'grid', 'no-latex'])
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.facecolor': 'white',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'lines.linewidth': 1.5
})

# 2. 读取数据
try:
    data = pd.read_csv('reward.csv')
    steps = data['Step'].values
    values = data['Value'].values
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 3. 生成置信区间（使用滑动窗口模拟）
window_size = 50  # 滑动窗口大小

# 计算滑动标准差（处理初始窗口）
std_values = np.zeros_like(values)
for i in range(len(values)):
    start = max(0, i - window_size)
    std_values[i] = np.std(values[start:i+1])
    
# 对标准差进行后处理
std_values[:window_size] = std_values[window_size]  # 用第一个完整窗口值填充初始段
conf_interval = 0.45 * std_values  # 置信区间范围系数

# 4. 创建图表
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

# 5. 绘制置信区间带
ax.fill_between(steps, 
                values - conf_interval,
                values + conf_interval,
                color='#1f77b4', 
                alpha=0.15,  # 更浅的透明度
                linewidth=0)  # 无边界线

# 6. 绘制主曲线
ax.plot(steps, values,
       color='#1f77b4',
       label='Reward',
       linestyle='-',
       alpha=0.9)

# 7. 图表装饰
ax.set_xlabel('Timesteps', fontweight='bold')
ax.set_ylabel('Reward', fontweight='bold')
ax.set_xlim(left=0)
ax.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True, loc='lower right')

# 8. 保存和显示
plt.tight_layout()
plt.savefig('22_Mean training reward over 300,000-step PPO optimization.png', bbox_inches='tight', dpi=600)
plt.show()