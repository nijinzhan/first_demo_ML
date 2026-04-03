import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体（使用 Times New Roman）
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 定义模型和目标变量（使用中文文件名和英文显示名称）
models = ['LightGBM', 'RandomForest', 'XGBoost']
targets = ['致密度', '阻抗(MΩ cm-2)', 'Jcorr(μA cm-2)']  # 用于文件路径
target_names = ['Relative Density (%)', 'Impedance (MΩ·cm⁻²)', 'Jcorr (μA/cm⁻²)']  # 用于显示

# 创建图形，3行3列
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('SHAP Grouped Bar Plots', fontsize=16, fontweight='bold')

# 特征名称映射（中文到英文）
feature_names = {
    '功率': 'Power',
    '扫描速度': 'Scan Speed', 
    '扫描间隙': 'Scan Spacing',
    '层厚': 'Layer Thickness'
}

# 遍历每个模型（行）和目标变量（列）
for i, model in enumerate(models):
    for j, target in enumerate(targets):
        ax = axes[i, j]
        
        # 读取 CSV 文件（使用中文文件名）
        file_path = f'model_plots_no_density/shap/{model}/{target}_SHAP_values.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # 计算每个特征的平均绝对 SHAP 值并排序（从上到下递减）
            shap_cols = [col for col in df.columns if col.startswith('SHAP_')]
            feature_cols = [col.replace('SHAP_', '') for col in shap_cols]
                        
            mean_abs_shap = []
            for shap_col in shap_cols:
                mean_abs_shap.append(np.mean(np.abs(df[shap_col])))
                        
            # 按 SHAP 值升序排序（这样重要性最高的在最上方）
            sorted_indices = np.argsort(mean_abs_shap)
            mean_abs_shap = [mean_abs_shap[idx] for idx in sorted_indices]
            feature_cols = [feature_cols[idx] for idx in sorted_indices]
                        
            # 创建特征名称列表（英文）
            features_en = [feature_names.get(feat, feat) for feat in feature_cols]
            
            # 创建水平条形图
            y_pos = np.arange(len(features_en))
            ax.barh(y_pos, mean_abs_shap, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # 设置标题（模型 - 目标变量），全部加粗（使用英文显示名称）
            ax.set_title(f'{model} - {target_names[j]}', fontsize=12, fontweight='bold')
                        
            # 设置 Y 轴标签（只在第一列显示）
            if j == 0:  # 第一列
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features_en, fontsize=10, fontweight='bold')
            else:  # 其他列不显示 Y 轴标签
                ax.set_yticks(y_pos)
                ax.set_yticklabels([''] * len(y_pos))
            
            # 设置 X 轴
            ax.set_xlabel('|SHAP value|' if i == 2 else '', fontsize=10)
            
        else:
            ax.text(0.5, 0.5, f'{model}\n{target}\nFile not found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 网格（移到循环内，每个子图都显示）
        ax.grid(axis='x', alpha=0.3)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93, left=0.08)  # 减小左侧空间，因为不需要显示模型名称

# 保存图像
plt.savefig('SHAP_grouped_bar_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('SHAP_grouped_bar_plots.pdf', bbox_inches='tight')

print("SHAP grouped bar plots have been generated and saved as 'SHAP_grouped_bar_plots.png' and 'SHAP_grouped_bar_plots.pdf'")

# 显示图像
plt.show()