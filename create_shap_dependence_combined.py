import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# 设置字体（使用 Times New Roman）
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 特征名称映射（中文到英文）
feature_names = {
    '功率': 'Power',
    '扫描速度': 'Scan Speed', 
    '扫描间隙': 'Scan Spacing',
    '层厚': 'Layer Thickness'
}

# 定义模型和目标变量
models = ['LightGBM', 'RandomForest', 'XGBoost']
targets = ['致密度', '阻抗(MΩ cm-2)', 'Jcorr(μA cm-2)']
target_names = ['Relative Density (%)', 'Impedance (MΩ·cm⁻²)', 'Jcorr (μA/cm⁻²)']

# 特征列表（中文，用于文件名匹配）
features = ['功率', '扫描速度', '扫描间隙', '层厚']
features_en = ['Power', 'Scan Speed', 'Scan Spacing', 'Layer Thickness']

# 为每个模型创建依赖图拼接
for model in models:
    # 创建图形，4 行 3 列（4 个特征 × 3 个目标）
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'{model} - SHAP Dependence Plots', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            
            # 构建文件名
            file_name = f'{target}_SHAP_dep_{feature}.png'
            file_path = f'model_plots_no_density/shap/{model}/{file_name}'
            
            if os.path.exists(file_path):
                # 读取图片
                img = mpimg.imread(file_path)
                ax.imshow(img)
                
                # 设置标题（英文）
                if i == 0:  # 第一行显示目标变量
                    ax.set_title(target_names[j], fontsize=12, fontweight='bold')
                
                # 设置特征名（英文）作为行标签
                if j == 0:  # 第一列显示特征名
                    ax.set_ylabel(f'{features_en[i]}', fontsize=12, fontweight='bold', rotation=90, fontname='Times New Roman')
                
            else:
                ax.text(0.5, 0.5, f'File not found:\n{file_name}', 
                       transform=ax.transAxes, ha='center', va='center')
            
            ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.08)
    
    # 保存图像
    output_name = f'SHAP_dependence_{model}_combined.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    
    plt.close()

print("\nAll SHAP dependence plots have been combined!")
