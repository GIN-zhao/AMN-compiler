import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# --- 全局绘图参数设置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 0.5

# --- 生成图 (SEFI 细分类型前后对比 - 横向版本) ---

# 1. 读取并准备数据
csv_file_path = 'fig5_data.csv'
try:
    df_breakdown = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"错误：无法找到 '{csv_file_path}' 文件。请确保它和脚本在同一个目录下。")
    exit()

if 'SEFI_Type' not in df_breakdown.columns:
    print(f"错误: CSV文件中未找到 'SEFI_Type' 列。实际列名为: {df_breakdown.columns.tolist()}")
    exit()

# 计算百分比并确保 Count 是整数类型
df_breakdown['Percentage'] = 100 * df_breakdown['Count'] / df_breakdown['Total_Experiments']
df_breakdown['Count'] = df_breakdown['Count'].astype(int)

# --- 关键调试步骤：确认数据 ---
print("--- 将用于绘图的 SEFI 细分数据 ---")
print(df_breakdown.to_string(index=False))
print("------------------------------------")

# 2. 定义绘图顺序、标签和颜色
sefi_type_order = ['Memory', 'Scheduling', 'Stack', 'Freeze']
ytick_labels_map = {
    'Memory': 'Invalid Memory Address',
    'Scheduling': 'Kernel Scheduling & Synchronization Exceptions',
    'Stack': 'Kernel Stack Space Exceptions',
    'Freeze': 'Kernel Freeze'
}
condition_order = ['Before', 'After']
color_palette = {
    'Before': '#d54949', # 深红
    'After': '#e88c8c'   # 浅红
}

# 3. 创建图表
fig, ax = plt.subplots(figsize=(7, 4.5))

# 绘制横向分组柱状图
sns.barplot(data=df_breakdown, x='Percentage', y='SEFI_Type', hue='Condition', ax=ax,
            order=sefi_type_order,
            hue_order=condition_order,
            palette=color_palette,
            orient='h')

# 4. 调整样式和标签
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 为每个柱子创建自定义的复合标签 "百分比% (数值)"
for i, container in enumerate(ax.containers):
    current_condition = condition_order[i]
    df_subset = df_breakdown[df_breakdown['Condition'] == current_condition].set_index('SEFI_Type')
    df_ordered = df_subset.reindex(sefi_type_order)
    
    labels = [
        f"{row['Percentage']:.1f}% ({row['Count']})" 
        if pd.notna(row['Percentage']) and row['Percentage'] > 0 
        else ''
        for index, row in df_ordered.iterrows()
    ]
    
    ax.bar_label(container, labels=labels, fontsize=8, padding=3)

# 动态调整X轴上限，为更长的标签留出更多空间
max_percentage_val = df_breakdown['Percentage'].max() if not df_breakdown['Percentage'].empty else 0
ax.set_xlim(0, max_percentage_val * 1.35) 

# 设置坐标轴标签和Y轴刻度标签
ax.set_xlabel('Occurrence Rate (%)')
ax.set_ylabel('')

WRAP_WIDTH = 25
wrapped_labels = [textwrap.fill(ytick_labels_map[t], width=WRAP_WIDTH) for t in sefi_type_order]
ax.set_yticklabels(wrapped_labels)

# 5. 调整图例
legend = ax.get_legend()
legend.set_title('Condition')
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('none')
plt.setp(legend.get_title(), fontweight='bold')

# 6. 显示和保存
plt.tight_layout()

# 保存图像 (在显示之前！)
eps_output_path = 'Fig5_2.eps'
pdf_output_path = 'Fig5.pdf' # 增加PDF保存，方便排查

try:
    fig.savefig(eps_output_path, format='eps', bbox_inches='tight', dpi=300)
    print(f"'{eps_output_path}' 已成功保存。")
except Exception as e:
    print(f"保存 EPS 文件时发生错误: {e}")
    print("错误提示：")
    print(e) # 打印详细的错误信息
    print("建议：")
    print("1. 检查您的系统是否安装了 Ghostscript，并且其可执行文件是否在系统PATH中。")
    print("   Ghostscript 官网: https://www.ghostscript.com/download/gsdnld.html")
    print("2. 尝试将图表保存为 PDF 格式，看是否可以成功。")

# 尝试保存为PDF，如果EPS失败，PDF通常更稳妥
# try:
#     fig.savefig(pdf_output_path, format='pdf', bbox_inches='tight', dpi=300)
#     print(f"'{pdf_output_path}' 已成功保存。")
# except Exception as e:
#     print(f"保存 PDF 文件时发生错误: {e}")
#     print("错误提示：")
#     print(e)
#     print("建议：检查是否有写入目标路径的权限。")

# 显示图表 (在保存之后！)
# plt.show()

# 关闭图表，释放内存
# plt.close(fig)
