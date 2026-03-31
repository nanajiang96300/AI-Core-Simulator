# import matplotlib.pyplot as plt
# import numpy as np

# # 全局 IEEE 论文风格设置
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 12,
#     'axes.labelsize': 13,
#     'axes.titlesize': 14,
#     'legend.fontsize': 11,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 11
# })

# def plot_e2_f_sweep():
#     """生成第一张图: (a) Dual-Objective Fusion Search"""
#     # 设置更扁平的尺寸，适合上下堆叠
#     fig, ax = plt.subplots(figsize=(7, 4.2))
    
#     F_vals_64 = [1, 2, 4, 7, 8]
#     cy_64 = [13656, 7792, 5128, 4762, 4686]
    
#     F_vals_4 = [1, 2, 4, 7]
#     cy_4 = [4699, 2751, 1777, 1806]
    
#     # 绘制 64T64R
#     ax.plot(F_vals_64, cy_64, marker='o', color='#2c3e50', linewidth=2.5, markersize=8, label='64T64R ($N_t=64$)')
#     ax.scatter([7], [4762], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
#     ax.annotate('Optimal $F^*=7$', xy=(7, 4762), xytext=(7, 7000),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
#                  fontsize=12, fontweight='bold', ha='center')

#     # 绘制 4T4R
#     ax.plot(F_vals_4, cy_4, marker='s', color='#2980b9', linewidth=2.5, markersize=8, linestyle='--', label='4T4R ($N_t=4$)')
#     ax.scatter([4], [1777], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
#     ax.annotate('Optimal $F^*=4$', xy=(4, 1777), xytext=(4, 4000),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
#                  fontsize=12, fontweight='bold', ha='center')
    
#     ax.set_xlabel('Subcarrier Fusion Factor ($F$)')
#     ax.set_ylabel('Total Execution Cycles')
#     ax.set_title('(a) Dual-Objective Fusion Search')
#     ax.set_xticks(range(1, 9))
#     ax.grid(True, linestyle=':', alpha=0.7)
#     ax.legend(loc='upper right')
    
#     plt.tight_layout()
#     plt.savefig('fig_e2_f_sweep_refined.pdf', bbox_inches='tight')
#     plt.savefig('fig_e2_f_sweep_refined.png', dpi=300, bbox_inches='tight')
#     print("Saved fig_e2_f_sweep_refined")

# def plot_e2_nt_sweep():
#     """生成第二张图: (b) Algorithm 1 Performance Gain"""
#     fig, ax = plt.subplots(figsize=(7, 4.2))
    
#     Nt_labels = ['4T4R', '8T8R', '16T32R', '32T32R', '64T64R']
#     naive_cy = np.array([4699, 5077, 5861, 7805, 13656])
#     opt_cy = np.array([1777, 1944, 2221, 2958, 4762])
    
#     x = np.arange(len(Nt_labels))
#     width = 0.35
    
#     ax.bar(x - width/2, naive_cy, width, label='Naïve Baseline ($F=1$)', color='#95a5a6', edgecolor='black')
#     ax.bar(x + width/2, opt_cy, width, label='Algorithm 1 ($F=F^*$)', color='#27ae60', edgecolor='black')
    
#     for i in range(len(Nt_labels)):
#         speedup = naive_cy[i] / opt_cy[i]
#         ax.text(x[i] + width/2, opt_cy[i] + 300, f'{speedup:.1f}×', ha='center', va='bottom', fontweight='bold', color='#27ae60')
        
#     ax.set_xlabel('Antenna Configuration')
#     ax.set_ylabel('Total Execution Cycles')
#     ax.set_title('(b) Algorithm 1 Performance Gain')
#     ax.set_xticks(x)
#     ax.set_xticklabels(Nt_labels)
#     ax.legend(loc='upper left')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linestyle=':', alpha=0.7)

#     plt.tight_layout()
#     plt.savefig('fig_e2_nt_sweep_refined.pdf', bbox_inches='tight')
#     plt.savefig('fig_e2_nt_sweep_refined.png', dpi=300, bbox_inches='tight')
#     print("Saved fig_e2_nt_sweep_refined")

# if __name__ == "__main__":
#     plot_e2_f_sweep()
#     plot_e2_nt_sweep()

import matplotlib.pyplot as plt

# 全局 IEEE 论文风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

def plot_e2_f_sweep_rich():
    """生成第一张图: (a) Dual-Objective Fusion Search (多组数据)"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # 提取 Table VIII 中所有维度的数据
    # 64T64R
    F_64 = [1, 2, 4, 7, 8]
    cy_64 = [13656, 7792, 5128, 4762, 4686]
    
    # 32T32R (提取表格中的关键点)
    F_32 = [1, 7]
    cy_32 = [7805, 2958]
    
    # 8T8R
    F_8 = [1, 4, 7]
    cy_8 = [5077, 1917, 1944]
    
    # 4T4R
    F_4 = [1, 2, 4, 7]
    cy_4 = [4699, 2751, 1777, 1806]
    
    # 绘制曲线 (使用不同颜色和线型拉开层次)
    ax.plot(F_64, cy_64, marker='o', color='#2c3e50', linewidth=2.5, markersize=7, label='64T64R')
    ax.plot(F_32, cy_32, marker='^', color='#8e44ad', linewidth=2.0, markersize=7, linestyle='--', label='32T32R')
    ax.plot(F_8, cy_8, marker='d', color='#27ae60', linewidth=2.0, markersize=7, linestyle='-.', label='8T8R')
    ax.plot(F_4, cy_4, marker='s', color='#2980b9', linewidth=2.5, markersize=7, label='4T4R')
    
    # === 统一标记所有曲线的最优解 F* ===
    # 64T64R 的最优解是 F=7
    ax.scatter([7], [4762], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
    # 32T32R 的最优解是 F=7
    ax.scatter([7], [2958], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
    # 8T8R 的最优解是 F=7
    ax.scatter([7], [1944], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
    # 4T4R 的最优解是 F=4
    ax.scatter([4], [1777], color='#f1c40f', edgecolor='black', s=250, marker='*', zorder=5)
    
    # 添加一个假的散点用于在图例中显示“最优解”标志
    ax.scatter([], [], color='#f1c40f', edgecolor='black', s=200, marker='*', label='Optimal $F^*$ (Alg. 1)')

    # 添加文字高亮提示 (只标注最高和最低的，避免画面拥挤)
    ax.annotate('Algorithm 1 Opt.', xy=(7, 4762), xytext=(7, 6500),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
                 
    ax.annotate('Optimal $F^*=4$', xy=(4, 1777), xytext=(4, 3500),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=12, fontweight='bold', ha='center', color='#2980b9')

    ax.set_xlabel('Subcarrier Fusion Factor ($F$)')
    ax.set_ylabel('Total Execution Cycles')
    ax.set_title('(a) Dual-Objective Fusion Search')
    ax.set_xticks(range(1, 9))
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 图例排版优化
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig_e2_f_sweep_refined.pdf', bbox_inches='tight')
    plt.savefig('fig_e2_f_sweep_refined.png', dpi=300, bbox_inches='tight')
    print("Saved multi-line fig_e2_f_sweep_refined.pdf")

if __name__ == "__main__":
    plot_e2_f_sweep_rich()