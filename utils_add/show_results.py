#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/19 13:43
# @Author  : zhuzhaowen
# @File    : show_results.py
# @Software: PyCharm
# @desc    : "Optimized version to connect model series with dashed lines and place legend in the lower-right corner."

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_latency_vs_accuracy(data, dataset_name='KITTI'):
    """
    单独绘制准确度 vs 延迟图，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有ms/img数据的模型
    filtered_data = {}
    for model_name, metrics in data.items():
        if metrics.get('ms/img') is not None and dataset_name in metrics:
            filtered_data[model_name] = metrics

    # 分组模型
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X']
    cmap = plt.get_cmap('tab10')
    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(12, 9))

    # 设置标签 - 使用更大的字体
    ax.set_xlabel('Latency (ms/img) ↓', fontsize=18)
    ax.set_ylabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=18)
    ax.set_title('Accuracy vs. Latency', fontsize=20)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 绘制每个组
    for group in sorted(model_groups.keys()):
        color = group_colors[group]
        marker = markers[sorted(model_groups.keys()).index(group) % len(markers)]

        # 获取组内模型并按ms/img排序
        models_in_group = sorted(model_groups[group], key=lambda x: x[1]['ms/img'])
        ms_img_values = [m[1]['ms/img'] for m in models_in_group]
        a1_values = [m[1][dataset_name]['a1'] for m in models_in_group]

        # 绘制散点
        ax.scatter(ms_img_values, a1_values, marker=marker, color=color, s=150, label=group, zorder=5)

        # 如果是系列模型，用虚线连接
        if len(models_in_group) > 1:
            ax.plot(ms_img_values, a1_values, color=color, linestyle='--', linewidth=2.5, zorder=4)

        # 添加文本标签 - 使用更大的字体
        for i, (model_name, model_info) in enumerate(models_in_group):
            # 根据位置决定旋转角度
            rotation_angle = 0
            if i == 0:  # 第一个点
                rotation_angle = -10
            elif i == len(models_in_group) - 1:  # 最后一个点
                rotation_angle = 10

            # 调整偏移量
            offset_x = model_info.get('ms/img') * 0.05 + 0.1
            offset_y = 0.005

            ax.text(model_info.get('ms/img') + offset_x,
                    model_info[dataset_name]['a1'] + offset_y,
                    model_name, fontsize=14, verticalalignment='center',
                    rotation=rotation_angle, rotation_mode='anchor')

    # 添加图例 - 使用更大的字体
    ax.legend(title="Model Series", loc='lower right', fontsize=16)

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_latency_vs_accuracy_enhanced.png", dpi=300, bbox_inches='tight')
    print(f"准确度 vs 延迟图已保存为: {dataset_name}_latency_vs_accuracy_enhanced.png")
    plt.show()

    return fig, ax


def plot_gflops_vs_accuracy(data, dataset_name='KITTI'):
    """
    单独绘制准确度 vs 计算量图，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有gflops和a1数据的模型
    # 设置为新罗马字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    filtered_data = {}
    for model_name, metrics in data.items():
        if (metrics.get('gflops') is not None and
                dataset_name in metrics and
                'a1' in metrics[dataset_name]):
            filtered_data[model_name] = metrics

    # 分组模型
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X']
    cmap = plt.get_cmap('tab10')
    cmap = plt.get_cmap('Set1')

    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}
    print(group_colors)
    group_colors["Ours Model"] = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
    group_colors["DSI-MonoViT"] = (1.0, 0.4980392156862745, 0.0, 1.0)
    group_colors['Lite-Mono'] = (2 / 255, 0, 1.0, 1.0)
    group_colors['GeoDepth'] = (0, 174 / 255, 238 / 255, 1.0)
    group_colors['FGTO'] = (0, 176 / 255, 84 / 255, 1.0)
    group_colors['TinyDepth'] = (5 / 255, 255 / 255, 255 / 255, 1.0)

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置标签 - 使用更大的字体
    ax.set_xlabel('Computational Cost (GFLOPs) ↓', fontsize=22)
    ax.set_ylabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=22)
    ax.set_title('Accuracy vs. Computational Cost', fontsize=24)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)

    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 修改：重新排序组，确保Ours Model在最后
    sorted_groups = sorted([group for group in model_groups.keys() if group != 'Ours Model'])
    sorted_groups.append('Ours Model')  # 将Ours Model添加到列表最后

    # 绘制每个组 - 使用新的排序
    for group in sorted_groups:
        color = group_colors[group]
        marker = markers[sorted_groups.index(group) % len(markers)]

        # 获取组内模型并按gflops排序
        models_in_group = sorted(model_groups[group], key=lambda x: x[1]['gflops'])
        gflops_values = [m[1]['gflops'] for m in models_in_group]
        a1_values = [m[1][dataset_name]['a1'] for m in models_in_group]

        # 绘制散点
        ax.scatter(gflops_values, a1_values, marker=marker, color=color, s=180, label=group, zorder=5)

        # 如果是系列模型，用虚线连接
        if len(models_in_group) > 1:
            ax.plot(gflops_values, a1_values, color=color, linestyle='--', linewidth=5.5, zorder=4)

        # 添加文本标签 - 使用更大的字体
        for i, (model_name, model_info) in enumerate(models_in_group):
            # 根据位置决定旋转角度
            rotation_angle = 0
            verticalalignment = 'bottom',
            offset_x = 0.15
            offset_y = 0.002
            if model_name.startswith("F"):
                rotation_angle = 0  # -10
                # verticalalignment = "center"
                offset_x = -6.15
                offset_y = -0.001
            elif model_name.startswith("MonoVit"):
                offset_x = -8.45
                offset_y = 0.000
            elif model_name.startswith("Lite-Mono-8M"):
                offset_x = 0.25
                offset_y = -0.002
                pass
            elif model_name.startswith("TinyDepth"):
                offset_x = 0.85
                offset_y = -0.001
                rotation_angle = -0

            elif model_name.startswith("DSI"):
                offset_x = -11.85
                offset_y = 0.00
                rotation_angle = -0

            elif model_name.startswith("GeoDepth"):
                offset_x = 0.45
                offset_y = 0.000
            elif model_name.startswith("ra-depth"):
                offset_x = 0.65
                offset_y = -0.001
            elif model_name.startswith("Ours-M"):
                offset_x = -6.45
                offset_y = -0.001
            elif model_name.startswith("Ours-N"):
                offset_x = -2.45
                offset_y = 0.002

            elif model_name.startswith("Ours-S"):
                offset_x = -4.45
                offset_y = 0.001
            elif model_name.startswith("Ours-L"):
                offset_x = -4.45
                offset_y = 0.00
            elif model_name.startswith("Ours-X"):
                offset_x = -2.45
                offset_y = 0.000
            elif model_name == "Lite-Mono":
                offset_x = 0.45
                offset_y = -0.002
            elif model_name == "Lite-Mono-Tiny":
                offset_x = -1.45
                offset_y = -0.001
                rotation_angle = -60
            elif model_name == "Lite-Mono-Small":
                offset_x = 0.45
                offset_y = 0.001
            elif model_name == "Monodepth2":
                offset_x = -2.45
                offset_y = -0.002

            else:
                if i == 0:  # 第一个点
                    rotation_angle = -70  # -10
                elif i == len(models_in_group) - 1:  # 最后一个点
                    rotation_angle = -60

            # 调整偏移量
            ax.text(model_info['gflops'] + offset_x,
                    model_info[dataset_name]['a1'] + offset_y,
                    model_name, fontsize=26, verticalalignment='bottom',
                    rotation=rotation_angle, rotation_mode='anchor', weight='bold', fontname='Times New Roman')

    # 修改：手动控制图例顺序，确保Ours Model在最后
    handles, labels = ax.get_legend_handles_labels()

    # 重新排序：将Ours Model移到最后一个位置
    if 'Ours Model' in labels:
        ours_index = labels.index('Ours Model')
        handles.append(handles.pop(ours_index))
        labels.append(labels.pop(ours_index))

    # 使用重新排序后的句柄和标签创建图例
    ax.legend(handles, labels, title="Model Series", loc='lower right',prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 26},
              title_fontproperties={'family': 'Times New Roman', 'weight': 'bold', 'size': 30},

              frameon=True,)#fontsize=20, title_fontsize=22,

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # 调整坐标轴范围，为文本标注留出空间
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] * 1.0, xlim[1] * 1.05)
    ax.set_ylim(ylim[0] * 0.99, ylim[1] * 1.001)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_gflops_vs_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"准确度 vs 计算量图已保存为: {dataset_name}_gflops_vs_accuracy.png")
    plt.show()

    return fig, ax

def plot_gflops_vs_accuracy_(data, dataset_name='KITTI'):
    """
    单独绘制准确度 vs 计算量图，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有gflops和a1数据的模型
    # 设置为新罗马字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    filtered_data = {}
    for model_name, metrics in data.items():
        if (metrics.get('gflops') is not None and
                dataset_name in metrics and
                'a1' in metrics[dataset_name]):
            filtered_data[model_name] = metrics

    # 分组模型
    # print(filtered_data)
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X']
    cmap = plt.get_cmap('tab10')
    cmap = plt.get_cmap('Set1')

    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}
    print(group_colors)
    group_colors["Ours Model"] =(0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
    group_colors["DSI-MonoViT"] =(1.0, 0.4980392156862745, 0.0, 1.0)
    group_colors['Lite-Mono'] = (2/255, 0, 1.0, 1.0)
    group_colors['GeoDepth'] = (0, 174/255,238/255, 1.0)
    group_colors['FGTO'] = (0, 176 / 255, 84 / 255, 1.0)
    group_colors['TinyDepth'] = (5/255, 255 / 255, 255 / 255, 1.0)

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置标签 - 使用更大的字体
    ax.set_xlabel('Computational Cost (GFLOPs) ↓', fontsize=22)
    ax.set_ylabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=22)
    ax.set_title('Accuracy vs. Computational Cost', fontsize=24)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)

    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 绘制每个组
    for group in sorted(model_groups.keys()):
        color = group_colors[group]
        marker = markers[sorted(model_groups.keys()).index(group) % len(markers)]

        # 获取组内模型并按gflops排序
        models_in_group = sorted(model_groups[group], key=lambda x: x[1]['gflops'])
        gflops_values = [m[1]['gflops'] for m in models_in_group]
        a1_values = [m[1][dataset_name]['a1'] for m in models_in_group]

        # 绘制散点
        ax.scatter(gflops_values, a1_values, marker=marker, color=color, s=180, label=group, zorder=5)

        # 如果是系列模型，用虚线连接
        if len(models_in_group) > 1:
            ax.plot(gflops_values, a1_values, color=color, linestyle='--', linewidth=5.5, zorder=4)

        # 添加文本标签 - 使用更大的字体
        for i, (model_name, model_info) in enumerate(models_in_group):
            # 根据位置决定旋转角度
            rotation_angle = 0
            verticalalignment = 'bottom',
            offset_x = 0.15
            offset_y = 0.002
            if model_name.startswith("F"):
                rotation_angle = 0  # -10
                #verticalalignment = "center"
                offset_x = -6.15
                offset_y = -0.001
            elif model_name.startswith("MonoVit"):
                offset_x = -8.45
                offset_y = 0.000
            elif model_name.startswith("Lite-Mono-8M"):
                offset_x = 0.25
                offset_y = -0.002
                pass
            elif model_name.startswith("TinyDepth")  :
                offset_x = 0.85
                offset_y = -0.001
                rotation_angle = -0

            elif model_name.startswith("DSI")  :
                offset_x = -11.85
                offset_y = 0.00
                rotation_angle = -0

            elif  model_name.startswith("GeoDepth"):
                offset_x = 0.45
                offset_y = 0.000
            elif model_name.startswith("ra-depth"):
                offset_x = 0.65
                offset_y = -0.001
            elif model_name.startswith("Ours-M"):
                offset_x = -6.45
                offset_y = -0.001
            elif model_name.startswith("Ours-N"):
                offset_x = -2.45
                offset_y = 0.002

            elif model_name.startswith("Ours-S"):
                offset_x = -4.45
                offset_y = 0.001
            elif model_name.startswith("Ours-L"):
                offset_x = -4.45
                offset_y = 0.00
            elif model_name.startswith("Ours-X"):
                offset_x = -2.45
                offset_y = 0.000
            elif model_name=="Lite-Mono":
                offset_x = 0.45
                offset_y = -0.002
            elif model_name=="Lite-Mono-Tiny":
                offset_x = -1.45
                offset_y = -0.001
                rotation_angle = -60
            elif model_name=="Lite-Mono-Small":
                offset_x = 0.45
                offset_y = 0.001
            elif model_name=="Monodepth2":
                offset_x = -2.45
                offset_y = -0.002

            else:
                if i == 0:  # 第一个点
                    rotation_angle = -70#-10
                elif i == len(models_in_group) - 1:  # 最后一个点
                    rotation_angle = -60

            # 调整偏移量


            ax.text(model_info['gflops'] + offset_x,
                    model_info[dataset_name]['a1'] + offset_y,
                    model_name, fontsize=26, verticalalignment='bottom',
                    rotation=rotation_angle, rotation_mode='anchor',weight='bold', fontname='Times New Roman')

    # 添加图例 - 使用更大的字体
    ax.legend(title="Model Series", loc='lower right', fontsize=16, title_fontsize=18,frameon=True)#prop={'family': 'Times New Roman'}

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # 添加性能方向说明文本
    # ax.text(0.02, 0.98, 'Higher is better →', transform=ax.transAxes,
    #         fontsize=20, verticalalignment='top', horizontalalignment='right',rotation=90,
    #         color='green', weight='bold')
    # ax.text(0.02, 0.02, '← Lower is better', transform=ax.transAxes,
    #         fontsize=20, verticalalignment='bottom',
    #         color='red', weight='bold')

    # 调整坐标轴范围，为文本标注留出空间
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] * 1.0, xlim[1] * 1.05)
    ax.set_ylim(ylim[0] * 0.99, ylim[1] * 1.001)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_gflops_vs_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"准确度 vs 计算量图已保存为: {dataset_name}_gflops_vs_accuracy.png")
    plt.show()

    return fig, ax


def plot_gflops_vs_error(data, dataset_name='KITTI'):
    """
    单独绘制误差 vs 计算量图，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有gflops数据的模型
    filtered_data = {}
    for model_name, metrics in data.items():
        if metrics.get('gflops') is not None and dataset_name in metrics:
            filtered_data[model_name] = metrics

    # 分组模型
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X']
    cmap = plt.get_cmap('tab10')
    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置标签 - 使用更大的字体
    ax.set_xlabel('Computational Cost (GFLOPs) ↓', fontsize=22)
    ax.set_ylabel(f'{dataset_name} AbsRel Error ↓', fontsize=22)
    ax.set_title('Error vs. Computational Complexity', fontsize=24)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 绘制每个组
    for group in sorted(model_groups.keys()):
        color = group_colors[group]
        marker = markers[sorted(model_groups.keys()).index(group) % len(markers)]

        # 获取组内模型并按gflops排序
        models_in_group = sorted(model_groups[group], key=lambda x: x[1]['gflops'])
        gflops_values = [m[1]['gflops'] for m in models_in_group]
        abs_rel_values = [m[1][dataset_name]['abs_rel'] for m in models_in_group]

        # 绘制散点
        ax.scatter(gflops_values, abs_rel_values, marker=marker, color=color, s=180, label=group, zorder=5)

        # 如果是系列模型，用虚线连接
        if len(models_in_group) > 1:
            ax.plot(gflops_values, abs_rel_values, color=color, linestyle='--', linewidth=2.5, zorder=4)

        # 添加文本标签 - 使用更大的字体
        for i, (model_name, model_info) in enumerate(models_in_group):
            # 根据位置决定旋转角度
            rotation_angle = 0
            if i == 0:  # 第一个点
                rotation_angle = -10
            elif i == len(models_in_group) - 1:  # 最后一个点
                rotation_angle = 10

            # 调整偏移量
            offset_x = 0.5
            offset_y = 0.0005

            ax.text(model_info['gflops'] + offset_x,
                    model_info[dataset_name]['abs_rel'] + offset_y,
                    model_name, fontsize=18, verticalalignment='bottom',
                    rotation=rotation_angle, rotation_mode='anchor')

    # 添加图例 - 使用更大的字体
    ax.legend(title="Model Series", loc='lower right', fontsize=16, title_fontsize=18)

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # 添加性能方向说明文本
    ax.text(0.02, 0.98, '← Lower is better', transform=ax.transAxes,
            fontsize=20, verticalalignment='top',
            color='red', weight='bold')
    ax.text(0.98, 0.02, '← Lower is better', transform=ax.transAxes,
            fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            color='red', weight='bold')

    # 反转y轴（误差越低越好）
    ax.invert_yaxis()

    # 调整坐标轴范围，为文本标注留出空间
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] * 0.95, xlim[1] * 1.05)
    ax.set_ylim(ylim[0] * 0.98, ylim[1] * 1.02)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_gflops_vs_error_enhanced.png", dpi=300, bbox_inches='tight')
    print(f"误差 vs 计算量图已保存为: {dataset_name}_gflops_vs_error_enhanced.png")
    plt.show()

    return fig, ax
def plot_accuracy_vs_error_with_gflops(data, dataset_name='KITTI'):
    """
    单独绘制准确度 vs 误差图，点大小表示计算量(GFLOPs)，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有a1, abs_rel和gflops数据的模型
    filtered_data = {}
    for model_name, metrics in data.items():
        if (dataset_name in metrics and
                'a1' in metrics[dataset_name] and
                'abs_rel' in metrics[dataset_name] and
                metrics.get('gflops') is not None):
            filtered_data[model_name] = metrics

    # 分组模型 - 只使用颜色区分，不使用不同标记
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    cmap = plt.get_cmap('tab10')
    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}

    # 计算计算量的范围，用于缩放点的大小
    gflops_values_all = [metrics['gflops'] for group in model_groups.values()
                         for _, metrics in group if metrics.get('gflops') is not None]
    if gflops_values_all:
        min_gflops, max_gflops = min(gflops_values_all), max(gflops_values_all)
        size_range = (50, 1800)  # 点的大小范围
    else:
        min_gflops, max_gflops = 0, 1
        size_range = (50, 50)

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(16, 12))

    # 设置标签 - 使用更大的字体，并添加箭头说明
    ax.set_xlabel(f'{dataset_name} AbsRel Error ↓', fontsize=24)
    ax.set_ylabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=24)
    ax.set_title('Accuracy vs. Error (Point Size = GFLOPs)', fontsize=26)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 绘制每个组 - 统一使用圆形标记
    for group in sorted(model_groups.keys()):
        color = group_colors[group]

        for model_name, model_info in model_groups[group]:
            a1 = model_info[dataset_name]['a1']
            abs_rel = model_info[dataset_name]['abs_rel']
            gflops = model_info['gflops']

            # 根据计算量计算点的大小
            if max_gflops > min_gflops:
                size = size_range[0] + (gflops - min_gflops) / (max_gflops - min_gflops) * (
                            size_range[1] - size_range[0])
            else:
                size = size_range[0]

            # 绘制点 - 统一使用圆形标记
            ax.scatter(abs_rel, a1, s=size, marker='o', color=color, alpha=0.8,
                       edgecolors='darkgray', linewidth=1.0)

            # 添加文字标注 - 使用更大的字体，减少距离
            # 根据模型名称长度调整偏移量
            name_length = len(model_name)
            offset_x = 0.00015  # 大幅减少水平偏移
            offset_y = 0.00015  # 大幅减少垂直偏移

            ax.text(abs_rel + offset_x, a1 + offset_y, model_name, fontsize=18,
                    verticalalignment='bottom', horizontalalignment='left',
                    alpha=0.9, weight='normal')

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.7)

    # 调整坐标轴范围，为文本标注留出空间
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] * 0.98, xlim[1] * 1.02)
    ax.set_ylim(ylim[0] * 0.98, ylim[1] * 1.02)

    # 添加性能方向说明文本 - 使用更大的字体
    ax.text(0.02, 0.98, 'Higher is better →', transform=ax.transAxes,
            fontsize=20, verticalalignment='top', color='green', weight='bold')
    ax.text(0.98, 0.02, '← Lower is better', transform=ax.transAxes,
            fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            color='red', weight='bold')

    # 添加计算量说明
    if gflops_values_all:
        # 在图表右下角添加计算量说明
        ax.text(0.98, 0.98, 'Point Size = GFLOPs', transform=ax.transAxes,
                fontsize=18, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_accuracy_vs_error_with_gflops.png", dpi=300, bbox_inches='tight')
    print(f"准确度 vs 误差图已保存为: {dataset_name}_accuracy_vs_error_with_gflops.png")
    plt.show()

    return fig, ax
def plot_accuracy_vs_error_with_params(data, dataset_name='KITTI'):
    """
    单独绘制准确度 vs 误差图，点大小表示参数量，使用大字体适合论文

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    # 过滤数据 - 只包含有a1, abs_rel和mparams数据的模型
    filtered_data = {}
    for model_name, metrics in data.items():
        if (dataset_name in metrics and
                'a1' in metrics[dataset_name] and
                'abs_rel' in metrics[dataset_name] and
                metrics.get('mparams') is not None):
            filtered_data[model_name] = metrics

    # 分组模型 - 只使用颜色区分，不使用不同标记
    model_groups = defaultdict(list)
    for model_name, metrics in filtered_data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups[group_name].append((model_name, metrics))

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-white')
    cmap = plt.get_cmap('tab10')
    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}

    # 计算参数量的范围，用于缩放点的大小
    mparams_values_all = [metrics['mparams'] for group in model_groups.values()
                          for _, metrics in group if metrics.get('mparams') is not None]
    if mparams_values_all:
        min_mparams, max_mparams = min(mparams_values_all), max(mparams_values_all)
        size_range = (200, 1500)  # 点的大小范围
    else:
        min_mparams, max_mparams = 0, 1
        size_range = (200, 200)

    # 创建图形 - 使用更大的尺寸
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置标签 - 使用更大的字体
    ax.set_xlabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=22)
    ax.set_ylabel(f'{dataset_name} AbsRel Error ↓', fontsize=22)
    ax.set_title('Accuracy vs. Error (Size = G)', fontsize=24)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 绘制每个组 - 统一使用圆形标记
    for group in sorted(model_groups.keys()):
        color = group_colors[group]

        for model_name, model_info in model_groups[group]:
            a1 = model_info[dataset_name]['a1']
            abs_rel = model_info[dataset_name]['abs_rel']
            mparams = model_info['mparams']

            # 根据参数量计算点的大小
            if max_mparams > min_mparams:
                size = size_range[0] + (mparams - min_mparams) / (max_mparams - min_mparams) * (
                            size_range[1] - size_range[0])
            else:
                size = size_range[0]

            # 绘制点 - 统一使用圆形标记
            ax.scatter(a1, abs_rel, s=size, marker='o', color=color, alpha=0.7,
                       label=group if model_name == model_groups[group][0][0] else "")

            # 添加文字标注 - 使用更大的字体
            ax.text(a1 + 0.002, abs_rel + 0.0005, model_name, fontsize=16,
                    verticalalignment='center', alpha=0.8)

    # 创建模型系列图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重

    # 添加参数量大小示例图例
    if mparams_values_all:
        # 选择几个代表性的参数量值
        if max_mparams > 10:
            params_legend = [min_mparams, max_mparams / 4, max_mparams / 2, max_mparams]
        else:
            params_legend = [min_mparams, (min_mparams + max_mparams) / 3, (min_mparams + max_mparams) * 2 / 3,
                             max_mparams]

        labels_legend = [f'{s:.1f}M' for s in params_legend]

        # 计算对应的大小
        sizes_points = [
            size_range[0] + (s - min_mparams) / (max_mparams - min_mparams) * (size_range[1] - size_range[0])
            for s in params_legend]

        # 创建大小图例句柄 - 使用灰色圆形
        size_handles = [plt.scatter([], [], s=size, c='gray', alpha=0.7, marker='o') for size in sizes_points]

        # 合并模型系列和参数量大小图例
        all_handles = list(by_label.values()) + size_handles
        all_labels = list(by_label.keys()) + labels_legend

        # 创建图例 - 使用更大的字体
        legend = ax.legend(all_handles, all_labels,
                           title="Model Series & Parameters", loc='upper right',
                           fontsize=16, title_fontsize=18)

        # 设置图例标题和标签的字体大小
        legend.get_title().set_fontsize(18)
        for text in legend.get_texts():
            text.set_fontsize(16)
    else:
        ax.legend(by_label.values(), by_label.keys(),
                  title="Model Series", loc='upper right', fontsize=16)

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)

    # 反转y轴（误差越低越好）
    ax.invert_yaxis()

    # 调整坐标轴范围，为图例留出空间
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0], xlim[1] * 1.05)
    ax.set_ylim(ylim[0], ylim[1] * 0.95)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_accuracy_vs_error_with_params.png", dpi=300, bbox_inches='tight')
    print(f"准确度 vs 误差图已保存为: {dataset_name}_accuracy_vs_error_with_params.png")
    plt.show()

    return fig, ax

def plot_all_individual_charts(data, dataset_name='KITTI'):
    """
    绘制所有单独的图表

    Args:
        data (dict): 模型性能数据
        dataset_name (str): 数据集名称
    """
    print(f"开始为 {dataset_name} 数据集绘制单独的图表...")

    # 绘制准确度 vs 延迟图
    # plot_latency_vs_accuracy(data, dataset_name)

    # 绘制误差 vs 计算量图
    # plot_gflops_vs_error(data, dataset_name)
    plot_gflops_vs_accuracy(data, dataset_name)

    # 绘制准确度 vs 误差图（点大小表示计算量）
    # plot_accuracy_vs_error_with_gflops(data, dataset_name)

    print(f"所有图表绘制完成！")


# 过滤函数：只保留含有KITTI数据集的模型
def filter_models_with_kitti(model_data):
    filtered_models = {}
    for model_name, model_info in model_data.items():
        # 检查模型是否包含KITTI数据集指标
        if 'KITTI' in model_info:
            filtered_models[model_name] = model_info
    return filtered_models
def plot_multidataset_depth_charts(data):
    """
    Generate optimized depth estimation charts for the KITTI dataset.

    This function creates a figure with two subplots to visualize model performance:
    1. Accuracy (a1) vs. Latency (ms/img)
    2. Error (abs_rel) vs. GFLOPs

    Models belonging to the same series (e.g., 'Ours-S', 'Ours-M') are
    connected by a dashed line to show their progression. The legend is
    placed in the bottom-right corner of each plot.

    Args:
        data (dict): A nested dictionary containing model performance metrics.
    """

    # --- Group models by series for line plots ---

    model_groups1 = defaultdict(list)
    for model_name, metrics in data.items():
        # 只包含有ms/img数据的模型
        if metrics.get('ms/img') is None:
            continue
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups1[group_name].append((model_name, metrics))

    # 第二个子图的分组（基于gflops数据）
    model_groups2 = defaultdict(list)
    for model_name, metrics in data.items():
        # 只包含有gflops数据的模型
        if metrics.get('gflops') is None:
            continue
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name
        model_groups2[group_name].append((model_name, metrics))

    model_groups = defaultdict(list)
    for model_name, metrics in data.items():
        if model_name.startswith('Ours-'):
            group_name = 'Ours Model'  # Use a clean name for the legend
        elif 'Lite-Mono' in model_name:
            group_name = 'Lite-Mono'
        else:
            group_name = model_name  # Treat single models as their own group
        model_groups[group_name].append((model_name, metrics))

    # --- Plotting settings ---
    plt.style.use('seaborn-v0_8-white')#-whitegrid
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X']
    # Use a color map that provides distinct colors
    cmap = plt.get_cmap('tab10')
    group_colors = {name: cmap(i) for i, name in enumerate(sorted(model_groups.keys()))}


    # --- Create the figure for the KITTI dataset ---
    dataset_name = 'KITTI'
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    # fig.suptitle(f'Monocular Depth Estimation on {dataset_name}', fontsize=22, fontweight='bold')

    # --- Subplot 1: Accuracy (a1) vs. Latency (ms/img) ---
    ax1 = axes[0]
    ax1.set_xlabel('Latency (ms/img) ↓', fontsize=14)
    ax1.set_ylabel(f'{dataset_name} Accuracy (a1) ↑', fontsize=14)
    #ax1.set_title('Accuracy vs. Latency', fontsize=16, fontweight='bold')

    # --- Subplot 2: Error (abs_rel) vs. GFLOPs ---
    ax2 = axes[1]
    ax2.set_xlabel('Computational Cost (GFLOPs) ↓', fontsize=14)
    ax2.set_ylabel(f'{dataset_name} AbsRel Error ↓', fontsize=14)
    #ax2.set_title('Error vs. GFLOPs', fontsize=16, fontweight='bold')

    # Iterate through sorted groups to ensure consistent colors
    for group in sorted(model_groups1.keys()):
        color = group_colors[group]
        marker = markers[sorted(model_groups1.keys()).index(group) % len(markers)]

        models_in_group1 = sorted(model_groups1[group], key=lambda x: x[1]['ms/img'])
        ms_img_values = [m[1]['ms/img'] for m in models_in_group1]
        a1_values = [m[1][dataset_name]['a1'] for m in models_in_group1]

        # 绘制点和连线（代码保持不变）
        ax1.scatter(ms_img_values, a1_values, marker=marker, color=color, s=100, label=group, zorder=5)
        if len(models_in_group1) > 1:
            ax1.plot(ms_img_values, a1_values, color=color, linestyle='--', linewidth=2, zorder=4)

        # 添加文字标注（代码保持不变）
        for i, (model_name, model_info) in enumerate(models_in_group1):
            print(model_info)
            if model_info.get("ms/img") is None:
                continue
            ax1.text(model_info.get('ms/img') * 1.03, model_info[dataset_name]['a1'],
                     model_name, fontsize=10, rotation=45, verticalalignment='center')

    # --- Plotting logic for Subplot 2 (Error vs. GFLOPs) ---
    for group in sorted(model_groups2.keys()):
        color = group_colors[group]
        marker = markers[sorted(model_groups2.keys()).index(group) % len(markers)]

        models_in_group2 = sorted(model_groups2[group], key=lambda x: x[1]['gflops'])
        gflops_values = [m[1]['gflops'] for m in models_in_group2]
        abs_rel_values = [m[1][dataset_name]['abs_rel'] for m in models_in_group2]

        # 绘制点和连线（代码保持不变）
        ax2.scatter(gflops_values, abs_rel_values, marker=marker, color=color, s=100, label=group, zorder=5)
        if len(models_in_group2) > 1:
            ax2.plot(gflops_values, abs_rel_values, color=color, linestyle='--', linewidth=2, zorder=4)

        # 添加文字标注（代码保持不变）
        for i, (model_name, model_info) in enumerate(models_in_group2):
            if model_info.get("gflops") is None:
                continue
            ax2.text(model_info['gflops'] * 0.93, model_info[dataset_name]['abs_rel'],
                     model_name, fontsize=10,rotation=45,  verticalalignment='center')


    # --- Final figure adjustments ---
    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Set legend to the bottom right corner
        ax.legend(title="Model Series", loc='lower right', fontsize=11)

    # Invert y-axis for error plot (lower is better)
    ax2.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.savefig(f"{dataset_name}_depth_analysis_optimized.png", dpi=600, bbox_inches='tight')
    print(f"Optimized plot saved for {dataset_name} as {dataset_name}_depth_analysis_optimized.png")
    save_individual_subplots(fig, axes, dataset_name)
    plt.show()
# 过滤函数：只保留含有KITTI数据集的模型
def filter_models_with_kitti(model_data):
    filtered_models = {}
    for model_name, model_info in model_data.items():
        # 检查模型是否包含KITTI数据集指标
        if 'KITTI' in model_info:
            filtered_models[model_name] = model_info
    return filtered_models


def save_individual_subplots(fig, axes, dataset_name):
    """单独保存每个子图为独立的图像文件，确保图例和绘图内容的形状一致"""
    subplot_names = ['latency_vs_accuracy', 'gflops_vs_error', 'accuracy_vs_error_with_gflops']

    # 预先收集所有轴的图例句柄信息
    legend_info = {}
    for i, ax in enumerate(axes):
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            legend_info[i] = {
                'handles': handles,
                'labels': labels,
                'title': ax.get_legend().get_title().get_text()
            }

            # 为每个句柄创建颜色到标记的映射
            color_to_marker = {}
            for handle in handles:
                if hasattr(handle, 'get_color'):
                    color = handle.get_color()
                    if hasattr(handle, 'get_marker'):
                        marker = handle.get_marker()
                        if marker is None:  # 对于Line2D对象，可能需要特殊处理
                            marker = handle.get_marker()
                        color_to_marker[tuple(color)] = marker
                    elif hasattr(handle, 'get_facecolor'):
                        # 对于Patch对象（如散点图图例）
                        color = handle.get_facecolor()
                        if hasattr(handle, 'get_marker'):
                            marker = handle.get_marker()
                            color_to_marker[tuple(color)] = marker
            legend_info[i]['color_to_marker'] = color_to_marker

    for i, (ax, name) in enumerate(zip(axes, subplot_names)):
        # 创建新的图形
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))

        # 复制原图内容到新图
        for line in ax.get_lines():
            ax_ind.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        marker=line.get_marker(),
                        markersize=line.get_markersize())

        # 复制散点图 - 使用从图例获取的正确标记
        for collection in ax.collections:
            offsets = collection.get_offsets()
            sizes = collection.get_sizes()
            facecolors = collection.get_facecolors()
            edgecolors = collection.get_edgecolors()
            linewidths = collection.get_linewidths()

            # 确定正确的标记
            marker = 'o'  # 默认
            if i in legend_info and len(facecolors) > 0:
                color_to_marker = legend_info[i]['color_to_marker']
                current_color = tuple(facecolors[0]) if hasattr(facecolors[0], '__len__') else facecolors

                # 查找最接近的颜色匹配
                for color_key, marker_val in color_to_marker.items():
                    if (hasattr(color_key, '__len__') and hasattr(current_color, '__len__') and
                            len(color_key) >= 3 and len(current_color) >= 3):
                        # 比较RGB值（忽略alpha通道）
                        if (abs(color_key[0] - current_color[0]) < 0.1 and
                                abs(color_key[1] - current_color[1]) < 0.1 and
                                abs(color_key[2] - current_color[2]) < 0.1):
                            marker = marker_val
                            break

            # 使用正确的标记创建散点
            scatter = ax_ind.scatter(offsets[:, 0], offsets[:, 1],
                                     s=sizes, c=facecolors,
                                     edgecolors=edgecolors, linewidths=linewidths,
                                     alpha=collection.get_alpha(),
                                     marker=marker)

        # 复制文本
        for text in ax.texts:
            ax_ind.text(text.get_position()[0], text.get_position()[1],
                        text.get_text(), fontsize=text.get_fontsize(),
                        color=text.get_color(), rotation=text.get_rotation(),
                        verticalalignment=text.get_verticalalignment())

        # 设置相同的坐标轴标签和范围
        ax_ind.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax_ind.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax_ind.set_xlim(ax.get_xlim())
        ax_ind.set_ylim(ax.get_ylim())

        # 复制图例
        if i in legend_info:
            info = legend_info[i]
            ax_ind.legend(info['handles'], info['labels'],
                          title=info['title'],
                          loc='lower right', fontsize=11)

        # 复制网格
        ax_ind.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 保存单独的子图
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"单独子图保存为: {dataset_name}_{name}.png")
if __name__ == '__main__':
    # Sample data provided by the user
    # prod
    model_data = {
        'Ours-N': {
            'mparams': 1.522, 'gflops': 0.718, 'ms/img': 0.6,
            'KITTI': {'abs_rel': 0.110, 'sq_rel': 0.794, 'rmse': 4.678, 'rmse_log': 0.184, 'a1': 0.878, 'a2': 0.961,
                      'a3': 0.983},
            'Cityscapes': {'abs_rel': 0.107, 'sq_rel': 1.261, 'rmse': 6.133, 'rmse_log': 0.164, 'a1': 0.893,
                           'a2': 0.971, 'a3': 0.989}
        },
        'Ours-S': {
            'mparams': 6.059, 'gflops': 2.776, 'ms/img': 0.8,
            'KITTI': {'abs_rel': 0.104, 'sq_rel': 0.713, 'rmse': 4.458, 'rmse_log': 0.179, 'a1': 0.890, 'a2': 0.964,
                      'a3': 0.983},
            'Cityscapes': {'abs_rel': 0.100, 'sq_rel': 1.078, 'rmse': 5.813, 'rmse_log': 0.153, 'a1': 0.904,
                           'a2': 0.975, 'a3': 0.991}
        },
        'Ours-M': {
            # 0.098	0.677	4.269	0.174	0.901	0.967	0.984
            # 0.097	0.649	4.268	0.173	0.901	0.967	0.985
            'mparams': 12.539, 'gflops': 9.994, 'ms/img': 1.3,
            'KITTI': {'abs_rel': 0.097, 'sq_rel': 0.649, 'rmse': 4.268, 'rmse_log': 0.173, 'a1': 0.901, 'a2': 0.967,
                      'a3': 0.985},
            'Cityscapes': {'abs_rel': 0.091, 'sq_rel': 0.908, 'rmse': 5.446, 'rmse_log': 0.143, 'a1': 0.916,
                           'a2': 0.980, 'a3': 0.993}
        },
        'Ours-L': {
            # 0.095	0.642	4.199	0.171	0.906	0.968
            # 0.096	0.634	4.225	0.171	0.904	0.968	0.985
            # 0.095   0.639   4.206   0.171   0.905   0.968   0.985
            'mparams': 15.020, 'gflops': 11.519, 'ms/img': 1.5,
            'KITTI': {'abs_rel': 0.095, 'sq_rel': 0.642, 'rmse': 4.199, 'rmse_log': 0.171, 'a1': 0.906, 'a2': 0.968,
                      'a3': 0.984},
            'Cityscapes': {'abs_rel': 0.089, 'sq_rel': 0.912, 'rmse': 5.364, 'rmse_log': 0.142, 'a1': 0.920,
                           'a2': 0.980, 'a3': 0.992}
        },
        'Ours-X': {
            'mparams': 32.030, 'gflops': 23.795, 'ms/img': 2.3,
            'KITTI': {'abs_rel': 0.093, 'sq_rel': 0.616, 'rmse': 4.176, 'rmse_log': 0.169, 'a1': 0.909, 'a2': 0.969,
                      'a3': 0.985},
            'Cityscapes': {'abs_rel': 0.086, 'sq_rel': 0.838, 'rmse': 5.252, 'rmse_log': 0.138, 'a1': 0.924,
                           'a2': 0.981, 'a3': 0.993}
        },
        'Lite-Mono-8M': {
            'mparams': 8.766, 'gflops': 11.212, 'ms/img': 1.7,
            'KITTI': {'abs_rel': 0.101, 'sq_rel': 0.724, 'rmse': 4.457, 'rmse_log': 0.178, 'a1': 0.897, 'a2': 0.965,
                      'a3': 0.983},
        },
        'Lite-Mono': {
            'mparams': 3.069, 'gflops': 5.032, 'ms/img': 1.3,
            'KITTI': {'abs_rel': 0.107, 'sq_rel': 0.763, 'rmse': 4.566, 'rmse_log': 0.184, 'a1': 0.886, 'a2': 0.963,
                      'a3': 0.983},
        },
        'Lite-Mono-Small': {
            'mparams': 2.472, 'gflops': 4.746, 'ms/img': 1.2,
            'KITTI': {'abs_rel': 0.110, 'sq_rel': 0.801, 'rmse': 4.680, 'rmse_log': 0.186, 'a1': 0.879, 'a2': 0.961,
                      'a3': 0.982},
        },
        'Lite-Mono-Tiny': {
            'mparams': 2.143, 'gflops': 2.842, 'ms/img': 1.0,
            'KITTI': {'abs_rel': 0.110, 'sq_rel': 0.829, 'rmse': 4.708, 'rmse_log': 0.188, 'a1': 0.880, 'a2': 0.960,
                      'a3': 0.982},
        },

        # 'ra-depth': {
        #     'mparams': 9.98, 'gflops': 10.78, 'ms/img': None,
        #     'KITTI': {'abs_rel': 0.096, 'sq_rel': 0.632, 'rmse': 4.216, 'rmse_log': 0.171, 'a1': 0.903, 'a2': 0.968,
        #               'a3': 0.985},
        # },
        'MonoVit': {
            'mparams': 33.631, 'gflops': 59.679, 'ms/img': 3.8,
            'KITTI': {'abs_rel': 0.099, 'sq_rel': 0.708, 'rmse': 4.372, 'rmse_log': 0.175, 'a1': 0.900, 'a2': 0.967,
                      'a3': 0.984},
        },
        'Monodepth2': {
            'mparams': 14.329, 'gflops': 8.038, 'ms/img': 0.7,
            'KITTI': {'abs_rel': 0.115, 'sq_rel': 0.903, 'rmse': 4.863, 'rmse_log': 0.193, 'a1': 0.877, 'a2': 0.959,
                      'a3': 0.981},
            'Cityscapes': {'abs_rel': 0.129, 'sq_rel': 1.569, 'rmse': 6.876, 'rmse_log': 0.187, 'a1': 0.849,
                           'a2': 0.957, 'a3': 0.983}
        },
        'FGTO': {
            'mparams': 27.870, 'gflops': 59.679, 'ms/img': None,
             'KITTI': {'abs_rel': 0.096, 'sq_rel': 0.696, 'rmse': 4.327, 'rmse_log': 0.174, 'a1': 0.904, 'a2': 0.968,
                       'a3': 0.985},
            'Cityscapes': {'abs_rel': 0.088, 'sq_rel': 0.795, 'rmse': 5.368, 'rmse_log': 0.140, 'a1': 0.920,
                           'a2': 0.981, 'a3': 0.994}
        },
        # 其他模型（表格中未包含，设置为None）
        'DSI-MonoViT': {
            'mparams': 27.870, 'gflops': 59.679, 'ms/img': None,
            'KITTI': {'abs_rel': 0.096, 'sq_rel': 0.711, 'rmse': 4.321, 'rmse_log': 0.172, 'a1': 0.905, 'a2': 0.968,
                      'a3': 0.984},
            'Cityscapes': {'abs_rel': 0.088, 'sq_rel': 0.872, 'rmse': 5.410, 'rmse_log': 0.141, 'a1': 0.918,
                           'a2': 0.981, 'a3': 0.993}
        },
        'TinyDepth': {
            'mparams': 6.200, 'gflops': 13.300, 'ms/img': None,
            'KITTI': {'abs_rel': 0.096, 'sq_rel': 0.665, 'rmse': 4.249, 'rmse_log': 0.171, 'a1': 0.904, 'a2': 0.968,
                      'a3': 0.985},
        },
        'prodepth': {
            'mparams': 39.005, 'gflops': 42.312, 'ms/img': None,
            # 'KITTI': {'abs_rel': 0.086, 'sq_rel': 0.629, 'rmse': 4.139, 'rmse_log': 0.166, 'a1': 0.918, 'a2': 0.969,
            #           'a3': 0.984},
            'Cityscapes': {'abs_rel': 0.095, 'sq_rel': 0.876, 'rmse': 5.531, 'rmse_log': 0.146, 'a1': 0.908,
                           'a2': 0.978, 'a3': 0.993}
        },


        # 'mg-mono': {
        #     'mparams': 1.400, 'gflops': 3.270, 'ms/img': None,
        #     'KITTI': {'abs_rel': 0.097, 'sq_rel': 0.720, 'rmse': 4.309, 'rmse_log': 0.173, 'a1': 0.903, 'a2': 0.966,
        #               'a3': 0.984},
        # },
        'GeoDepth': {
            'mparams': 10.0, 'gflops': 11.9, 'ms/img': None,
            'KITTI': {'abs_rel': 0.100, 'sq_rel': 0.694, 'rmse': 4.381, 'rmse_log': 0.176, 'a1': 0.897, 'a2': 0.966,
                      'a3': 0.984},
        },
        'hybrid-depth': {
            'mparams': None, 'gflops': None, 'ms/img': None,
            'KITTI': {'abs_rel': 0.093, 'sq_rel': 0.596, 'rmse': 4.113, 'rmse_log': 0.167, 'a1': 0.910, 'a2': 0.970,
                      'a3': 0.986},
        },
        'manydepth': {
            'mparams': 26.923, 'gflops': 12.077, 'ms/img': None,
            'Cityscapes': {'abs_rel': 0.114, 'sq_rel': 1.193, 'rmse': 6.223, 'rmse_log': 0.170, 'a1': 0.875,
                           'a2': 0.967, 'a3': 0.989}
        },

        'dynamicdepth': {
            'mparams': 41.253, 'gflops': 18.508, 'ms/img': None,
            'Cityscapes': {'abs_rel': 0.103, 'sq_rel': 1.000, 'rmse': 5.867, 'rmse_log': 0.157, 'a1': 0.895,
                           'a2': 0.974, 'a3': 0.991}
        },
    }
    # 过滤
    model_data = filter_models_with_kitti(model_data)

    #plot_multidataset_depth_charts(model_data)
    plot_all_individual_charts(model_data, dataset_name='KITTI')