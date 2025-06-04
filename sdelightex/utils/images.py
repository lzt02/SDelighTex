import cv2
import numpy as np
import argparse
import os
import re
import time

def extract_number(filename):
    """从文件名中提取数字部分"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"文件名中无数字部分：{filename}")

def align_saturation(input_dir, reference_dir, output_dir, group_size=5, ext=("png", "jpg", "jpeg")):
    """
    批量对齐输入目录中的图像到参考目录中的对应图像，按组计算S通道参数
    
    参数:
        input_dir (str): 输入图像目录路径
        reference_dir (str): 参考图像目录路径
        output_dir (str): 输出目录路径
        group_size (int): 组的大小，默认为5
        ext (tuple): 图像文件扩展名，默认为('png', 'jpg', 'jpeg')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = tuple(f".{e.lower()}" for e in ext)
    
    # 获取并配对文件
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_exts)])
    
    if len(input_files) != len(reference_files) or \
        any(inf != ref for inf, ref in zip(input_files, reference_files)):
        raise ValueError("文件不匹配或数量不一致")
    
    # 按数字排序文件对
    file_pairs = sorted(
        [(os.path.join(input_dir, f), os.path.join(reference_dir, f)) for f in input_files],
        key=lambda x: extract_number(x[0].split(os.sep)[-1])
    )
    
    # 按组划分
    groups = [file_pairs[i:i+group_size] for i in range(0, len(file_pairs), group_size)]
    
    start_time = time.time()
    processed = 0
    
    for group_idx, group in enumerate(groups):
        # 计算组统计量
        sum_st, sum_s2, sum_s, sum_t, count = 0.0, 0.0, 0.0, 0.0, 0
        
        for inf_path, ref_path in group:
            img_input = cv2.imread(inf_path)
            img_ref = cv2.imread(ref_path)
            if img_input is None or img_ref is None:
                print(f"警告: 无法读取图像 {inf_path} 或 {ref_path}")
                continue
            
            # 转换到HSV空间并提取S通道
            s_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            s_gt = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            
            # 计算绝对差值并选择40%最小差值的像素
            diff = np.abs(s_input - s_gt)
            sorted_indices = np.argsort(diff.flatten())
            k = int(0.4 * diff.size)
            
            if k > 0:
                s_input_flat = s_input.flatten()
                s_gt_flat = s_gt.flatten()
                selected_indices = sorted_indices[:k]
                
                s_sel = s_input_flat[selected_indices]
                t_sel = s_gt_flat[selected_indices]
                
                # 累加统计量
                sum_st += np.sum(s_sel * t_sel)
                sum_s2 += np.sum(s_sel ** 2)
                sum_s += np.sum(s_sel)
                sum_t += np.sum(t_sel)
                count += k
        
        if count == 0:
            print(f"警告: 组 {group_idx} 无可用于计算参数的像素")
            continue
        
        # 计算线性变换参数 alpha 和 beta
        alpha = sum_st / sum_s2 if sum_s2 != 0 else 1.0
        beta = (sum_t - alpha * sum_s) / count
        
        # 应用参数到组内每个文件
        for inf_path, ref_path in group:
            img = cv2.imread(inf_path)
            if img is None:
                continue
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 应用线性变换
            s = np.clip(alpha * s.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            
            # 合并通道并保存结果
            aligned = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            output_path = os.path.join(output_dir, os.path.basename(inf_path))
            cv2.imwrite(output_path, aligned)
            processed += 1
    
    total_time = time.time() - start_time
    print(f"处理完成: 共处理 {processed} 张图像, 耗时 {total_time:.2f}秒")
    return processed, total_time
