import os
import torch
import math
import matplotlib.pyplot as plt
import json
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import cv2
import io
from matplotlib import colors
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from collections import defaultdict

# def subplots(input, color=None):
#     if color is True:
#         light_colors = ['white', 'lightcoral', 'coral'] 
#         my_light_cmap = LinearSegmentedColormap.from_list("my_light_cmap", light_colors)
#         cmap = my_light_cmap
#     else:
#         cmap = 'gray'
#     input = input.squeeze()
#     # vmaxs = torch.max(torch.abs(input))
#     if isinstance(input, torch.Tensor):
#         if input.is_cuda:
#             input = input.detach().cpu()
#     length = input.squeeze().shape[0]
#     num = math.ceil(math.sqrt(length))
#     fig, axe = plt.subplots(num,num)
#     for i in range(num):
#         for j in range(num):
#             if i+num*j >= length:
#                 continue
#             if color:
#                 vmaxs = abs(input[i+num*j]).max()
#                 axe[i][j].imshow(input[i+num*j], cmap='bwr', vmin=-vmaxs, vmax=vmaxs)
#             else:
#                 axe[i][j].imshow(input[i+num*j], cmap='gray')
#     plt.show()

def subplots(input_data, color=False):
    """
    显示多张图像切片。
    参数:
        input_data: torch.Tensor 或 np.ndarray。支持形状 SHW, S1HW, S3HW, SHW3。
        color: bool。True 显示三通道 RGB，False 显示单通道灰度图。
    """
    # 1. 统一转换类型为 Numpy Array
    if isinstance(input_data, torch.Tensor):
        if input_data.is_cuda:
            input_data = input_data.detach().cpu()
        input_data = input_data.numpy()
    else:
        input_data = np.array(input_data)
        
    # 2. 形状对齐与安全变换 (取代危险的 squeeze)
    if color:
        # 目标形状: (S, H, W, 3)
        if input_data.ndim == 3: 
            # 输入可能是 3HW 或 HW3 (单张图的情况)
            if input_data.shape[0] == 3:      # 3HW -> 1HW3
                input_data = np.expand_dims(input_data.transpose(1, 2, 0), axis=0)
            elif input_data.shape[-1] == 3:   # HW3 -> 1HW3
                input_data = np.expand_dims(input_data, axis=0)
            else:
                raise ValueError(f"要求 3 通道，但获取到的形状为: {input_data.shape}")
        
        elif input_data.ndim == 4:
            # 输入可能是 S3HW 或 SHW3
            if input_data.shape[1] == 3:      # S3HW -> SHW3
                input_data = input_data.transpose(0, 2, 3, 1)
            elif input_data.shape[-1] == 3:   # 已经是 SHW3
                pass 
            else:
                raise ValueError(f"要求 3 通道，但获取到的形状为: {input_data.shape}")
    else:
        # 目标形状: (S, H, W)
        if input_data.ndim == 2:              # 单张图 HW -> 1HW
            input_data = np.expand_dims(input_data, axis=0)
        elif input_data.ndim == 4:
            # 去除多余的通道维度 (S1HW -> SHW 或 SHW1 -> SHW)
            if input_data.shape[1] == 1:
                input_data = np.squeeze(input_data, axis=1)
            elif input_data.shape[-1] == 1:
                input_data = np.squeeze(input_data, axis=-1)

    # 3. 计算子图布局
    length = input_data.shape[0]
    num = math.ceil(math.sqrt(length))
    
    # 动态调整画布大小，让图片看起来更清晰
    fig, axes = plt.subplots(num, num, figsize=(num * 3, num * 3))
    
    # 将二维的 axes 矩阵展平为一维，这样只需一个 for 循环，极其方便
    if num == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # 4. 遍历并绘图
    for i in range(len(axes)):
        if i < length:
            img = input_data[i]
            if color:
                # RGB 图像显示安全机制：如果是浮点数，将其归一化到 [0, 1] 避免 Matplotlib 报错
                if img.dtype != np.uint8:
                    min_val = input_data.min()
                    max_val = input_data.max()
                    if max_val > min_val:
                        img = (img - min_val) / (max_val - min_val)
                axes[i].imshow(img)
            else:
                # 灰度图直接显示
                axes[i].imshow(img, cmap='gray')
        
        # 隐藏所有的坐标轴刻度线和边框（包括多出来的空白子图）
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def load_json(pth):
    f = open(pth)
    content = f.read()
    data = json.loads(content)
    return data

def save_json(input,save):
    with open(save, "w") as file:
        json.dump(input, file)

def savenii(input, pth):
    if isinstance(input, list):
        input = np.stack(input, axis=0)
    header = nib.Nifti1Header()
    new_image = nib.Nifti1Image(input, np.eye(4), header=header) 
    nib.save(new_image, pth) 

def load_nii(pth):
    img = nib.load(pth)
    data = img.get_fdata()
    return data

def plot_mask_on_image(image, mask, color=False):

    image = image.squeeze()
    mask = mask.squeeze()
    if isinstance(image, torch.Tensor):
        if image.is_cuda:
            image = image.detach().cpu()
        if mask.is_cuda:
            mask = mask.detach().cpu()

    colors = [(0, 0, 0, 0),
              (1, 0, 0, 0.7),
              (0, 1, 0, 0.7)] 

    cmap = ListedColormap(colors)
    length = image.squeeze().shape[0]
    num = math.ceil(math.sqrt(length))
    _, axe = plt.subplots(num,num)
    for i in range(num):
        for j in range(num):
            if i+num*j >= length:
                continue
            # axe[i][j].imshow(image[i+num*j], cmap='gray')
            vmaxs = torch.max(torch.abs(image[i+num*j]))
            # axe[i][j].imshow(image[i+num*j], cmap='bwr', vmin=-vmaxs, vmax=vmaxs)
            if color:
                vmaxs = torch.max(torch.abs(image))
                axe[i][j].imshow(image[i+num*j], cmap='bwr', vmin=-vmaxs, vmax=vmaxs)
            else:
                axe[i][j].imshow(image[i+num*j], cmap='gray')
            axe[i][j].imshow(mask[i+num*j], cmap=cmap, interpolation='none', alpha=0.8)
    plt.show()

def analyze_and_plot_histograms(images):
    
    images = images.squeeze()
    if isinstance(images, torch.Tensor):
        if images.is_cuda:
            images = images.detach().cpu()

    num_images = images.shape[0]
    ncols = int(math.ceil(math.sqrt(num_images)))
    nrows = int(math.ceil(num_images / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i] # 获取当前子图
        # 加载图像并转为灰度图
        gray_image = images[i]
        image_array = np.array(gray_image)

        # 计算直方图
        hist, bin_edges = np.histogram(image_array.flatten(), range=[image_array.min(), image_array.max()])

        # 在当前子图(ax)上绘图
        ax.plot(bin_edges[:-1], hist, color='steelblue')
        # ax.set_title(filename, fontsize=9) # 将文件名作为标题
        # ax.set_xlim(0, 255)
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

    # --- 第4步: 隐藏多余的子图 ---
    # 如果图像数量不是网格大小的整数倍，有些子图会是空白的
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    # # 自动调整布局，防止标题和图像重叠
    # plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect为总标题留出空间
    
    # 显示最终的图表
    plt.show()

def overlap_plotting(x, y, info, color=['red', 'green'], casepth=None, cls='predition'):
        '''
        only for unique [0,1]
        '''
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.detach().cpu()
                y = y.detach().cpu()
        x = x.squeeze()
        y = y.squeeze()
        N, H, W = x.shape
        colors = ['None', color]
        cmap = ListedColormap(colors)
        for i in range(N):
            plt.figure(figsize=(6, 6))
            plt.imshow(x[i], cmap=ListedColormap(['None', color[1]]), alpha=0.5)
            plt.imshow(y[i], cmap=ListedColormap(['None', color[0]]), alpha=0.5)
            plt.axis('off')
            # plt.show()
            slc_name = info[1][i][0].removesuffix('.nii.gz')
            slcname = slc_name.removesuffix('.nii.gz')
            savepth = os.path.join(casepth, slcname+'.png')
            plt.savefig(savepth, format='png',bbox_inches='tight', pad_inches=0)
            plt.close()

def plotting(x, info, savepth=None, cls='predition'):
        '''
        only for unique [0,1]
        '''
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.detach().cpu()
        x = x.squeeze()
        N, H, W = x.shape
        case_pth = os.path.join(savepth, cls, info[0][0])
        if not os.path.exists(case_pth):
            os.makedirs(case_pth)
        for i in range(N):
            plt.figure(figsize=(6, 6))
            plt.imshow(x[i], cmap='gray')
            plt.axis('off')
            # plt.show()
            slc_name = info[1][i][0].removesuffix('.nii.gz')
            slcname = slc_name.removesuffix('.nii.gz')
            savepth = os.path.join(case_pth, slcname+'.png')
            plt.savefig(savepth, format='png',bbox_inches='tight', pad_inches=0)
            plt.close()

def Plot_flows(tensor):
    """
    将矢量场转换为白色背景的图像（矢量越小越白）。
    输入: tensor - 形状为 (B, 2, H, W)
    输出: rgb_images - 列表，包含 B 张 RGB 图像
    """
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    
    B, C, H, W = tensor.shape
    rgb_images = []

    for i in range(B):
        dx = tensor[i, 0, :, :]
        dy = tensor[i, 1, :, :]

        # 1. 计算极坐标
        mag = np.sqrt(dx**2 + dy**2)
        ang = np.arctan2(dy, dx)

        # 2. 映射到 HSV 
        # H (色调): 方向映射
        h = (ang + np.pi) / (2 * np.pi)
        
        # S (饱和度): 模长映射 (关键：模长小则饱和度低，即显现白色)
        max_mag = mag.max()
        s = mag / max_mag if max_mag > 0 else mag
        
        # V (明度): 固定为 1.0 确保背景是白的而不是黑的
        v = np.ones_like(h)

        # 3. 合成并转换
        hsv = np.stack([h, s, v], axis=-1)
        rgb = colors.hsv_to_rgb(hsv)
        rgb_images.append(rgb)

    return np.stack(rgb_images)

def overlay_masks(all_gray_images, all_masks_list, spth=None):
    """
    批量处理 S 张灰度图及其对应的 N 个掩膜。

    参数:
    all_gray_images (list): 包含 S 张 (H, W) 灰度图 (np.array) 的列表。
    all_masks_list (list): 包含 S 组掩膜的列表。
                           其中每个元素 (all_masks_list[s]) 是一个 (N, H, W) 
                           形状的 Numpy 数组，代表第 s 张图的 N 个掩膜。
    alpha (float): 掩膜的透明度。

    返回:
    list: 包含 S 张 (H, W, 3) BGR 结果图像 (np.array) 的列表。
    """
    colors = [
            # (44/255, 160/255, 44/255),   # 绿色
            (40/255, 39/255, 214/255),
            (180/255, 119/255, 31/255)


            # (214/255, 39/255, 40/255),   # 红色
            # (31/255, 119/255, 180/255),  # 蓝色
            # (255/255, 127/255, 14/255),  # 橙色
        ]

    # 从第一组掩膜中获取 N（种类数）
    # 假设所有图像的 N 值都相同
    results = []
    alpha=0.6
    # 2. 循环处理 S 张图像
    for s in range(all_gray_images.shape[0]):
        gray_img = np.array(all_gray_images[s].cpu(), dtype=np.float32)
        # 3. 调用核心函数进行叠加
        color_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        mask_overlay = np.zeros_like(color_image)

        for i, mask in enumerate(all_masks_list):
            mask = mask.cpu().numpy()
            mask_indices = (mask[s] > 0)
            mask_overlay[mask_indices] = colors[i]
            # a = np.unique(mask_overlay)
            # cv2.imshow("overlay", mask_overlay)
            # cv2.waitKey(0)
        beta = (1.0 - alpha)
        blended_image = cv2.addWeighted(mask_overlay, alpha, color_image, beta, 0)
        blended_image = blended_image * 255.0
        savepth = os.path.join(spth, f"overlay_gt_{s}.png")    
        cv2.imwrite(savepth, blended_image)
        # cv2.imshow("overlay", blended_image)
        # cv2.waitKey(0)
        # results.append(blended_image)
    # return results
    
    # return blended_image


def generate_grid_ppt(images, i, title_text, ppt_filename="output.pptx"):
    """
    处理 [N, H, W] 图像组，将它们逐一作为独立图片插入 PPT。
    排列成 N行 5列 的网格布局。
    """
    # 1. 提取 4 个核心输入
    gray1 = images[0].cpu().numpy()
    gray2 = images[1].cpu().numpy()
    mask1 = images[2].cpu().numpy()
    mask2 = images[3].cpu().numpy()
    
    N, H, W = gray1.shape
    
    # --- 辅助函数：将 [N, H, W] 灰度图转为 [N, H, W, 3] 的 RGB 图 ---
    def to_rgb_3d(gray_arr):
        ptp = np.ptp(gray_arr)
        if ptp == 0:
            normalized = np.zeros_like(gray_arr)
        else:
            normalized = (gray_arr - np.min(gray_arr)) / ptp * 255
        normalized = normalized.astype(np.uint8)
        return np.stack([normalized, normalized, normalized], axis=-1)
    
    # --- 辅助函数：将 [N, H, W] 的二值 mask 染成指定颜色 ---
    def colorize_mask_3d(mask_arr, color_rgb):
        colored = np.zeros((N, H, W, 3), dtype=np.uint8)
        colored[mask_arr == 1] = color_rgb
        return colored

    # 2. 图像染色与转换
    img_gray1 = to_rgb_3d(gray1)
    img_gray2 = to_rgb_3d(gray2)
    img_mask1 = colorize_mask_3d(mask1, [255, 0, 0])   # 鲜红色
    img_mask2 = colorize_mask_3d(mask2, [0, 0, 255])   # 亮蓝色
    
    # 生成重叠 Mask
    img_overlap = np.zeros((N, H, W, 3), dtype=np.uint8)
    img_overlap[mask1 == 1] = [255, 0, 0]
    img_overlap[mask2 == 1] = [0, 0, 255]
    overlap_condition = (mask1 == 1) & (mask2 == 1)
    img_overlap[overlap_condition] = [0, 255, 0]       # 鲜绿色
    
    # 3. PPT 初始化逻辑 (读取追加或新建)
    if os.path.exists(ppt_filename):
        prs = Presentation(ppt_filename)
    else:
        prs = Presentation()
        
    # 添加一页新的空白幻灯片
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_width = prs.slide_width
    slide_height = prs.slide_height
    
    # 4. 添加顶部标题文本框
    text_box_top = Inches(0.2)
    text_box_left = Inches(0)
    text_box_height = Inches(0.8)
    textbox = slide.shapes.add_textbox(text_box_left, text_box_top, slide_width, text_box_height)
    text_frame = textbox.text_frame
    text_frame.text = title_text
    
    for paragraph in text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.font.size = Pt(20)
        paragraph.font.bold = True

    # 5. 网格布局计算
    # 设置页面的边距
    margin_top = Inches(1.2)    # 给顶部标题留出空间
    margin_bottom = Inches(0.5) # 底部留白
    margin_left = Inches(0.5)   # 左侧留白
    margin_right = Inches(0.5)  # 右侧留白
    
    # 计算可用来放置图片的有效宽高
    avail_width = slide_width - margin_left - margin_right
    avail_height = slide_height - margin_top - margin_bottom
    
    # 计算每一个单元格（每张图片）的宽高
    cell_width = avail_width / 5
    cell_height = avail_height / N

    # 6. 逐张插入图片 (N 行, 5 列)
    for row_idx in range(N):
        row_images = [
            img_gray1[row_idx], 
            img_gray2[row_idx], 
            img_mask1[row_idx], 
            img_mask2[row_idx], 
            img_overlap[row_idx]
        ]
        
        for col_idx, img_arr in enumerate(row_images):
            img_pil = Image.fromarray(img_arr)
            image_stream = io.BytesIO()
            img_pil.save(image_stream, format='PNG')
            image_stream.seek(0)
            
            # --- 修复：计算等比例缩放和居中 ---
            img_h, img_w = img_arr.shape[:2]
            img_ratio = img_w / img_h
            cell_ratio = cell_width / cell_height
            
            if img_ratio > cell_ratio:
                # 原图比格子更“宽扁”，所以宽度顶满，高度等比缩小
                actual_width = cell_width
                actual_height = cell_width / img_ratio
            else:
                # 原图比格子更“瘦高”，所以高度顶满，宽度等比缩小
                actual_height = cell_height
                actual_width = cell_height * img_ratio
                
            # 计算在当前格子内的居中偏移量
            offset_x = (cell_width - actual_width) / 2
            offset_y = (cell_height - actual_height) / 2
            
            # 加上偏移量后的最终真实绝对坐标
            current_left = margin_left + (col_idx * cell_width) + offset_x
            current_top = margin_top + (row_idx * cell_height) + offset_y
            
            # 插入图片（填入计算后的等比宽高，PPT就不会挤压它了）
            slide.shapes.add_picture(
                image_stream, 
                left=current_left, 
                top=current_top, 
                width=actual_width, 
                height=actual_height
            )
            
    # 7. 保存文件
    prs.save(ppt_filename)
    # print(f"成功！已将索引 i={i} 的图像写入 PPT，文件路径: {ppt_filename}")

def generate_transposed_grid_ppt(images, i, title_text, ppt_filename="output.pptx"):
    """
    处理 [N, H, W] 图像组，将它们逐一插入 PPT。
    布局转置为 5行（图像类型） x N列（切片层数） 的网格。
    """
    # 1. 提取 4 个核心输入
    gray1 = images[0].cpu().numpy()
    gray2 = images[1].cpu().numpy()
    mask1 = images[2].cpu().numpy()
    mask2 = images[3].cpu().numpy()
    
    N, H, W = gray1.shape
    
    # --- 辅助函数：将 [N, H, W] 灰度图转为 [N, H, W, 3] 的 RGB 图 ---
    def to_rgb_3d(gray_arr):
        ptp = np.ptp(gray_arr)
        if ptp == 0:
            normalized = np.zeros_like(gray_arr)
        else:
            normalized = (gray_arr - np.min(gray_arr)) / ptp * 255
        normalized = normalized.astype(np.uint8)
        return np.stack([normalized, normalized, normalized], axis=-1)
    
    # --- 辅助函数：将 [N, H, W] 的二值 mask 染成指定颜色 ---
    def colorize_mask_3d(mask_arr, color_rgb):
        colored = np.zeros((N, H, W, 3), dtype=np.uint8)
        colored[mask_arr == 1] = color_rgb
        return colored

    # 2. 图像染色与转换
    img_gray1 = to_rgb_3d(gray1)
    img_gray2 = to_rgb_3d(gray2)
    img_mask1 = colorize_mask_3d(mask1, [255, 0, 0])   # 鲜红色
    img_mask2 = colorize_mask_3d(mask2, [0, 0, 255])   # 亮蓝色
    
    # 生成重叠 Mask
    img_overlap = np.zeros((N, H, W, 3), dtype=np.uint8)
    img_overlap[mask1 == 1] = [255, 0, 0]
    img_overlap[mask2 == 1] = [0, 0, 255]
    overlap_condition = (mask1 == 1) & (mask2 == 1)
    img_overlap[overlap_condition] = [0, 255, 0]       # 鲜绿色
    
    # 将这 5 组数据打包，方便后续按行遍历
    # 顺序为: 第一行灰度1, 第二行灰度2, 第三行红Mask, 第四行蓝Mask, 第五行绿Mask
    grid_data = [img_gray1, img_gray2, img_mask1, img_mask2, img_overlap]
    
    # 3. PPT 初始化逻辑 (读取追加或新建)
    if os.path.exists(ppt_filename):
        prs = Presentation(ppt_filename)
    else:
        prs = Presentation()
        
    # 添加一页新的空白幻灯片
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_width = prs.slide_width
    slide_height = prs.slide_height
    
    # 4. 添加顶部标题文本框
    text_box_top = Inches(0.2)
    text_box_left = Inches(0)
    text_box_height = Inches(0.8)
    textbox = slide.shapes.add_textbox(text_box_left, text_box_top, slide_width, text_box_height)
    text_frame = textbox.text_frame
    text_frame.text = title_text
    
    for paragraph in text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.font.size = Pt(20)
        paragraph.font.bold = True

    # 5. 网格布局计算 (已转置)
    margin_top = Inches(1.2)    # 给顶部标题留出空间
    margin_bottom = Inches(0.5) # 底部留白
    margin_left = Inches(0.5)   # 左侧留白
    margin_right = Inches(0.5)  # 右侧留白
    
    avail_width = slide_width - margin_left - margin_right
    avail_height = slide_height - margin_top - margin_bottom
    
    # ================= 变更点：计算转置后的单元格宽高 =================
    # 宽被平分给 N 列，高被平分给 5 行
    cell_width = avail_width / N
    cell_height = avail_height / 5
    # ====================================================================

    # 6. 逐张插入图片 (5 行, N 列)
    for row_idx, row_images in enumerate(grid_data):
        for col_idx in range(N):
            # 取出当前行的第 col_idx 个切片
            img_arr = row_images[col_idx]
            
            # 转为 PIL 并存入内存流
            img_pil = Image.fromarray(img_arr)
            image_stream = io.BytesIO()
            img_pil.save(image_stream, format='PNG')
            image_stream.seek(0)
            
            # 等比例缩放和居中计算 (逻辑保持不变)
            img_h, img_w = img_arr.shape[:2]
            img_ratio = img_w / img_h
            cell_ratio = cell_width / cell_height
            
            if img_ratio > cell_ratio:
                # 宽度顶满
                actual_width = cell_width
                actual_height = cell_width / img_ratio
            else:
                # 高度顶满
                actual_height = cell_height
                actual_width = cell_height * img_ratio
                
            offset_x = (cell_width - actual_width) / 2
            offset_y = (cell_height - actual_height) / 2
            
            # 计算变更后的横纵坐标
            current_left = margin_left + (col_idx * cell_width) + offset_x
            current_top = margin_top + (row_idx * cell_height) + offset_y
            
            # 插入图片
            slide.shapes.add_picture(
                image_stream, 
                left=current_left, 
                top=current_top, 
                width=actual_width, 
                height=actual_height
            )
            
    # 7. 保存文件
    prs.save(ppt_filename)


class MetricCalculator3D:
    """
    3D 医疗影像分割指标计算类。
    支持输入 logits [b, c, d, h, w] 和 groundtruth [b, d, h, w]。
    """
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        # 新增：使用 defaultdict(list) 可以自动为新的指标创建列表，非常方便后期扩展
        self.history = defaultdict(list)
        self.global_probs = []
        self.global_gts = []

    def _get_prediction(self, logit):
        """
        预处理函数：将网络输出的 logits 转换为二值预测。
        参数:
            logit: shape 为 [b, c, d, h, w], c通常为2(背景和前景)
        返回:
            pred: shape 为 [b, d, h, w] 的二值张量 (0 和 1)
        """
        # 在通道维度 (dim=1) 上取最大值的索引，将背景和前景的概率转化为具体的类别 0 或 1
        pred = torch.argmax(logit, dim=1)
        return pred

    def compute_dice(self, pred, gt):
        """
        计算 3D Dice 系数
        """
        # 在三维空间 (d, h, w) 上计算交集和各自的面积
        intersection = torch.sum(pred * gt, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        gt_sum = torch.sum(gt, dim=(1, 2, 3))
        
        # Dice 公式
        dice = (2.0 * intersection + self.epsilon) / (pred_sum + gt_sum + self.epsilon)
        
        # 如果 batch size 大于 1，取平均值；使用 .item() 将其转为 Python 原生浮点数
        return dice.mean().item()

    def compute_iou(self, pred, gt):
        """
        计算 3D IoU (交并比)
        """
        intersection = torch.sum(pred * gt, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        gt_sum = torch.sum(gt, dim=(1, 2, 3))
        
        # 并集 = 预测和 + 真实和 - 交集
        union = pred_sum + gt_sum - intersection
        
        # IoU 公式
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return iou.mean().item()

    def compute_area_error(self, pred, gt):
        # """
        # 计算 3D 体素数量(面积)误差
        # 返回: 预测数量 - 真实数量
        # """
        # pred_sum = torch.sum(pred, dim=(1, 2, 3))
        # gt_sum = torch.sum(gt, dim=(1, 2, 3))
        
        # error = pred_sum - gt_sum
        # return error.float().mean().item()
        """
        计算 3D 体素数量(面积/体积)误差百分比。
        计算公式: (预测数量 - 真实数量) / 真实数量 * 100%
        返回值含义：
            - 正数 (例如 15.0): 模型多预测了 15% 的体积 (过分割)
            - 负数 (例如 -10.0): 模型少预测了 10% 的体积 (欠分割)
            - 0: 体积完全一致
        """
        # 将求和结果转换为 float，以确保除法运算的精度
        pred_sum = torch.sum(pred, dim=(1, 2, 3)).float()
        gt_sum = torch.sum(gt, dim=(1, 2, 3)).float()
        
        # 计算百分比误差，分母加入 epsilon 防止 gt_sum 为 0 时报错
        error_percent = (abs(pred_sum - gt_sum) / (gt_sum + self.epsilon))

        
        return error_percent.mean().item()


    def compute_recall(self, pred, gt):
        """
        计算 3D 召回率 (Recall / True Positive Rate / Sensitivity)
        公式: TP / (TP + FN) -> 预测正确的正样本 / 真实的所有正样本
        """
        tp = torch.sum(pred * gt, dim=(1, 2, 3))
        gt_sum = torch.sum(gt, dim=(1, 2, 3)) # 真实标签的总和即为 TP + FN
        
        recall = (tp + self.epsilon) / (gt_sum + self.epsilon)
        return recall.mean().item()

    def compute_roc_auc(self, probs, gt):
        """
        计算全局 ROC AUC。
        注意：需要输入正类的概率图，而非二值预测。
        """
        # 转为 numpy 一维数组，交由 sklearn 计算
        probs_flat = probs.detach().cpu().numpy().flatten()
        gt_flat = gt.detach().cpu().numpy().flatten()
        
        try:
            auc = roc_auc_score(gt_flat, probs_flat)
        except ValueError:
            # 异常处理：当 batch 中全是背景或全是前景（只有单一类别）时，ROC 无法计算
            auc = float('nan') 
        return auc

    def compute_pr_auc(self, probs, gt):
        """
        计算全局 PR AUC (Average Precision)。
        适用于正负样本极度不平衡的医疗图像分割。
        """
        probs_flat = probs.detach().cpu().numpy().flatten()
        gt_flat = gt.detach().cpu().numpy().flatten()
        
        try:
            pr_auc = average_precision_score(gt_flat, probs_flat)
        except ValueError:
            pr_auc = float('nan')
        return pr_auc
    
    def compute_roc_curve_data(self, probs, gt, decimals=3):
        """
        提取 ROC 曲线数据。
        参数 decimals=3 用于将概率四舍五入到 3 位小数。
        这极大减少了 thresholds 的数量，从而大幅减小 JSON 文件体积，且不影响画图观感。
        """
        # 转为 numpy 并保留小数
        probs_flat = np.round(probs.detach().cpu().numpy().flatten(), decimals=decimals)
        gt_flat = gt.detach().cpu().numpy().flatten()
        
        try:
            # sklearn 返回 fpr, tpr 和 thresholds
            fpr, tpr, _ = roc_curve(gt_flat, probs_flat)
            # 转换为 Python list 以便能够被 json 序列化保存
            return {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        except ValueError:
            # 同样处理全纯色背景的异常情况
            return None

    def forward(self, logit, gt):
        """
        前向计算：计算基础指标，并收集用于全局曲线的数据。
        """
        pred = self._get_prediction(logit)
        gt = gt.to(pred.device).type_as(pred)
        probs = torch.softmax(logit, dim=1)[:, 1, ...]
        
        # 1. 记录常规的 Batch 级别指标
        results = {
            "dice": self.compute_dice(pred, gt),
            "iou": self.compute_iou(pred, gt),
            "area_error": self.compute_area_error(pred, gt),
            "recall": self.compute_recall(pred, gt),
            "pr_auc": self.compute_pr_auc(probs, gt),   # 计算当前 batch 的 PR AUC
            "roc_auc": self.compute_roc_auc(probs, gt)  # 计算当前 batch 的 ROC AUC
        }
        for key, value in results.items():
            self.history[key].append(value)

        # 2. 收集全局 ROC/PR 需要的概率图和标签
        probs = torch.softmax(logit, dim=1)[:, 1, ...] 
        
        # 【内存优化核心】：
        # 3D 数据极大，如果保存原始 float32 会迅速撑爆内存。
        # 我们提前将概率保留 3 位小数，并转换为 float16；标签转换为 int8。
        # 这样既不影响最终画图的平滑度，又能将内存占用降低数倍！
        probs_flat = np.round(probs.detach().cpu().numpy().flatten(), decimals=3).astype(np.float16)
        gt_flat = gt.detach().cpu().numpy().flatten().astype(np.int8)
        
        self.global_probs.append(probs_flat)
        self.global_gts.append(gt_flat)
        
        return results
    
    def summary(self, path=None):
        """
        计算历史记录的统计数据（均值、标准差、方差），以及全局的 ROC 和 PR 曲线数据。
        """
        stats = {}
        # 1. 计算常规指标的统计量 
        for key, values in self.history.items():
            valid_values = [v for v in values if not math.isnan(v)]
            if len(valid_values) > 0:
                stats[key] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "var": float(np.var(valid_values)) # [修改点]：新增计算方差 (Variance)
                }
            else:
                stats[key] = {"mean": 0.0, "std": 0.0, "var": 0.0}

        # 2. 计算全局 ROC 和 PR 曲线数据
        curve_data = {}
        if len(self.global_probs) > 0:
            print("⏳ 正在计算全局 ROC 和 PR 曲线，数据量较大，请稍候...")
            all_probs = np.concatenate(self.global_probs)
            all_gts = np.concatenate(self.global_gts)
            
            try:
                # ROC
                fpr, tpr, _ = roc_curve(all_gts, all_probs)
                roc_auc = roc_auc_score(all_gts, all_probs)
                stats["global_roc_auc"] = float(roc_auc)
                curve_data["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                
                # PR
                precision, recall, _ = precision_recall_curve(all_gts, all_probs)
                pr_auc = average_precision_score(all_gts, all_probs)
                stats["global_pr_auc"] = float(pr_auc)
                curve_data["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
                
            except ValueError:
                stats["global_roc_auc"] = 0.0
                stats["global_pr_auc"] = 0.0
                curve_data["roc_curve"] = None
                curve_data["pr_curve"] = None

        # 3. 导出 JSON 报告
        if path is not None:
            clean_history = {k: [None if math.isnan(v) else v for v in vals] for k, vals in self.history.items()}
            
            full_report = {
                "statistics": stats,
                "curves": curve_data,
                "raw_data": clean_history 
            }
            
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=4, ensure_ascii=False)
            
            print(f"🎯 验证完成！完整指标报告及双曲线数据已保存至: {path}")

        return stats, curve_data

    def reset(self):
        """
        清空所有历史记录，必须在每个 Epoch 开始前调用。
        """
        self.history.clear()
        self.global_probs.clear()
        self.global_gts.clear()

def plot_displacement(displacement_field, t_index=0, step=4, scale=1.0, save_path=None):
    """
    根据位移矢量场生成带有箭头的可视化图像。
    
    参数:
    displacement_field (numpy.ndarray 或 torch.Tensor): shape 为 [T, 2, H, W] 的位移场
    t_index (int): 需要绘制的时间/序列帧索引，默认为第 0 帧
    step (int): 采样步长。如果设为 4，则每 4x4 的区域只画一个箭头，避免图像过于拥挤
    scale (float): 箭头长度的缩放因子。值越大，画出来的箭头越短 (matplotlib的机制)
    save_path (str, optional): 如果提供路径，则将图片保存到本地
    """
    # 1. 兼容性处理：如果输入是 PyTorch 张量，先转换为 NumPy 数组
    if hasattr(displacement_field, 'detach'):
        displacement_field = displacement_field.detach().cpu().numpy()
        
    # 2. 从 [T, 2, H, W] 中提取出指定帧的 x 和 y 位移
    # u 为水平 (x) 方向位移，v 为垂直 (y) 方向位移
    # 使用切片 [::step, ::step] 进行降采样
    u = displacement_field[t_index, 0, ::step, ::step]
    v = displacement_field[t_index, 1, ::step, ::step]
    
    # 3. 获取降采样后的 H 和 W 尺寸
    H, W = displacement_field.shape[2], displacement_field.shape[3]
    
    # 4. 生成与提取的 u, v 对应的网格坐标 (X, Y)
    # np.arange 的用法：从 0 开始，到 W/H 结束，步长为 step
    x_coords = np.arange(0, W, step)
    y_coords = np.arange(0, H, step)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 5. 开始绘制图像
    plt.figure(figsize=(8, 8)) # 设置画布大小
    
    # 使用 quiver 绘制箭头
    # angles='xy': 箭头方向与 xy 坐标系一致
    # scale_units='xy': 箭头的比例尺单位基于 xy 坐标
    # color: 箭头颜色，你可以随意修改为 'red', 'green' 等
    plt.quiver(X, Y, u, v, angles='xy', scale_units='xy', scale=scale, color='blue', alpha=0.8)
    
    # 6. 反转 Y 轴，匹配通常的图像坐标系（原点在左上角）
    plt.gca().invert_yaxis()
    
    # 设置网格、标题和坐标轴比例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Displacement Vector Field (Frame t={t_index}, Step={step})")
    plt.xlabel("Width (X)")
    plt.ylabel("Height (Y)")
    plt.axis('equal') # 保证 X 轴和 Y 轴的缩放比例一致，防止箭头变形
    
    # 7. 保存或展示结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"矢量场图像已保存至: {save_path}")
    
    plt.show()