# 最初思路，验证码部分是固定的
import cv2
import random
import os
import numpy as np
from collections import defaultdict, Counter
from analysis import SimpleVideoNoiseAnalyzer

# 运动方向定义（8方向 + 静止）
DIRECTIONS = {
    0: ("静止", (128, 128, 128)),      # 灰色
    1: ("右", (0, 255, 0)),            # 绿色
    2: ("左", (255, 0, 0)),            # 蓝色
    3: ("下", (0, 0, 255)),            # 红色
    4: ("上", (255, 255, 0)),          # 青色
    5: ("右下", (0, 128, 255)),        # 橙色
    6: ("左上", (255, 128, 0)),        # 天蓝色
    7: ("左下", (255, 0, 255)),        # 紫色
    8: ("右上", (0, 255, 255)),        # 黄色
}

def process_two_frames(frame1, frame2):
    """功能1：两帧做差取绝对值"""
    diff = cv2.absdiff(frame1, frame2)
    return diff

def auto_detect_block_size(gray1, gray2, min_size=8, max_size=32):
    """自动检测马赛克方块大小"""
    height, width = gray1.shape
    
    # 计算帧间差异
    diff = cv2.absdiff(gray1, gray2)
    
    # 计算水平投影
    horizontal_proj = np.mean(diff, axis=0)
    
    # 找到峰值和谷值
    peaks = []
    valleys = []
    
    for i in range(1, width-1):
        if horizontal_proj[i] > horizontal_proj[i-1] and horizontal_proj[i] > horizontal_proj[i+1]:
            peaks.append(i)
        elif horizontal_proj[i] < horizontal_proj[i-1] and horizontal_proj[i] < horizontal_proj[i+1]:
            valleys.append(i)
    
    # 计算峰谷之间的距离
    distances = []
    for i in range(1, len(peaks)):
        distances.append(peaks[i] - peaks[i-1])
    for i in range(1, len(valleys)):
        distances.append(valleys[i] - valleys[i-1])
    
    if not distances:
        return 16  # 默认值
    
    # 找到最常见的距离
    distance_counts = defaultdict(int)
    for d in distances:
        distance_counts[d] += 1
    
    most_common = max(distance_counts.items(), key=lambda x: x[1])[0]
    
    # 确保在合理范围内
    block_size = most_common
    if block_size < min_size:
        block_size = min_size
    elif block_size > max_size:
        block_size = max_size
    
    return block_size

def process_three_frames_optimized(frame1, frame2, frame3, video_path=None):
    """功能2：优化算法，自动检测马赛克大小并检测平移部分"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # 使用analysis模块检测马赛克大小
    if video_path and os.path.exists(video_path):
        analyzer = SimpleVideoNoiseAnalyzer(video_path, num_samples=3)
        grain_size = analyzer.analyze_video()
        if grain_size > 0:
            block_size = int(round(grain_size))
            print(f"使用analysis模块检测到的马赛克方块大小: {block_size} 像素")
        else:
            block_size = auto_detect_block_size(gray1, gray2)
            print(f"analysis模块检测失败，使用备用方法。马赛克方块大小: {block_size} 像素")
    else:
        block_size = auto_detect_block_size(gray1, gray2)
        print(f"自动检测到的马赛克方块大小: {block_size} 像素")
    
    height, width = gray1.shape
    
    # 确保尺寸能被块大小整除
    new_height = (height // block_size) * block_size
    new_width = (width // block_size) * block_size
    
    if new_height != height or new_width != width:
        print(f"调整图片尺寸从 {width}x{height} 到 {new_width}x{new_height} 以适配块大小")
        gray1 = cv2.resize(gray1, (new_width, new_height))
        gray2 = cv2.resize(gray2, (new_width, new_height))
        gray3 = cv2.resize(gray3, (new_width, new_height))
        height, width = gray1.shape
    
    # 分割为块
    h_blocks = height // block_size
    w_blocks = width // block_size
    
    # 计算每个块的差异
    block_diffs1 = np.zeros((h_blocks, w_blocks))
    block_diffs2 = np.zeros((h_blocks, w_blocks))
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            y_start = i * block_size
            y_end = y_start + block_size
            x_start = j * block_size
            x_end = x_start + block_size
            
            block1 = gray1[y_start:y_end, x_start:x_end]
            block2 = gray2[y_start:y_end, x_start:x_end]
            block3 = gray3[y_start:y_end, x_start:x_end]
            
            # 使用更鲁棒的差异度量
            diff1 = np.mean(cv2.absdiff(block1, block2))
            diff2 = np.mean(cv2.absdiff(block2, block3))
            
            block_diffs1[i, j] = diff1
            block_diffs2[i, j] = diff2
    
    # 自适应阈值
    mean_diff1 = np.mean(block_diffs1[block_diffs1 > 5])
    mean_diff2 = np.mean(block_diffs2[block_diffs2 > 5])
    threshold = min(mean_diff1, mean_diff2) * 0.5
    
    # 标记平移块
    motion_mask = np.zeros((h_blocks, w_blocks), dtype=np.uint8)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            diff1 = block_diffs1[i, j]
            diff2 = block_diffs2[i, j]
            
            # 检查是否满足平移条件
            if diff1 > threshold and diff2 > threshold:
                # 差异相似性检查
                if abs(diff1 - diff2) < max(diff1, diff2) * 0.3:  # 差异在30%以内
                    motion_mask[i, j] = 1
    
    # 使用形态学操作平滑结果
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # 创建输出图像
    result = np.ones((height, width), dtype=np.uint8) * 255
    
    # 标记平移部分为黑色
    for i in range(h_blocks):
        for j in range(w_blocks):
            if motion_mask[i, j] == 1:
                y_start = i * block_size
                y_end = y_start + block_size
                x_start = j * block_size
                x_end = x_start + block_size
                
                result[y_start:y_end, x_start:x_end] = 0
    
    return result, motion_mask, block_size

def detect_and_draw_boundaries(frame1, frame2, frame3, video_path=None):
    """功能3：检测平移部分并绘制区域边界线"""
    # 首先复用功能2的检测逻辑
    result, motion_mask, block_size = process_three_frames_optimized(frame1, frame2, frame3, video_path)

    # 在原始帧上绘制边界线
    boundary_image = frame1.copy()
    height, width = motion_mask.shape

    # 将块级mask放大到像素级，用于轮廓检测
    mask_upsampled = cv2.resize(
        (motion_mask * 255).astype(np.uint8),
        (frame1.shape[1], frame1.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # 找到移动区域的轮廓
    contours, _ = cv2.findContours(mask_upsampled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在图像上绘制轮廓边界线（红色，线宽2像素）
    cv2.drawContours(boundary_image, contours, -1, (0, 0, 255), 2)

    # 创建另一个版本：在原图上叠加半透明mask
    overlay_image = frame1.copy()
    overlay = np.zeros_like(frame1)
    overlay[mask_upsampled > 128] = [0, 0, 255]  # 红色半透明覆盖
    overlay_image = cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0)
    # 在叠加图像上也绘制边界线
    cv2.drawContours(overlay_image, contours, -1, (0, 255, 255), 2)  # 黄色边界线

    return result, motion_mask, block_size, boundary_image, overlay_image, contours


def analyze_local_motion_direction(gray1, gray2, gray3, x_start, y_start, unit_size):
    """
    分析局部单元的运动方向
    使用三帧差分和相位相关来估计运动向量

    返回: 方向编码 (0-8, 0表示静止)
    """
    # 提取局部区域
    h, w = gray1.shape
    x_end = min(x_start + unit_size, w)
    y_end = min(y_start + unit_size, h)

    if x_end - x_start < 8 or y_end - y_start < 8:
        return 0  # 区域太小，认为是静止

    roi1 = gray1[y_start:y_end, x_start:x_end].astype(np.float32)
    roi2 = gray2[y_start:y_end, x_start:x_end].astype(np.float32)
    roi3 = gray3[y_start:y_end, x_start:x_end].astype(np.float32)

    # 计算帧间差异，判断是否有运动
    diff12 = cv2.absdiff(roi1, roi2)
    diff23 = cv2.absdiff(roi2, roi3)

    mean_diff = (np.mean(diff12) + np.mean(diff23)) / 2

    # 如果差异太小，认为是静止
    if mean_diff < 5.0:
        return 0

    # 使用相位相关法计算运动向量
    try:
        # 计算1->2的运动
        (dx1, dy1), _ = cv2.phaseCorrelate(roi1, roi2)
        # 计算2->3的运动
        (dx2, dy2), _ = cv2.phaseCorrelate(roi2, roi3)

        # 平均运动向量
        dx = (dx1 + dx2) / 2
        dy = (dy1 + dy2) / 2

        # 如果运动幅度太小，认为是静止
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.5:
            return 0

        # 判断方向
        return classify_motion_direction(dx, dy)

    except Exception:
        return 0


def classify_motion_direction(dx, dy):
    """
    将运动向量分类为8个方向之一
    dx: x方向位移（正为右，负为左）
    dy: y方向位移（正为下，负为上）

    返回方向编码:
    0: 静止
    1: 右, 2: 左, 3: 下, 4: 上
    5: 右下, 6: 左上, 7: 左下, 8: 右上
    """
    # 计算角度
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # 根据角度分类（22.5度为边界）
    # 右: -22.5 ~ 22.5
    # 右下: 22.5 ~ 67.5
    # 下: 67.5 ~ 112.5
    # 左下: 112.5 ~ 157.5
    # 左: 157.5 ~ 180 或 -180 ~ -157.5
    # 左上: -157.5 ~ -112.5
    # 上: -112.5 ~ -67.5
    # 右上: -67.5 ~ -22.5

    if -22.5 <= angle < 22.5:
        return 1  # 右
    elif 22.5 <= angle < 67.5:
        return 5  # 右下
    elif 67.5 <= angle < 112.5:
        return 3  # 下
    elif 112.5 <= angle < 157.5:
        return 7  # 左下
    elif angle >= 157.5 or angle < -157.5:
        return 2  # 左
    elif -157.5 <= angle < -112.5:
        return 6  # 左上
    elif -112.5 <= angle < -67.5:
        return 4  # 上
    else:  # -67.5 <= angle < -22.5
        return 8  # 右上


def detect_motion_with_direction(frame1, frame2, frame3, video_path=None):
    """
    功能4：检测运动并分析运动方向
    基于analysis反馈尺寸的3倍作为局部单元大小
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # 获取analysis检测的尺寸
    if video_path and os.path.exists(video_path):
        analyzer = SimpleVideoNoiseAnalyzer(video_path, num_samples=3)
        grain_size = analyzer.analyze_video()
        if grain_size > 0:
            base_size = int(round(grain_size))
            print(f"使用analysis模块检测到的基础尺寸: {base_size} 像素")
        else:
            base_size = 16
            print(f"analysis模块检测失败，使用默认基础尺寸: {base_size} 像素")
    else:
        base_size = 16
        print(f"使用默认基础尺寸: {base_size} 像素")

    # 局部单元大小为base_size的3倍
    unit_size = base_size * 3
    print(f"局部运动分析单元大小: {unit_size} 像素")

    height, width = gray1.shape

    # 确保尺寸能被单元大小整除
    new_height = (height // unit_size) * unit_size
    new_width = (width // unit_size) * unit_size

    if new_height != height or new_width != width:
        print(f"调整图片尺寸从 {width}x{height} 到 {new_width}x{new_height} 以适配单元大小")
        gray1 = cv2.resize(gray1, (new_width, new_height))
        gray2 = cv2.resize(gray2, (new_width, new_height))
        gray3 = cv2.resize(gray3, (new_width, new_height))
        height, width = new_height, new_width
        # 同步调整彩色帧
        frame1_resized = cv2.resize(frame1, (new_width, new_height))
    else:
        frame1_resized = frame1.copy()

    # 计算单元网格
    h_units = height // unit_size
    w_units = width // unit_size

    # 分析每个单元的运动方向
    direction_map = np.zeros((h_units, w_units), dtype=np.int32)
    direction_counts = Counter()

    print(f"分析网格: {h_units}x{w_units} = {h_units*w_units} 个单元")

    for i in range(h_units):
        for j in range(w_units):
            y_start = i * unit_size
            x_start = j * unit_size

            direction = analyze_local_motion_direction(
                gray1, gray2, gray3, x_start, y_start, unit_size
            )
            direction_map[i, j] = direction
            direction_counts[direction] += 1

    # 找出主要的两种运动状态
    print("\n运动方向统计:")
    print("-" * 40)
    for dir_code, count in direction_counts.most_common():
        percentage = (count / (h_units * w_units)) * 100
        dir_name, _ = DIRECTIONS[dir_code]
        print(f"  {dir_name}: {count} 单元 ({percentage:.1f}%)")
    print("-" * 40)

    # 获取前两种主要运动状态
    top_two = direction_counts.most_common(2)
    dominant_directions = [d[0] for d in top_two]

    if len(dominant_directions) >= 2:
        print(f"\n主要运动状态:")
        for i, dir_code in enumerate(dominant_directions, 1):
            dir_name, color = DIRECTIONS[dir_code]
            print(f"  状态{i}: {dir_name} ({direction_counts[dir_code]} 单元)")
    else:
        print("\n警告: 未检测到明显的运动状态区分")

    # 创建可视化结果
    result_images = visualize_motion_directions(
        frame1_resized, direction_map, unit_size, dominant_directions
    )

    return {
        'base_size': base_size,
        'unit_size': unit_size,
        'direction_map': direction_map,
        'direction_counts': direction_counts,
        'dominant_directions': dominant_directions,
        **result_images
    }


def visualize_motion_directions(frame, direction_map, unit_size, dominant_directions):
    """
    可视化运动方向分析结果
    返回多种可视化图像
    """
    h_units, w_units = direction_map.shape
    height, width = frame.shape[:2]

    # 1. 创建方向颜色图
    direction_color_image = frame.copy()

    for i in range(h_units):
        for j in range(w_units):
            dir_code = direction_map[i, j]
            _, color = DIRECTIONS[dir_code]

            y_start = i * unit_size
            x_start = j * unit_size
            y_end = min(y_start + unit_size, height)
            x_end = min(x_start + unit_size, width)

            # 在单元格上叠加半透明颜色
            overlay = direction_color_image[y_start:y_end, x_start:x_end].copy()
            cv2.rectangle(overlay, (0, 0), (x_end-x_start, y_end-y_start), color, -1)
            direction_color_image[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                direction_color_image[y_start:y_end, x_start:x_end], 0.6,
                overlay, 0.4, 0
            )

            # 在单元格中心画小圆点表示方向
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            cv2.circle(direction_color_image, (center_x, center_y), 3, color, -1)

    # 2. 创建边界线图（只显示主要两种状态之间的边界）
    boundary_image = frame.copy()

    for i in range(h_units):
        for j in range(w_units):
            dir_code = direction_map[i, j]
            _, color = DIRECTIONS[dir_code]

            y_start = i * unit_size
            x_start = j * unit_size
            y_end = min(y_start + unit_size, height)
            x_end = min(x_start + unit_size, width)

            # 绘制单元格边框
            if dir_code in dominant_directions[:2]:
                # 主要状态用粗边框
                thickness = 2
            else:
                # 其他状态用细边框
                thickness = 1

            cv2.rectangle(boundary_image, (x_start, y_start), (x_end-1, y_end-1), color, thickness)

    # 3. 创建掩码图像（黑底，不同方向用不同亮度）
    mask_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(h_units):
        for j in range(w_units):
            dir_code = direction_map[i, j]
            # 方向编码映射到亮度值
            brightness = min(dir_code * 28, 255)

            y_start = i * unit_size
            x_start = j * unit_size
            y_end = min(y_start + unit_size, height)
            x_end = min(x_start + unit_size, width)

            mask_image[y_start:y_end, x_start:x_end] = brightness

    return {
        'direction_color_image': direction_color_image,
        'boundary_image': boundary_image,
        'mask_image': mask_image
    }

def extract_frames(video_path, mode=1):
    """从视频中提取帧进行处理"""
    if not os.path.exists(video_path):
        print("错误：视频文件不存在，请检查路径。")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查文件格式。")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if mode == 1 and total_frames < 2:
        print("错误：视频帧数不足，无法提取连续两帧。")
        cap.release()
        return None
    elif mode in [2, 3, 4] and total_frames < 3:
        print("错误：视频帧数不足，无法提取连续三帧。")
        cap.release()
        return None

    frames_needed = 3 if mode in [2, 3, 4] else 2
    frame1 = None
    frame2 = None
    frame3 = None
    random_start = 0

    # 尝试多个随机位置
    for attempt in range(5):
        random_start = random.randint(0, total_frames - frames_needed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_start)
        
        if mode == 1:
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            if ret1 and ret2 and frame1 is not None and frame2 is not None:
                break
        elif mode in [2, 3, 4]:
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            ret3, frame3 = cap.read()
            if ret1 and ret2 and ret3 and frame1 is not None and frame2 is not None and frame3 is not None:
                break

    cap.release()

    if mode == 1 and (frame1 is None or frame2 is None):
        print("错误：无法读取视频帧。")
        return None
    elif mode in [2, 3, 4] and (frame1 is None or frame2 is None or frame3 is None):
        print("错误：无法读取视频帧。")
        return None

    return (frame1, frame2, frame3, random_start) if mode in [2, 3, 4] else (frame1, frame2, random_start)

print("CCD LAB Project - Recognition")
print("=" * 50)
print("请输入数字选择对应版本：")
print("1. V1.0.0（支持基本识别，要求有固定元素）")
print("2. V2.0.0（仅支持动态识别）")
print("3. V2.0.1（开发中）")
print("=" * 50)
mode = int(input("请输入版本编号（1-3）：").strip())

print("请输入视频文件路径：")
video_path = input().strip()

# V1.0.0 - 支持基本识别，要求有固定元素
if mode == 1:
    print("\n运行 V1.0.0 - 基本识别模式")
    frames = extract_frames(video_path, mode=1)
    if frames:
        frame1, frame2, random_start = frames
        result = process_two_frames(frame1, frame2)
        
        output_path = input("请输入输出图片路径（留空则使用默认路径）：").strip()
        if not output_path:
            output_path = "frame_difference.jpg"
        
        cv2.imwrite(output_path, result)
        print("=" * 50)
        print("V1.0.0 处理完成！")
        print(f"差异图片已保存为：{output_path}")
        print(f"随机起始帧：{random_start}")
        print("=" * 50)

# V2.0.0 - 仅支持动态识别
elif mode == 2:
    print("\n运行 V2.0.0 - 动态识别模式")
    frames = extract_frames(video_path, mode=2)
    if frames:
        frame1, frame2, frame3, random_start = frames

        result, motion_mask, block_size, boundary_image, overlay_image, contours = detect_and_draw_boundaries(frame1, frame2, frame3, video_path)

        # 计算运动块的比例
        motion_blocks = np.sum(motion_mask)
        total_blocks = motion_mask.size
        motion_ratio = motion_blocks / total_blocks * 100
        num_contours = len(contours)

        output_path = input("请输入输出图片路径（留空则使用默认路径）：").strip()
        if not output_path:
            output_path = f"motion_detection_block{block_size}.jpg"

        cv2.imwrite(output_path, result)

        # 保存带边界线的图像
        boundary_path = f"boundary_block{block_size}.jpg"
        cv2.imwrite(boundary_path, boundary_image)

        # 保存带半透明叠加的图像
        overlay_path = f"overlay_boundary_block{block_size}.jpg"
        cv2.imwrite(overlay_path, overlay_image)

        print("=" * 50)
        print("V2.0.0 处理完成！")
        print(f"自动检测到的马赛克方块大小: {block_size} 像素")
        print(f"随机起始帧：{random_start}")
        print(f"检测到运动块：{motion_blocks}/{total_blocks} ({motion_ratio:.1f}%)")
        print(f"检测到 {num_contours} 个运动区域")
        print(f"平移检测结果已保存为：{output_path}")
        print(f"边界线标注结果已保存为：{boundary_path}")
        print(f"叠加可视化结果已保存为：{overlay_path}")
        print("平移部分为黑色，其他部分为白色")
        print("=" * 50)

# V2.0.1 - 开发中（自动辨别整合版）
elif mode == 3:
    print("\n运行 V2.0.1 - 自动辨别整合模式（开发中）")
    print("此版本尝试整合动态识别功能，自动辨别运动区域和方向")
    print("-" * 50)
    
    frames = extract_frames(video_path, mode=4)
    if frames:
        frame1, frame2, frame3, random_start = frames

        # 第一步：检测平移区域和边界
        print("\n[步骤1] 检测平移区域...")
        result_motion, motion_mask, block_size, boundary_image, overlay_image, contours = detect_and_draw_boundaries(frame1, frame2, frame3, video_path)
        
        motion_blocks = np.sum(motion_mask)
        total_blocks = motion_mask.size
        motion_ratio = motion_blocks / total_blocks * 100
        
        print(f"  检测到运动块：{motion_blocks}/{total_blocks} ({motion_ratio:.1f}%)")
        print(f"  检测到 {len(contours)} 个运动区域")

        # 第二步：分析运动方向
        print("\n[步骤2] 分析运动方向...")
        direction_result = detect_motion_with_direction(frame1, frame2, frame3, video_path)
        
        direction_counts = direction_result['direction_counts']
        dominant_directions = direction_result['dominant_directions']

        # 输出综合结果
        print("\n" + "=" * 60)
        print("V2.0.1 自动辨别结果（开发中）")
        print("=" * 60)
        
        print(f"\n基本信息：")
        print(f"  马赛克方块大小: {block_size} 像素")
        print(f"  随机起始帧：{random_start}")
        
        print(f"\n运动区域检测：")
        print(f"  运动块比例：{motion_ratio:.1f}%")
        print(f"  运动区域数量：{len(contours)}")
        
        print(f"\n运动方向分析：")
        if len(dominant_directions) >= 2:
            dir1_name, _ = DIRECTIONS[dominant_directions[0]]
            dir2_name, _ = DIRECTIONS[dominant_directions[1]]
            print(f"  主要运动状态1: {dir1_name} ({direction_counts[dominant_directions[0]]} 单元)")
            print(f"  主要运动状态2: {dir2_name} ({direction_counts[dominant_directions[1]]} 单元)")
        elif len(dominant_directions) == 1:
            dir_name, _ = DIRECTIONS[dominant_directions[0]]
            print(f"  主要运动状态: {dir_name}")
        else:
            print(f"  未检测到明显运动")

        # 保存结果
        output_base = input("\n请输入输出文件前缀（留空则使用默认）：").strip()
        if not output_base:
            output_base = "auto_detect"
        
        # 保存平移检测结果
        cv2.imwrite(f"{output_base}_motion.jpg", result_motion)
        cv2.imwrite(f"{output_base}_boundary.jpg", boundary_image)
        cv2.imwrite(f"{output_base}_overlay.jpg", overlay_image)
        
        # 保存方向分析结果
        cv2.imwrite(f"{output_base}_direction_color.jpg", direction_result['direction_color_image'])
        cv2.imwrite(f"{output_base}_direction_boundary.jpg", direction_result['boundary_image'])
        cv2.imwrite(f"{output_base}_direction_mask.jpg", direction_result['mask_image'])
        
        print(f"\n输出文件：")
        print(f"  平移检测: {output_base}_motion.jpg")
        print(f"  边界标注: {output_base}_boundary.jpg")
        print(f"  叠加可视化: {output_base}_overlay.jpg")
        print(f"  方向颜色图: {output_base}_direction_color.jpg")
        print(f"  方向边界图: {output_base}_direction_boundary.jpg")
        print(f"  方向掩码图: {output_base}_direction_mask.jpg")
        
        print("\n" + "-" * 60)
        print("注意：此版本仍在开发中，结果仅供参考")
        print("=" * 60)

else:

    print("错误：无效的版本选择，请输入1-3。")
