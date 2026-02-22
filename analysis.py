import cv2
import numpy as np
from scipy import ndimage
import warnings
import random
import time

warnings.filterwarnings('ignore')

class SimpleVideoNoiseAnalyzer:
    """简化版视频噪声分析器"""
    
    def __init__(self, video_path: str, num_samples: int = 5):
        """
        初始化视频噪声分析器
        
        Args:
            video_path: 视频文件路径
            num_samples: 随机采样帧数
        """
        self.video_path = video_path
        self.num_samples = num_samples
        self.grain_sizes = []
        
    def analyze_video(self) -> float:
        """分析视频噪声，返回估计的颗粒尺寸（像素）"""
        print(f"正在分析视频: {self.video_path}")
        
        # 获取视频信息
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("错误: 无法打开视频文件")
            return -1.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频信息: {total_frames}帧, {fps:.1f} FPS")
        
        # 随机选择帧号
        if self.num_samples >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(random.sample(range(total_frames), self.num_samples))
        
        print(f"随机抽取 {len(frame_indices)} 帧进行分析...")
        
        # 分析每一帧
        for i, frame_idx in enumerate(frame_indices):
            print(f"分析第 {i+1}/{len(frame_indices)} 帧 (帧号: {frame_idx})")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 转换为灰度图
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                grain_size = self._analyze_frame(gray_frame)
                
                if grain_size > 0:
                    self.grain_sizes.append(grain_size)
                    print(f"  估计颗粒尺寸: {grain_size:.2f} 像素")
                else:
                    print("  分析失败")
            else:
                print(f"  无法读取帧 {frame_idx}")
        
        cap.release()
        
        # 计算最终结果
        if self.grain_sizes:
            final_size = np.median(self.grain_sizes)
            self._print_final_result(final_size)
            return float(final_size)
        else:
            print("错误: 无法从视频中分析噪声")
            return -1.0
    
    def _analyze_frame(self, frame: np.ndarray) -> float:
        """分析单帧图像，返回颗粒尺寸估计"""
        estimates = []
        
        # 方法1: 自相关分析
        size1 = self._autocorrelation_analysis(frame)
        if size1 > 0:
            estimates.append(size1)
        
        # 方法2: 局部方差分析
        size2 = self._local_variance_analysis(frame)
        if size2 > 0:
            estimates.append(size2)
        
        # 方法3: 功率谱分析
        size3 = self._power_spectrum_analysis(frame)
        if size3 > 0:
            estimates.append(size3)
        
        # 方法4: 梯度分析
        size4 = self._gradient_analysis(frame)
        if size4 > 0:
            estimates.append(size4)
        
        if estimates:
            # 使用中位数减少异常值影响
            return np.median(estimates)
        return 0.0
    
    def _autocorrelation_analysis(self, image: np.ndarray) -> float:
        """自相关函数分析"""
        try:
            # 归一化
            img_norm = (image - np.mean(image)) / (np.std(image) + 1e-10)
            
            # 计算自相关函数
            autocorr = np.fft.ifft2(np.abs(np.fft.fft2(img_norm))**2).real
            autocorr = np.fft.fftshift(autocorr)
            autocorr_norm = autocorr / (np.max(autocorr) + 1e-10)
            
            # 提取中心剖面
            center = np.array(autocorr_norm.shape) // 2
            profile = autocorr_norm[center[0], center[1]:]
            
            # 找到自相关下降到1/e的位置
            target_value = 1 / np.e
            idx = np.where(profile < target_value)[0]
            
            if len(idx) > 0:
                return float(idx[0])
        except:
            pass
        
        return 0.0
    
    def _local_variance_analysis(self, image: np.ndarray) -> float:
        """局部方差分析"""
        try:
            window_sizes = [2, 3, 4]
            estimates = []
            
            for ws in window_sizes:
                kernel = np.ones((ws, ws), dtype=np.float32) / (ws ** 2)
                local_mean = cv2.filter2D(image, -1, kernel)
                local_var = cv2.filter2D((image - local_mean) ** 2, -1, kernel)
                
                var_mean = np.mean(local_var)
                var_std = np.std(local_var)
                
                if var_std > 1e-10:
                    feature_ratio = var_mean / var_std
                    estimate = ws * np.log1p(feature_ratio)
                    estimates.append(estimate)
            
            if estimates:
                return float(np.median(estimates))
        except:
            pass
        
        return 0.0
    
    def _power_spectrum_analysis(self, image: np.ndarray) -> float:
        """功率谱分析"""
        try:
            # 计算功率谱
            img_zero_mean = image - np.mean(image)
            f_transform = np.fft.fft2(img_zero_mean)
            f_shift = np.fft.fftshift(f_transform)
            power_spectrum = np.abs(f_shift) ** 2
            
            # 计算径向平均功率谱
            center = np.array(power_spectrum.shape) // 2
            y, x = np.indices(power_spectrum.shape)
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            r_int = r.astype(int)
            
            radial_profile = ndimage.mean(power_spectrum, labels=r_int, 
                                         index=np.arange(0, int(np.max(r_int)) + 1))
            
            # 找到功率下降到一半的半径
            if len(radial_profile) > 10:
                peak_power = np.max(radial_profile[:10])
                half_power = peak_power / 2
                
                idx = np.where(radial_profile < half_power)[0]
                if len(idx) > 0 and idx[0] > 0:
                    characteristic_size = image.shape[0] / (2 * idx[0])
                    return float(characteristic_size)
        except:
            pass
        
        return 0.0
    
    def _gradient_analysis(self, image: np.ndarray) -> float:
        """梯度分析"""
        try:
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            grad_mean = np.mean(gradient_magnitude)
            grad_std = np.std(gradient_magnitude)
            
            if grad_std > 1e-10 and grad_mean > 1e-10:
                gradient_variation = grad_std / grad_mean
                characteristic_size = 2.0 / gradient_variation
                return float(np.clip(characteristic_size, 0.5, 10.0))
        except:
            pass
        
        return 0.0
    
    def _print_final_result(self, grain_size: float):
        """打印最终结果"""
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        
        if self.grain_sizes:
            print(f"分析帧数: {len(self.grain_sizes)}")
            print(f"颗粒尺寸范围: {min(self.grain_sizes):.2f} - {max(self.grain_sizes):.2f} 像素")
            print(f"颗粒尺寸中位数: {grain_size:.2f} 像素")
        
        print("\n最终判定结果:")
        print("-" * 40)
        
        if grain_size < 1.2:
            print(f"噪声类型: 极细颗粒高斯噪声")
            print(f"特征描述: 噪声颗粒非常细小，接近白噪声特性")
        elif grain_size < 2.0:
            print(f"噪声类型: 细颗粒高斯噪声")
            print(f"特征描述: 噪声颗粒细小，纹理细腻")
        elif grain_size < 3.5:
            print(f"噪声类型: 中等颗粒高斯噪声")
            print(f"特征描述: 噪声颗粒大小适中")
        elif grain_size < 5.0:
            print(f"噪声类型: 粗颗粒高斯噪声")
            print(f"特征描述: 噪声颗粒较粗，纹理明显")
        else:
            print(f"噪声类型: 极粗颗粒高斯噪声")
            print(f"特征描述: 噪声颗粒非常粗大")
        
        print("-" * 40)
        print(f"最小像素块大小: {grain_size:.2f} 像素")
        print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("视频高斯噪声分析系统")
    print("=" * 60)
    
    # 获取用户输入的视频文件路径
    video_path = input("请输入视频文件路径: ").strip()
    
    if not video_path:
        print("错误: 请输入有效的视频文件路径")
        return
    
    # 创建分析器
    analyzer = SimpleVideoNoiseAnalyzer(video_path, num_samples=5)
    
    # 执行分析
    start_time = time.time()
    grain_size = analyzer.analyze_video()
    elapsed_time = time.time() - start_time
    
    print(f"\n分析耗时: {elapsed_time:.1f} 秒")


def quick_analysis():
    """快速分析函数"""
    print("快速视频噪声分析")
    print("-" * 40)
    
    video_path = input("请输入视频文件路径: ").strip()
    
    if not video_path:
        print("错误: 请输入有效的视频文件路径")
        return
    
    num_samples = input("请输入采样帧数 (默认5帧): ").strip()
    if num_samples:
        try:
            num_samples = int(num_samples)
            if num_samples < 1:
                print("警告: 采样帧数至少为1，使用默认值5")
                num_samples = 5
        except:
            print("警告: 输入无效，使用默认值5")
            num_samples = 5
    else:
        num_samples = 5
    
    analyzer = SimpleVideoNoiseAnalyzer(video_path, num_samples=num_samples)
    grain_size = analyzer.analyze_video()
    
    return grain_size


if __name__ == "__main__":
    # 直接运行主函数
    main()
    
    # 或者运行快速分析
    # quick_analysis()

