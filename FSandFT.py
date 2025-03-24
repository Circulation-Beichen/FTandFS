import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import sympy as sp
from scipy import signal
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import matplotlib as mpl
import platform

# 配置中文字体
if platform.system() == 'Windows':
    # Windows系统常见中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong']
elif platform.system() == 'Darwin':
    # macOS系统常见中文字体
    font_list = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
else:
    # Linux系统常见中文字体
    font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']

# 增大matplotlib字体尺寸
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.size'] = 24  # 增大基础字体大小
plt.rcParams['axes.titlesize'] = 28  # 增大标题字体大小
plt.rcParams['axes.labelsize'] = 24  # 增大轴标签字体大小
plt.rcParams['xtick.labelsize'] = 20  # 增大x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 20  # 增大y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 24  # 增大图例字体大小
mpl.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif族字体，会从font.sans-serif中按顺序查找可用字体

class FourierDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("傅里叶级数与傅里叶变换演示")
        self.root.geometry("1400x900")  # 增大窗口尺寸
        
        # 设置全局字体大小
        self.style = ttk.Style()
        self.style.configure('.', font=('TkDefaultFont', 20))  # 设置默认字体大小
        self.style.configure('TLabel', font=('TkDefaultFont', 20))
        self.style.configure('TButton', font=('TkDefaultFont', 20))
        self.style.configure('TRadiobutton', font=('TkDefaultFont', 20))
        self.style.configure('TLabelframe.Label', font=('TkDefaultFont', 24))
        
        # 创建变量
        self.function_var = tk.StringVar(value="sin(2*pi*x)")
        self.transform_type = tk.StringVar(value="傅里叶变换")
        self.x_min = tk.DoubleVar(value=-10)
        self.x_max = tk.DoubleVar(value=10)
        self.samples = tk.IntVar(value=1000)
        
        # 创建UI
        self.create_ui()
        
        # 初始绘图
        self.update_plots()
        
    def create_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 控制区域
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="20")
        control_frame.pack(fill=tk.X, pady=15)
        
        # 函数输入
        ttk.Label(control_frame, text="输入函数:").grid(row=0, column=0, sticky=tk.W, pady=15)
        function_entry = ttk.Entry(control_frame, textvariable=self.function_var, width=30, font=('TkDefaultFont', 20))
        function_entry.grid(row=0, column=1, sticky=tk.W, pady=15)
        
        # 快捷按钮区域
        quick_buttons_frame = ttk.Frame(control_frame)
        quick_buttons_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=15)
        
        # 预设函数按钮
        presets = [
            ("δ函数", "dirac(x)"),
            ("正弦", "sin(2*pi*x)"),
            ("余弦", "cos(2*pi*x)"),
            ("指数", "exp(-x**2)"),
            ("方波", "square(x)"),
            ("三角波", "sawtooth(x)"),
            ("常数", "1"),
        ]
        
        # 将预设按钮分成两行显示，每行最多4个按钮
        row_count = 0
        col_count = 0
        max_buttons_per_row = 4
        
        for name, func in presets:
            ttk.Button(
                quick_buttons_frame, 
                text=name, 
                command=lambda f=func: self.set_function(f),
                width=10  # 增加按钮宽度
            ).grid(row=row_count, column=col_count, padx=12, pady=8)
            
            col_count += 1
            if col_count >= max_buttons_per_row:
                col_count = 0
                row_count += 1
            
        # 参数设置
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=15)
        
        ttk.Label(param_frame, text="x最小值:").grid(row=0, column=0, padx=12)
        ttk.Entry(param_frame, textvariable=self.x_min, width=6, font=('TkDefaultFont', 20)).grid(row=0, column=1, padx=12)
        
        ttk.Label(param_frame, text="x最大值:").grid(row=0, column=2, padx=12)
        ttk.Entry(param_frame, textvariable=self.x_max, width=6, font=('TkDefaultFont', 20)).grid(row=0, column=3, padx=12)
        
        ttk.Label(param_frame, text="采样点数:").grid(row=0, column=4, padx=12)
        ttk.Entry(param_frame, textvariable=self.samples, width=8, font=('TkDefaultFont', 20)).grid(row=0, column=5, padx=12)
        
        # 变换类型选择
        transform_frame = ttk.Frame(control_frame)
        transform_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=15)
        
        ttk.Radiobutton(
            transform_frame, 
            text="傅里叶变换", 
            variable=self.transform_type, 
            value="傅里叶变换"
        ).grid(row=0, column=0, padx=15)
        
        ttk.Radiobutton(
            transform_frame, 
            text="逆傅里叶变换", 
            variable=self.transform_type, 
            value="逆傅里叶变换"
        ).grid(row=0, column=1, padx=15)
        
        # 计算按钮
        ttk.Button(
            control_frame, 
            text="计算", 
            command=self.update_plots,
            width=15  # 增加按钮宽度
        ).grid(row=5, column=0, columnspan=2, pady=15)
        
        # 图表区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # 创建图表 - 增大图表尺寸
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.tight_layout(pad=5.0)  # 增大填充以减少重叠
        
        # 在Tkinter窗口中嵌入Matplotlib图表
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def set_function(self, func_str):
        self.function_var.set(func_str)
        self.update_plots()
    
    def parse_function(self, func_str, x):
        # 定义特殊函数
        def dirac(x):
            result = np.zeros_like(x)
            center_idx = np.argmin(np.abs(x))
            result[center_idx] = 1.0
            return result
            
        def square(x):
            return signal.square(2 * np.pi * x)
            
        def sawtooth(x):
            return signal.sawtooth(2 * np.pi * x)
            
        # 安全的函数求值
        try:
            # 空字符串或无效输入处理
            if not func_str or func_str.isspace():
                return np.zeros_like(x)
            
            # 定义可用的数学函数和常量
            safe_dict = {
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'sqrt': np.sqrt,
                'pi': np.pi,
                'abs': np.abs,
                'dirac': dirac,
                'square': square,
                'sawtooth': sawtooth,
                'x': x
            }
            
            # 求值函数
            result = eval(func_str, {"__builtins__": {}}, safe_dict)
            
            # 检查结果是否有效
            if np.isscalar(result):
                # 如果结果是标量，转换为数组
                return np.full_like(x, result)
            elif isinstance(result, np.ndarray) and result.size == x.size:
                # 如果结果是适合的数组，直接返回
                return result
            else:
                # 其他情况，返回零数组
                print(f"函数计算结果形状不匹配: {result}")
                return np.zeros_like(x)
            
        except Exception as e:
            # 详细打印错误信息以便调试
            print(f"函数求值错误: {e}, 函数表达式: '{func_str}'")
            return np.zeros_like(x)
    
    def compute_fourier_transform(self, f, x, is_inverse=False):
        # 计算采样间隔
        dx = x[1] - x[0]
        
        # 确保f是numpy数组且形状正确
        f = np.asarray(f)
        
        if not is_inverse:
            # 正向傅里叶变换
            if f.size > 0:  # 确保输入数组不为空
                F = fftshift(fft(f) * dx)
                freq = fftshift(fftfreq(len(x), dx))
                return freq, F
            else:
                # 处理空数组情况
                return np.array([]), np.array([])
        else:
            # 逆傅里叶变换
            if f.size > 0:  # 确保输入数组不为空
                F = ifft(ifftshift(f)) * len(x) * dx
                return x, F
            else:
                # 处理空数组情况
                return x, np.zeros_like(x)
    
    def update_plots(self):
        try:
            # 清除当前图表
            self.ax1.clear()
            self.ax2.clear()
            
            # 获取参数
            x_min = self.x_min.get()
            x_max = self.x_max.get()
            samples = self.samples.get()
            func_str = self.function_var.get()
            is_inverse = (self.transform_type.get() == "逆傅里叶变换")
            
            # 参数验证
            if x_min >= x_max:
                self.show_error("错误", "x最小值必须小于x最大值")
                return
            
            if samples <= 0:
                self.show_error("错误", "采样点数必须大于0")
                return
            
            # 限制采样点数以避免性能问题
            if samples > 10000:
                self.show_error("警告", f"采样点数过大 ({samples})，已限制为10000")
                samples = 10000
                self.samples.set(10000)
            
            # 生成x轴数据
            x = np.linspace(x_min, x_max, samples)
            
            # 计算函数值
            f = self.parse_function(func_str, x)
            
            # 检查函数值是否有效
            if np.all(f == 0) and func_str != "0" and func_str != "0*x":
                self.show_warning("警告", "函数求值结果全为零，请检查函数表达式")
            
            # 计算傅里叶变换或逆变换
            freq, F = self.compute_fourier_transform(f, x, is_inverse)
            
            # 检查结果是否有效
            if freq.size == 0 or F.size == 0:
                self.show_error("错误", "傅里叶变换计算失败，请检查输入函数")
                return
            
            # 绘制原始函数
            self.ax1.plot(x, np.real(f))
            self.ax1.set_title("原始函数" if not is_inverse else "傅里叶变换")
            self.ax1.set_xlabel("x" if not is_inverse else "频率 (Hz)")
            self.ax1.set_ylabel("f(x)")
            self.ax1.grid(True)
            
            # 绘制傅里叶变换结果
            if not is_inverse:
                # 为了更好的可视化，只显示振幅
                self.ax2.plot(freq, np.abs(F))
                self.ax2.set_title("傅里叶变换 (振幅谱)")
                self.ax2.set_xlabel("频率 (Hz)")
                self.ax2.set_ylabel("|F(ω)|")
            else:
                # 显示逆变换的实部
                self.ax2.plot(x, np.real(F))
                self.ax2.set_title("逆傅里叶变换")
                self.ax2.set_xlabel("x")
                self.ax2.set_ylabel("f(x)")
            
            self.ax2.grid(True)
            
            # 更新画布
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            # 捕获所有异常并显示
            import traceback
            error_msg = f"发生错误: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.show_error("错误", f"更新图表时出错: {str(e)}")
    
    def show_error(self, title, message):
        """显示错误对话框"""
        import tkinter.messagebox as messagebox
        messagebox.showerror(title, message)
    
    def show_warning(self, title, message):
        """显示警告对话框"""
        import tkinter.messagebox as messagebox
        messagebox.showwarning(title, message)

    def run(self):
        self.root.mainloop()

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = FourierDemoApp(root)

    
    app.run()
