#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ui.styles.ui_styles import ModernStyle
from ui.components.ui_components import ScrollableFrame, CollapsibleFrame
from ui.views.ui_main import UIMain
from ui.views.ui_tabs import UITabs
from ui.views.ui_updates import UIUpdates
from ai.ai_display_manager import AIDisplayManager
from managers.data_manager import DataManager
from ai.ai_manager import AIManager
from managers.analysis_manager import AnalysisManager
from managers.prediction_manager import PredictionManager
from managers.backtest_manager import BacktestManager
from core.utils import UtilsMixin
import tkinter as tk
from managers.investment_manager import InvestmentManager
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import os
from datetime import datetime
import threading
import webbrowser
from pathlib import Path
import math
from typing import List
from threading import Lock
try:
    import sklearn
    print(f"sklearn导入成功，版本: {sklearn.__version__}")
    
    # 改为导入新的终极AI引擎
    from ai.local_ai_engine import UltimateOnlineAIEngine
    print("UltimateOnlineAIEngine导入成功")
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"终极在线AI引擎不可用，错误: {e}")
    print("请检查以下依赖包是否已安装：")
    print("1. pip install scikit-learn")
    print("2. pip install numpy")
    print("3. pip install torch (可选，用于深度学习)")
    print("4. pip install river (可选，用于在线学习)")
    import traceback
    traceback.print_exc()
except Exception as e:
    AI_AVAILABLE = False
    print(f"AI引擎初始化时出现异常: {e}")
    import traceback
    traceback.print_exc()

# 尝试导入matplotlib，如果失败则给出友好提示
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    import numpy as np
    import matplotlib
    
    # 配置matplotlib支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，图表功能将不可用")

class TailNumberAnalyzer(UtilsMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("天天发大财")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # 初始化路径相关属性
        self.data_file = None
        self.backup_file = None
        
        # 初始化数据管理器（必须在create_top_toolbar之前）
        self.data_manager = DataManager(self)

        # 初始化其他管理器
        self.ai_manager = AIManager(self)
        self.analysis_manager = AnalysisManager(self)
        self.prediction_manager = PredictionManager(self)
        self.backtest_manager = BacktestManager(self)
        self.investment_manager = InvestmentManager(self)

        # 初始化AI显示管理器（在UI组件之前）
        self.ai_display_manager = AIDisplayManager(self)

        # 初始化UI组件
        self.ui_main = UIMain(self)
        self.ui_tabs = UITabs(self)
        self.ui_updates = UIUpdates(self)
    
        # 设置数据文件路径（使用用户文档目录）
        self.data_manager.setup_data_path()

        # 初始化投资相关的UI组件变量
        self.investment_history_tree = None
        self.weight_pool_history_tree = None
        self.current_investment_status_label = None
        self.current_investment_details_label = None
        self.investment_fig = None
        self.investment_canvas = None
        self.investment_chart_widget = None
        self.model_investment_detail_tree = None
        
        # 初始化投资指标Label变量
        self.investment_pool_size_label = None
        self.investment_total_rounds_label = None
        self.investment_total_rewards_label = None
        self.investment_total_penalties_label = None

        # 初始化图表相关的IntVar变量（如果还没有创建的话）
        if not hasattr(self, 'display_periods'):
            self.display_periods = None  # 将在UI创建时初始化
        if not hasattr(self, 'advanced_analysis_periods'):
            self.advanced_analysis_periods = None  # 将在UI创建时初始化

        # 创建顶部工具栏
        self.create_top_toolbar()
        
        # 初始化AI预测结果
        self.ai_prediction_result = None

        # 示例数据
        self.example_data = """32,47,21,41,07,04,28
30,24,20,21,32,38,47
18,15,21,11,40,29,08
12,48,16,44,35,21,32
11,27,23,03,20,07,29
01,37,24,49,45,11,22
21,42,20,10,22,23,49
34,36,19,21,49,47,14
26,35,32,22,23,11,02
36,17,27,20,11,48,21"""
        
        # 初始化图表相关属性
        self.chart_cols_per_page = 10  # 每页显示的列数
        self.chart_data = []  # 图表数据
        self.chart_start_index = 0  # 图表开始显示的索引
        # display_periods 将在UI初始化时创建
        
        # advanced_analysis_periods 将在UI初始化时创建
        
        # 初始化滑块变量（将在UI创建时初始化）
        self.chart_slider_var = None
        self.chart_info_var = None
        
        # 移除表格滑块限制，让表格显示完整期数
        
        # 初始化计分板数据
        self.scoreboard_stats = {
            'totalPredictions': 0,
            'correctPredictions': 0,
            'incorrectPredictions': 0,
            'currentCorrectStreak': 0,
            'currentIncorrectStreak': 0,
            'maxCorrectStreak': 0,
            'maxIncorrectStreak': 0,
            'lastPredictionCorrect': None,
            'currentIncorrectStreak_beforeCorrect': 0,
            'correctAfterIncorrect': {},
            'incorrectAfterCorrect': {}
        }
        
        # 初始化预测数据
        self.prediction_data = {
            'hasPendingPrediction': False,
            'predictedResult': None,  # 0-9尾数
            'predictionTime': None,
            'predictionHistory': []  # 存储历史预测记录
        }
        
        # 初始化回测数据
        self.backtest_data = {
            'test_data_lines': [],  # 待回测的数据
            'current_test_index': 0,  # 当前回测索引
            'total_test_count': 0,  # 总回测数量
            'is_running': False,  # 是否正在回测
            'results': []  # 回测结果记录
        }

        # 初始化高级分析数据
        self.advanced_analysis_data = {
            'strategy_analysis': {},
            'interval_analysis': {},
            'distribution_analysis': {},
        }
        
        # 确保所有必要的目录在UI初始化前都已创建
        if hasattr(self, 'data_file') and self.data_file:
            try:
                # 再次验证AI目录结构
                ai_data_dir = self.data_file.parent / "ultimate_ai_learning_data"
                if not ai_data_dir.exists():
                    self.data_manager.ensure_ai_directories(self.data_file.parent)
            except Exception as ui_prep_error:
                pass
        
        # 初始化AI引擎（在UI设置之前）
        if self.ai_manager.initialize_ai_engine():
            self.local_ai = self.ai_manager.local_ai
            self.ai_engine = self.ai_manager.ai_engine
            
            # 建立AI引擎对主应用的引用，用于数据质量分析
            if self.local_ai:
                self.local_ai._main_app_ref = self
        else:
            self.local_ai = None
            self.ai_engine = None

        # 设置UI - 在UIMain的setup_ui中会调用UITabs的setup_tabs
        self.ui_main.setup_ui()
        
        # 设置窗口样式
        self.ui_main.setup_window_style()

        # 初始化预处理数据相关属性
        self.preprocessed_data = {
            'features_matrix': {},           # 预计算的特征矩阵
            'tail_statistics': {},           # 尾数统计信息
            'sliding_windows': {},           # 滑动窗口统计
            'metadata': {
                'last_preprocessed_period_count': 0,
                'last_update_time': None,
                'data_hash': None,
                'version': '1.0'
            }
        }
        # 添加预处理数据的线程锁
        self.preprocessed_data_lock = Lock()

        # 将预处理文件保存在与数据文件相同的目录
        try:
            # 使用与主数据文件相同的目录
            self.preprocessed_file = self.data_file.parent / "preprocessed_data.json"
            print(f"预处理数据文件路径: {self.preprocessed_file}")
        except Exception as e:
            # 如果获取失败，使用当前工作目录
            self.preprocessed_file = Path("preprocessed_data.json")
            print(f"预处理数据文件路径设置失败，使用当前目录: {e}")
        
        self.preprocessing_enabled = True

        # 应用现代化样式
        ModernStyle.setup_styles()

        # 修复matplotlib兼容性
        if MATPLOTLIB_AVAILABLE:
            ModernStyle.fix_matplotlib_compatibility()

        self.data_manager.load_data()
        self.ui_updates.update_scoreboard()

        def final_check_ai_learning_data():
            # 首先检查UI组件是否存在
            if hasattr(self, 'ai_display_manager') and self.ai_display_manager is not None:
                self.ai_display_manager.check_ai_ui_components_status()

            if AI_AVAILABLE and hasattr(self, 'local_ai') and self.local_ai is not None:
                try:
                    # 强制刷新学习进展显示
                    if hasattr(self, 'ui_updates') and self.ui_updates is not None:
                        self.ui_updates.update_learning_progress_display()
                except Exception as e:
                    pass

            # 再次尝试更新，确保数据正确显示
            if hasattr(self, 'ai_display_manager') and self.ai_display_manager is not None:
                self.root.after(2000, lambda: self.ai_display_manager.force_update_all_ai_displays())

        self.root.after(5000, final_check_ai_learning_data)  # 延长到5秒，确保UI完全初始化
        
        # 添加自动保存提示
        self.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 应用启动，数据已加载")

        # 绑定窗口大小变化事件
        self.root.bind('<Configure>', self.on_window_resize)
        
        # AI状态变量
        self.ai_prediction_result = None
        
        # 初始化投资相关变量
        self.ai_engine = None  # 投资引擎引用，指向local_ai

    def force_update_all_displays(self):
        """强制更新所有显示组件（简化版，避免循环调用）"""
        try:
            print("🔄 开始强制更新显示组件")
            
            # 添加调用保护
            if hasattr(self, '_force_updating') and self._force_updating:
                return
            
            self._force_updating = True
            
            try:
                # 更新基础UI组件（不使用定时器）
                if hasattr(self, 'ui_updates') and self.ui_updates is not None:
                    self.ui_updates.update_scoreboard()
                    self.ui_updates.update_prediction_status()
                    self.ui_updates.update_ai_status_display()
                
                # 更新投资详情（只进行一次）
                if hasattr(self, 'investment_manager') and self.investment_manager is not None:
                    self.investment_manager.refresh_investment_details()
                
                print("✅ 强制更新显示组件完成")
                
            finally:
                self._force_updating = False
                
        except Exception as e:
            self._force_updating = False
            print(f"❌ 强制更新显示组件失败: {e}")

    def mark_data_changed(self):
        """标记数据已更改，需要保存"""
        try:
            if hasattr(self, 'save_status_label') and self.save_status_label:
                self.save_status_label.config(fg='#ffc107')  # 黄色表示未保存
            if hasattr(self, 'save_status_text') and self.save_status_text:
                self.save_status_text.config(text="数据未保存")
        except Exception as e:
            print(f"❌ 标记数据更改状态失败: {e}")

    def mark_data_saved(self):
        """标记数据已保存"""
        try:
            if hasattr(self, 'save_status_label') and self.save_status_label:
                self.save_status_label.config(fg='#28a745')  # 绿色表示已保存
            if hasattr(self, 'save_status_text') and self.save_status_text:
                self.save_status_text.config(text="数据已保存")
        except Exception as e:
            print(f"❌ 标记数据保存状态失败: {e}")

    def check_ai_data_integrity(self):
        """检查AI数据完整性的代理方法"""
        try:
            if hasattr(self, 'ai_manager') and self.ai_manager is not None:
                self.ai_manager.check_ai_data_integrity()
        except Exception as e:
            print(f"❌ 检查AI数据完整性失败: {e}")
            
    def create_top_toolbar(self):
        """创建顶部工具栏"""
        # 创建工具栏框架
        self.toolbar_frame = tk.Frame(self.root, bg='white', height=40)
        self.toolbar_frame.pack(fill=tk.X, side=tk.TOP)
        self.toolbar_frame.pack_propagate(False)
        
        # 左侧按钮区域
        left_frame = tk.Frame(self.toolbar_frame, bg='white')
        left_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 保存数据按钮
        self.save_data_btn = tk.Button(
            left_frame,
            text="💾 保存数据",
            command=self.data_manager.manual_save_all_data,
            bg='#dc3545',  # 红色背景
            fg='white',
            font=ModernStyle.FONTS['default'],
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        self.save_data_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 加载数据按钮
        self.load_data_btn = tk.Button(
            left_frame,
            text="📂 加载数据",
            command=self.data_manager.manual_load_all_data,
            bg='#28a745',  # 绿色背景
            fg='white',
            font=ModernStyle.FONTS['default'],
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        self.load_data_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 右侧状态区域
        right_frame = tk.Frame(self.toolbar_frame, bg='white')
        right_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 数据保存状态指示器
        self.save_status_label = tk.Label(
            right_frame,
            text="●",
            fg='#ffc107',  # 黄色表示未保存
            bg='white',
            font=('Arial', 16, 'bold')
        )
        self.save_status_label.pack(side=tk.RIGHT, padx=(0, 5))
        
        self.save_status_text = tk.Label(
            right_frame,
            text="数据未保存",
            fg='black',
            bg='white',
            font=ModernStyle.FONTS['small']
        )
        self.save_status_text.pack(side=tk.RIGHT)
        
        # 绑定按钮悬停效果
        self.setup_button_hover_effects()
    
    def setup_button_hover_effects(self):
        """设置按钮悬停效果"""
        def on_enter_save(e):
            self.save_data_btn.config(bg='#c82333')
        
        def on_leave_save(e):
            self.save_data_btn.config(bg='#dc3545')
        
        def on_enter_load(e):
            self.load_data_btn.config(bg='#218838')
        
        def on_leave_load(e):
            self.load_data_btn.config(bg='#28a745')
        
        self.save_data_btn.bind("<Enter>", on_enter_save)
        self.save_data_btn.bind("<Leave>", on_leave_save)
        self.load_data_btn.bind("<Enter>", on_enter_load)
        self.load_data_btn.bind("<Leave>", on_leave_load)

    def get_drift_count(self):
        """获取漂移次数 - 代理到 ai_display_manager"""
        return self.ai_display_manager.get_drift_count()
    
def main():
    """主函数"""
    root = tk.Tk()
    app = TailNumberAnalyzer(root)
    
    # 设置窗口关闭事件
    def on_closing():
        try:
            app.data_manager.save_data()
            print("数据已保存")
        except Exception as e:
            print(f"关闭时保存数据失败：{e}")
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 启动应用
    root.mainloop()

if __name__ == "__main__":
    main()