#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import json
import threading
from datetime import datetime
import sqlite3

try:
    import sklearn
    print(f"sklearn导入成功，版本: {sklearn.__version__}")
    
    # 改为导入新的终极AI引擎
    import sys
    import os
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from .local_ai_engine import UltimateOnlineAIEngine
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

from ui.styles.ui_styles import ModernStyle

class AIManager:
    def __init__(self, main_app):
        self.main_app = main_app
        self.local_ai = None
        self.ai_engine = None
        self.ai_prediction_result = None
        
    def initialize_ai_engine(self):
        """初始化AI引擎"""
        if not AI_AVAILABLE:
            self.local_ai = None
            print("⚠️ AI引擎不可用：未安装依赖库")
            return False
            
        try:
            # 确保数据文件路径已经设置
            if not hasattr(self.main_app, 'data_file') or not self.main_app.data_file:
                self.main_app.data_manager.setup_data_path()
            
            ai_data_dir = self.main_app.data_file.parent / "ultimate_ai_learning_data"
            
            # 再次确保AI数据目录存在并检查权限
            try:
                if not ai_data_dir.exists():
                    self.main_app.data_manager.ensure_ai_directories(self.main_app.data_file.parent)
                
                # 检查目录是否可写
                test_file = ai_data_dir / "test_write.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()  # 删除测试文件
                except Exception as write_error:
                    raise Exception(f"AI数据目录无写入权限: {write_error}")
                    
            except Exception as dir_error:
                raise Exception(f"无法创建或访问AI数据目录: {dir_error}")
            
            # 转换为绝对路径字符串，避免路径问题
            ai_data_dir_str = str(ai_data_dir.resolve())
            
            self.local_ai = UltimateOnlineAIEngine(ai_data_dir_str)
            # 设置主程序引用，让AI引擎能访问主程序数据
            self.local_ai._main_app_ref = self.main_app
            
            # 验证AI引擎是否真的可用
            if not self.local_ai.is_initialized:
                self.local_ai = None
                return False
            else:
                # 设置投资引擎引用
                self.ai_engine = self.local_ai
                return True
            
        except Exception as e:
            print(f"❌ AI引擎初始化失败: {e}")
            print("详细错误信息：")
            import traceback
            traceback.print_exc()
            
            # 如果是模型文件相关错误，提供更详细的诊断信息
            if "pytorch" in str(e).lower() or "pth" in str(e).lower():
                print("\n🔍 PyTorch模型文件诊断:")
                try:
                    ai_data_dir = self.main_app.data_file.parent / "ultimate_ai_learning_data"
                    
                    # 检查根目录的模型文件
                    root_models = list(ai_data_dir.glob("*.pth"))
                    if root_models:
                        print(f"  📁 根目录模型文件: {[f.name for f in root_models]}")
                    
                    # 检查deep_learning/models目录的文件
                    dl_dir = ai_data_dir / "deep_learning" / "models"
                    if dl_dir.exists():
                        dl_models = list(dl_dir.glob("*.pth"))
                        if dl_models:
                            print(f"  📁 deep_learning/models/: {[f.name for f in dl_models]}")
                        else:
                            print(f"  📁 deep_learning/models/: 空目录")
                    else:
                        print(f"  📁 deep_learning/models/: 目录不存在")
                        
                    print("  💡 建议: 重新运行程序，模型文件位置已自动修复")
                except Exception as diag_error:
                    print(f"  ❌ 诊断失败: {diag_error}")
            
            self.local_ai = None
            return False

    def train_local_ai(self):
        """训练本地AI模型"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn依赖包。")
            return
            
        if not self.main_app.chart_data:
            messagebox.showwarning("警告", "请先分析数据！")
            return
    
        # 显示训练进度
        progress_window = tk.Toplevel(self.main_app.root)
        progress_window.title("AI模型训练中...")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
    
        # 居中显示
        progress_window.transient(self.main_app.root)
        progress_window.grab_set()
    
        # 进度标签
        status_label = tk.Label(progress_window, text="正在训练AI模型，请稍候...", 
                           font=ModernStyle.FONTS['default'])
        status_label.pack(pady=30)
    
        # 进度条
        progress_bar = tk.ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=30, fill=tk.X)
        progress_bar.start()
    
        def training_thread():
            try:
                # 检查是否有足够的历史数据
                if not self.main_app.chart_data or len(self.main_app.chart_data) < 10:
                    def show_insufficient_data():
                        progress_bar.stop()
                        progress_window.destroy()
                        messagebox.showerror("数据不足", "需要至少10期历史数据才能进行训练！\n\n请先在统计分析页面分析更多历史数据。")
                
                    self.main_app.root.after(0, show_insufficient_data)
                    return
            
                # 更新进度显示
                def update_status(text):
                    try:
                        status_label.config(text=text)
                    except:
                        pass
            
                # 执行真正的训练过程
                self.main_app.root.after(0, lambda: update_status("正在准备训练数据..."))
            
                # 使用历史数据进行预训练
                self.main_app.root.after(0, lambda: update_status("正在执行批量预训练..."))
            
                # 调用批量预训练方法
                pretrain_result = self.local_ai.batch_pretrain(self.main_app.chart_data)
            
                if pretrain_result.get('success', False):
                    training_samples = pretrain_result.get('total_samples', 0)
                    successful_trainings = pretrain_result.get('successful_samples', 0)
                
                    self.main_app.root.after(0, lambda: update_status(f"预训练完成: {successful_trainings}/{training_samples} 成功"))
                else:
                    training_samples = len(self.main_app.chart_data)
                    successful_trainings = 0
                
                    self.main_app.root.after(0, lambda: update_status("预训练失败"))
            
                # 检查训练结果
                success = successful_trainings > 0 and self.local_ai.is_initialized
            
                # 在主线程中更新UI
                def update_ui():
                    progress_bar.stop()
                    progress_window.destroy()
                
                    if success:
                        success_message = f"🎉 AI模型训练成功！\n\n"
                        success_message += f"📊 训练统计：\n"
                        success_message += f"• 总训练样本：{training_samples} 个\n"
                        success_message += f"• 成功训练：{successful_trainings} 个\n"
                        success_message += f"• 成功率：{successful_trainings/training_samples*100:.1f}%\n\n"
                        success_message += f"🚀 您的个人AI助手已完成训练！\n\n"
                        success_message += f"✨ 训练完成的特性：\n"
                        success_message += f"• 🌳 随机森林算法\n"
                        success_message += f"• 🚄 梯度提升算法\n"
                        success_message += f"• 🧠 神经网络算法\n"
                        success_message += f"• 🎯 集成学习预测\n"
                        success_message += f"• 📊 丰富特征工程\n"
                        success_message += f"• 🎲 智能随机策略\n\n"
                        success_message += f"现在可以开始智能预测了！"
                    
                        messagebox.showinfo("训练成功", success_message)
                        self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI模型训练成功，使用了{training_samples}个样本，成功训练{successful_trainings}个")
                    
                        # 强制刷新状态显示
                        self.main_app.root.after(50, self.main_app.ui_updates.update_ai_status_display)
                        self.main_app.root.after(100, self.main_app.ui_updates.update_ai_progress_display)
                        # 强制更新所有AI显示
                        self.main_app.root.after(300, self.main_app.ai_display_manager.force_update_all_ai_displays)

                    else:
                        messagebox.showerror("失败", f"AI模型训练失败！\n\n统计信息：\n• 尝试训练样本：{training_samples}\n• 成功训练样本：{successful_trainings}\n\n建议：\n1. 确保有至少30期历史数据\n2. 检查数据格式是否正确\n3. 检查scikit-learn是否正确安装")
        
                self.main_app.root.after(0, update_ui)
        
            except Exception as e:
                def show_error():
                    progress_bar.stop()
                    progress_window.destroy()
                    messagebox.showerror("错误", f"训练过程出错：{str(e)}")
        
                self.main_app.root.after(0, show_error)
    
        # 启动训练线程
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def train_deep_learning(self):
        """深度学习模型训练"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn和PyTorch依赖包。")
            return
            
        if not self.main_app.chart_data:
            messagebox.showwarning("警告", "请先分析数据！")
            return
    
        # 检查PyTorch可用性
        try:
            import torch
            pytorch_available = True
        except ImportError:
            pytorch_available = False
        
        if not pytorch_available:
            messagebox.showerror("错误", "PyTorch不可用！\n\n请安装PyTorch：\npip install torch torchvision torchaudio")
            return
    
        # 显示训练进度
        progress_window = tk.Toplevel(self.main_app.root)
        progress_window.title("深度学习模型训练中...")
        progress_window.geometry("500x300")
        progress_window.resizable(False, False)
    
        # 居中显示
        progress_window.transient(self.main_app.root)
        progress_window.grab_set()
    
        # 进度标签
        status_label = tk.Label(progress_window, text="正在准备深度学习训练，请稍候...", 
                           font=ModernStyle.FONTS['default'])
        status_label.pack(pady=20)
        
        # 详细信息文本框
        info_text = scrolledtext.ScrolledText(progress_window, height=10, width=60)
        info_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
        # 进度条
        progress_bar = tk.ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=30, fill=tk.X)
        progress_bar.start()
    
        def training_thread():
            try:
                # 检查是否有足够的历史数据
                if not self.main_app.chart_data or len(self.main_app.chart_data) < 30:
                    def show_insufficient_data():
                        progress_bar.stop()
                        progress_window.destroy()
                        messagebox.showerror("数据不足", "深度学习训练需要至少30期历史数据！\n\n请先在统计分析页面分析更多历史数据。")
                
                    self.main_app.root.after(0, show_insufficient_data)
                    return
            
                # 更新进度显示
                def update_status(text):
                    try:
                        status_label.config(text=text)
                    except:
                        pass
                        
                def update_info(text):
                    try:
                        info_text.insert(tk.END, text + "\n")
                        info_text.see(tk.END)
                    except:
                        pass
            
                # 执行深度学习训练
                self.main_app.root.after(0, lambda: update_status("正在初始化深度学习模型..."))
                self.main_app.root.after(0, lambda: update_info("🤖 检查深度学习模块状态..."))
                
                # 确保PyTorch模型目录存在
                self.main_app.root.after(0, lambda: update_info("🔧 准备PyTorch模型目录..."))
                models_dir = self.main_app.data_manager.ensure_pytorch_model_directory()
                if not models_dir:
                    def show_dir_error():
                        progress_bar.stop()
                        progress_window.destroy()
                        messagebox.showerror("错误", "无法创建PyTorch模型目录！")
    
                    self.main_app.root.after(0, show_dir_error)
                    return
                
                # 检查深度学习模块
                if not hasattr(self.local_ai, 'deep_learning_module') or not self.local_ai.deep_learning_module:
                    def show_module_error():
                        progress_bar.stop()
                        progress_window.destroy()
                        messagebox.showerror("错误", "深度学习模块未初始化！\n\n请检查PyTorch安装或重新启动应用程序。")
                
                    self.main_app.root.after(0, show_module_error)
                    return
                
                self.main_app.root.after(0, lambda: update_info("✅ 深度学习模块检查通过"))
                self.main_app.root.after(0, lambda: update_info(f"📊 准备训练数据：{len(self.main_app.chart_data)} 期历史数据"))
                
                self.main_app.root.after(0, lambda: update_status("正在执行深度学习批量训练..."))
                self.main_app.root.after(0, lambda: update_info("🚀 开始训练LSTM和Transformer模型..."))
                
                # 将模型目录路径传递给AI引擎
                if hasattr(self.local_ai, 'deep_learning_module') and self.local_ai.deep_learning_module:
                    self.main_app.root.after(0, lambda: update_info("✓ 设置模型保存目录..."))
                    # 确保AI引擎知道模型目录
                    self.local_ai.ensure_pytorch_model_directory(models_dir)
    
                # 调用深度学习批量训练方法
                training_result = self.local_ai.deep_learning_batch_train(self.main_app.chart_data, epochs=50)
            
                if training_result.get('success', False):
                    training_details = training_result.get('training_details', {})
                    models_trained = training_result.get('models_trained', [])
                    
                    self.main_app.root.after(0, lambda: update_info("✅ 深度学习训练完成！"))
                    self.main_app.root.after(0, lambda: update_info(f"📚 已训练模型：{', '.join(models_trained)}"))
                    
                    # 显示训练结果详情
                    if 'results' in training_details:
                        for model_name, result in training_details['results'].items():
                            final_acc = result.get('final_val_accuracy', 0)
                            epochs_completed = result.get('epochs_completed', 0)
                            self.main_app.root.after(0, lambda m=model_name, a=final_acc, e=epochs_completed: 
                                          update_info(f"  🎯 {m.upper()}: 验证准确率 {a:.2%}, 训练轮数 {e}"))
                    
                    self.main_app.root.after(0, lambda: update_status("深度学习训练完成！"))
                else:
                    error_msg = training_result.get('message', '未知错误')
                    self.main_app.root.after(0, lambda: update_info(f"❌ 训练失败：{error_msg}"))
                    self.main_app.root.after(0, lambda: update_status("训练失败"))
            
                # 在主线程中更新UI
                def update_ui():
                    progress_bar.stop()
                    
                    # 添加关闭按钮
                    close_btn = tk.ttk.Button(progress_window, text="关闭", 
                                          command=progress_window.destroy,
                                          style='Primary.TButton')
                    close_btn.pack(pady=10)
                
                    if training_result.get('success', False):
                        success_message = f"🎉 深度学习训练成功！\n\n"
                        success_message += f"📊 训练统计：\n"
                        
                        training_details = training_result.get('training_details', {})
                        if 'train_samples' in training_details:
                            success_message += f"• 训练样本：{training_details['train_samples']} 个\n"
                            success_message += f"• 验证样本：{training_details['val_samples']} 个\n"
                        
                        models_trained = training_result.get('models_trained', [])
                        success_message += f"• 训练模型：{', '.join(models_trained)}\n\n"
                        success_message += f"🧠 深度学习特性：\n"
                        success_message += f"• 🔥 LSTM长短期记忆网络\n"
                        success_message += f"• 🚀 Transformer注意力机制\n"
                        success_message += f"• ⚡ GPU加速训练（如可用）\n"
                        success_message += f"• 📈 批量训练优化\n"
                        success_message += f"• 🎯 序列模式识别\n\n"
                        success_message += f"现在可以使用增强的AI预测功能！"
                    
                        self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 深度学习训练成功，训练了{', '.join(models_trained)}模型")
                    
                        # 强制刷新状态显示
                        self.main_app.root.after(50, self.main_app.ui_updates.update_ai_status_display)
                        self.main_app.root.after(100, self.main_app.ui_updates.update_ai_progress_display)
                        self.main_app.root.after(150, self.main_app.ui_updates.update_learning_progress_display)
                    else:
                        error_msg = training_result.get('message', '未知错误')
                        messagebox.showerror("失败", f"深度学习训练失败！\n\n错误信息：{error_msg}\n\n建议：\n1. 确保有至少30期历史数据\n2. 检查PyTorch是否正确安装\n3. 确保有足够的内存和计算资源")
        
                self.main_app.root.after(0, update_ui)
        
            except Exception as e:
                def show_error():
                    progress_bar.stop()
                    progress_window.destroy()
                    messagebox.showerror("错误", f"深度学习训练过程出错：{str(e)}")
        
                self.main_app.root.after(0, show_error)
    
        # 启动训练线程
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    def run_ai_prediction(self):
        """运行AI预测"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn依赖包。")
            return
            
        if not self.local_ai.is_initialized:
            result = messagebox.askyesno("模型未训练", 
                                   "AI模型尚未训练，是否现在开始训练？\n\n训练完成后即可开始智能预测。")
            if result:
                self.train_local_ai()
            return
    
        if not self.main_app.chart_data:
            messagebox.showwarning("警告", "请先在统计分析页面分析数据！")
            return
        
        try:
            # 添加数据诊断
            print(f"🔍 主程序chart_data数据量：{len(self.main_app.chart_data) if self.main_app.chart_data else 0}期")
            if self.main_app.chart_data and len(self.main_app.chart_data) > 0:
                print(f"🔍 最新5期数据示例：{self.main_app.chart_data[:5]}")
    
            # 执行AI预测
            prediction = self.local_ai.predict_online(self.main_app.chart_data)
        
            if prediction['success']:
                # 保存预测结果到ai_manager和主程序
                self.ai_prediction_result = prediction
                self.main_app.ai_prediction_result = prediction
                
                # 立即更新状态显示
                self.main_app.root.after(50, self.main_app.ui_updates.update_ai_status_display)
                self.main_app.root.after(100, self.main_app.ui_updates.update_ai_progress_display)
                
                # 强制更新所有AI学习相关显示
                self.main_app.root.after(140, self.main_app.ai_display_manager.force_update_all_ai_displays)
                
                # 最后更新AI预测结果显示
                self.main_app.root.after(200, self.main_app.ui_updates.update_ai_prediction_display)
                
                # 简化的投资详情更新（只进行一次）
                if hasattr(self.main_app, 'investment_manager') and self.main_app.investment_manager is not None:
                    print("🔄 AI预测完成，更新投资详情...")
                    self.main_app.root.after(300, self.main_app.investment_manager.refresh_investment_details)

                # 额外触发学习进展显示更新
                self.main_app.root.after(300, self.main_app.ui_updates.update_learning_progress_display)

                # 显示详细的预测结果
                recommended_tails = prediction['recommended_tails']
                confidence = prediction['confidence']

                # 记录AI预测结果保存日志
                self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI预测结果已保存：推荐尾数{recommended_tails}")
                
                result_message = "🤖 改进版AI预测完成！\n\n"
                result_message += f"🎯 最佳推荐尾数：{recommended_tails[0] if recommended_tails else '无'}\n"
                result_message += f"📊 预测置信度：{confidence:.1%}\n"
                result_message += f"🔥 精准单选推荐\n\n"
                result_message += f"💡 使用了多种算法：\n"
                result_message += f"• 随机森林\n• 梯度提升\n• 神经网络\n• 集成学习\n\n"
                result_message += f"请在统计分析页面添加最新开奖号码来验证预测结果。"
                
                messagebox.showinfo("AI预测完成", result_message)
            else:
                messagebox.showerror("预测失败", f"AI预测失败：{prediction['message']}")
            
        except Exception as e:
            messagebox.showerror("错误", f"AI预测过程出错：{str(e)}")

    def verify_ai_prediction(self, actual_tails):
        """验证AI预测结果"""
        # 检查是否有AI预测结果需要验证
        ai_prediction = None
    
        # 从ai_manager获取预测结果
        if hasattr(self, 'ai_prediction_result') and self.ai_prediction_result:
            ai_prediction = self.ai_prediction_result
        # 从主程序获取预测结果（备用）
        elif hasattr(self.main_app, 'ai_prediction_result') and self.main_app.ai_prediction_result:
            ai_prediction = self.main_app.ai_prediction_result
    
        if not ai_prediction:
            print("🔍 没有待验证的AI预测结果")
            return

        # 检查AI引擎是否可用
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            print("⚠️ AI引擎不可用，跳过学习")
            return

        predicted_tails = ai_prediction.get('recommended_tails', [])
        confidence = ai_prediction.get('confidence', 0.0)

        print(f"🔍 开始验证AI预测:")
        print(f"  预测尾数: {predicted_tails}")
        print(f"  实际尾数: {actual_tails}")
        print(f"  置信度: {confidence:.3f}")

        # 在线学习系统验证预测结果
        try:
            learn_result = self.local_ai.learn_online(self.main_app.chart_data, actual_tails)
            print(f"✅ 在线学习完成: {learn_result}")
        except Exception as e:
            print(f"❌ 在线学习失败: {e}")

        # 判断预测结果
        is_correct = any(tail in actual_tails for tail in predicted_tails)
        result_text = "✅正确" if is_correct else "❌错误"

        print(f"🎯 预测结果: {result_text}")

        # 添加历史记录
        self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI预测验证: 预测{predicted_tails}, 实际{actual_tails}, {result_text}")

        # 更新计分板统计
        if hasattr(self.main_app, 'prediction_manager'):
            self.main_app.prediction_manager.update_scoreboard_with_prediction(is_correct)
            # 更新计分板显示
            self.main_app.ui_updates.update_scoreboard()

        # 显示验证结果
        messagebox.showinfo("AI预测验证", 
                      f"🤖 AI预测验证完成\n\n预测尾数：{predicted_tails}\n实际尾数：{actual_tails}\n结果：{result_text}")

        # 清除预测结果（在显示结果之后）
        self.ai_prediction_result = None
        self.main_app.ai_prediction_result = None
        print("🔍 AI预测结果已清除")

        # 立即保存数据
        self.main_app.data_manager.save_data()

        # 延迟更新UI显示，确保数据已经处理完毕
        self.main_app.root.after(200, self.main_app.ui_updates.update_ai_status_display)
        self.main_app.root.after(300, self.main_app.ui_updates.update_ai_prediction_display)
        if hasattr(self.main_app.ui_updates, 'update_ai_progress_display'):
            self.main_app.root.after(400, self.main_app.ui_updates.update_ai_progress_display)
        
        # 额外触发ai_display_manager的强制更新
        if hasattr(self.main_app, 'ai_display_manager') and self.main_app.ai_display_manager is not None:
            self.main_app.root.after(800, self.main_app.ai_display_manager.force_update_all_ai_displays)
            self.main_app.root.after(2000, self.main_app.ai_display_manager.force_update_all_ai_displays)
        
        # 简化的投资详情更新（AI验证后）
        if hasattr(self.main_app, 'investment_manager') and self.main_app.investment_manager is not None:
            print("🔄 AI预测验证完成，更新投资详情...")
            self.main_app.root.after(500, self.main_app.investment_manager.refresh_investment_details)

    def show_ai_stats(self):
        """显示AI学习统计"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn依赖包。")
            return
            
        try:
            stats = self.local_ai.get_comprehensive_stats()
            
            stats_window = tk.Toplevel(self.main_app.root)
            stats_window.title("AI学习统计")
            stats_window.geometry("500x400")
            stats_window.resizable(False, False)
        
            # 居中显示
            stats_window.transient(self.main_app.root)
        
            # 统计信息显示
            basic_stats = stats.get('basic_stats', {})
            model_stats = stats.get('model_stats', {})
            data_count = len(self.main_app.chart_data) if hasattr(self.main_app, 'chart_data') and self.main_app.chart_data else 0
            
            stats_text = f"""
    🤖 终极在线AI学习统计报告

    📚 数据统计：
      • 历史数据量：{data_count} 期
      • 处理样本次数：{basic_stats.get('total_samples_seen', 0)} 次
      • 正确预测次数：{basic_stats.get('correct_predictions', 0)} 次

    📈 准确率统计：
      • 当前准确率：{basic_stats.get('current_accuracy', 0):.1%}
      • 模型状态：{'已初始化' if basic_stats.get('is_initialized', False) else '未初始化'}

    🎯 模型信息：
      • River模型：{model_stats.get('river_models', 0)} 个
      • SKM模型：{model_stats.get('sklearn_models', 0)} 个
      • 深度模型：{model_stats.get('deep_models', 0)} 个
      • 总模型数：{model_stats.get('total_models', 0)} 个
      • AI引擎：终极在线学习引擎
      • 学习模式：实时在线学习
    """
        
            # 添加数据质量分析显示
            data_quality = stats.get('data_quality_analysis', {})
            if 'randomness_analysis' in data_quality:
                randomness = data_quality['randomness_analysis']
                recommendations = data_quality['learning_recommendations']
                
                stats_text += f"""
                
📊 数据质量分析报告：
  • 样本数量：{randomness.get('sample_size', 0)} 期
  • 可预测性评分：{randomness.get('predictability_score', 0):.2f}/1.0
  • 适合机器学习：{'是' if data_quality.get('data_suitable_for_ml', False) else '否'}

🎲 随机性检验：
  • 频率分布：{randomness['randomness_tests']['frequency']['randomness_level']}随机性
  • 卡方统计量：{randomness['randomness_tests']['frequency']['chi_square']:.2f}
  • 连续性模式：{'发现异常模式' if randomness['randomness_tests']['continuity']['has_patterns'] else '正常随机'}

💡 学习建议："""
                
                for rec in recommendations:
                    stats_text += f"\n  {rec}"
        
            text_widget = scrolledtext.ScrolledText(stats_window, 
                                                   font=ModernStyle.FONTS['default'],
                                                   bg=ModernStyle.COLORS['surface'],
                                                   wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            text_widget.insert(tk.END, stats_text)
            text_widget.config(state=tk.DISABLED)
        
        except Exception as e:
            messagebox.showerror("错误", f"获取统计信息失败：{str(e)}")
    
    def reset_ai_model(self):
        """重置AI模型"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn依赖包。")
            return
        
        result = messagebox.askyesno("确认重置", 
                                   "⚠️ 确定要重置AI模型吗？\n\n这将删除：\n• 所有已训练的模型\n• 历史预测记录\n• 学习统计数据\n\n操作不可撤销！")
        if result:
            try:
                # 重置AI模型 - 但保留学习统计
                if hasattr(self, 'local_ai') and self.local_ai:
                    # 保存当前的学习统计
                    current_samples = self.local_ai.total_samples_seen
                    current_correct = self.local_ai.correct_predictions
                    print(f"🔄 重置前统计：样本数={current_samples}, 正确预测={current_correct}")
                    
                    # 备份数据库
                    import shutil
                    ai_data_dir = self.main_app.data_file.parent / "ultimate_ai_learning_data"
                    db_path = ai_data_dir / "ultimate_online_database.db"
                    if db_path.exists():
                        backup_path = ai_data_dir / f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                        shutil.copy2(db_path, backup_path)
                        print(f"📦 数据库已备份到：{backup_path}")
                
                # 调用AI引擎内部的重置方法来真正清空所有数据
                reset_success = self.local_ai.reset_model()

                if reset_success:
                    # 清除预测结果
                    self.ai_prediction_result = None
                    self.main_app.ai_prediction_result = None
    
                    # 重置主程序中的相关数据
                    if hasattr(self.main_app, 'chart_data'):
                        self.main_app.chart_data = []
    
                    # 更新显示
                    self.main_app.ui_updates.update_ai_status_display()
                    if hasattr(self.main_app.ui_updates, 'update_ai_progress_display'):
                        self.main_app.ui_updates.update_ai_progress_display()
                    if hasattr(self.main_app, 'ui_updates') and hasattr(self.main_app.ui_updates, 'update_learning_progress_display'):
                        self.main_app.ui_updates.update_learning_progress_display()
    
                    # 强制更新所有AI显示组件
                    if hasattr(self.main_app, 'ai_display_manager') and self.main_app.ai_display_manager is not None:
                        self.main_app.root.after(500, self.main_app.ai_display_manager.force_update_all_ai_displays)
    
                    messagebox.showinfo("重置成功", "🔄 AI模型已完全重置！\n\n• 所有学习数据已清除\n• 所有模型已重新初始化\n• 数据库记录已清空\n\n可以重新开始训练")
                    self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI模型已完全重置")
                else:
                    messagebox.showerror("重置失败", "AI模型重置失败！\n\n请检查控制台输出获取详细错误信息。")
                
            except Exception as e:
                messagebox.showerror("错误", f"重置AI模型失败：{str(e)}")

    def sync_ai_data_with_main_data(self):
        """同步AI引擎数据与主程序数据"""
        if not AI_AVAILABLE or not self.local_ai:
            return
        
        try:
            # 获取主程序中的实际数据量
            main_data_count = len(self.main_app.chart_data) if self.main_app.chart_data else 0
            
            # 获取AI引擎中的数据量
            ai_stats = self.local_ai.get_comprehensive_stats()
            ai_data_count = ai_stats['basic_stats'].get('total_samples_seen', 0)
            
            print(f"🔍 数据同步检查：主程序数据={main_data_count}期，AI数据={ai_data_count}个样本")
            
            # 如果数据不一致，重置AI引擎状态以匹配主程序
            if main_data_count != ai_data_count:
                print(f"⚠️ 检测到数据不一致，正在同步...")
                
                # 方案1：如果主程序数据更少，说明用户删除了数据，重置AI引擎
                if main_data_count < ai_data_count:
                    print(f"🔄 主程序数据较少，重置AI引擎状态")
                    self.reset_ai_to_match_main_data()
                
                # 方案2：如果主程序数据更多，更新AI引擎的计数
                elif main_data_count > ai_data_count:
                    print(f"📈 主程序数据较多，更新AI引擎计数")
                    self.update_ai_count_to_match_main_data()
                
                print(f"✅ 数据同步完成")
            else:
                print(f"✅ 数据已同步")
                
        except Exception as e:
            print(f"❌ 数据同步失败：{str(e)}")
    
    def reset_ai_to_match_main_data(self):
        """重置AI引擎以匹配主程序数据"""
        if not AI_AVAILABLE or not self.local_ai:
            return
        
        try:
            # 保存当前的AI预测结果，避免在数据同步时被清空
            saved_ai_prediction = self.ai_prediction_result
            
            # 重置AI引擎的计数器
            self.local_ai.total_samples_seen = 0
            self.local_ai.correct_predictions = 0
            
            # 恢复AI预测结果
            self.ai_prediction_result = saved_ai_prediction
            
            # 重置集成权重为初始状态 - 安全获取模型列表
            all_models = []
            
            # 安全获取river模型
            if hasattr(self.local_ai, 'river_models') and self.local_ai.river_models:
                all_models.extend(list(self.local_ai.river_models.keys()))
            
            # 安全获取sklearn模型
            if hasattr(self.local_ai, 'sklearn_models') and self.local_ai.sklearn_models:
                all_models.extend(list(self.local_ai.sklearn_models.keys()))
            
            # 安全获取深度学习模型
            if hasattr(self.local_ai, 'deep_models') and self.local_ai.deep_models:
                all_models.extend(list(self.local_ai.deep_models.keys()))
            
            # 重置集成权重
            if all_models and hasattr(self.local_ai, 'ensemble_weights'):
                initial_weight = 1.0 / len(all_models)
                for model_name in all_models:
                    if model_name in self.local_ai.ensemble_weights:
                        self.local_ai.ensemble_weights[model_name]['weight'] = initial_weight
                        self.local_ai.ensemble_weights[model_name]['performance_history'] = []
            
            # 清空数据库中的学习记录
            try:
                if hasattr(self.local_ai, 'db_path') and self.local_ai.db_path:
                    conn = sqlite3.connect(self.local_ai.db_path)
                    cursor = conn.cursor()
                    
                    # 检查表是否存在再进行清空操作
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='online_predictions'")
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM online_predictions")
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='individual_model_predictions'")
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM individual_model_predictions")
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='drift_events'")
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM drift_events")
                    
                    conn.commit()
                    conn.close()
                    
                    print(f"🗑️ 已清空AI数据库记录")
                    
            except Exception as db_e:
                print(f"⚠️ 清空数据库失败：{str(db_e)}")
            
            # 保存重置后的状态
            if hasattr(self.local_ai, 'save_model_state'):
                self.local_ai.save_model_state()
            
        except Exception as e:
            print(f"❌ 重置AI引擎失败：{str(e)}")
    
    def update_ai_count_to_match_main_data(self):
        """更新AI引擎计数以匹配主程序数据"""
        if not AI_AVAILABLE or not self.local_ai:
            return
        
        try:
            main_data_count = len(self.main_app.chart_data) if self.main_app.chart_data else 0
            
            # 更新AI引擎的样本计数（保守方式，不修改正确预测数）
            if hasattr(self.local_ai, 'total_samples_seen'):
                self.local_ai.total_samples_seen = main_data_count
            
            # 保存更新后的状态
            if hasattr(self.local_ai, 'save_model_state'):
                self.local_ai.save_model_state()
            
        except Exception as e:
            print(f"❌ 更新AI计数失败：{str(e)}")

    def check_ai_data_integrity(self):
        """检查AI数据完整性"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！")
            return
        
        try:
            # 获取内存中的统计
            stats = self.local_ai.get_comprehensive_stats()
            basic_stats = stats.get('basic_stats', {})
            memory_samples = basic_stats.get('total_samples_seen', 0)
            memory_correct = basic_stats.get('correct_predictions', 0)
            
            # 获取数据库中的统计
            conn = sqlite3.connect(self.local_ai.db_path)
            cursor = conn.cursor()
            
            # 检查预测记录表
            cursor.execute("SELECT COUNT(*) FROM online_predictions")
            db_predictions_result = cursor.fetchone()
            db_predictions = db_predictions_result[0] if db_predictions_result else 0
            
            cursor.execute("SELECT COUNT(*) FROM online_predictions WHERE is_correct = 1")
            db_correct_result = cursor.fetchone()
            db_correct = db_correct_result[0] if db_correct_result else 0
            
            # 检查样本编号
            cursor.execute("SELECT MIN(sample_number), MAX(sample_number), COUNT(DISTINCT sample_number) FROM online_predictions WHERE sample_number IS NOT NULL")
            sample_info_result = cursor.fetchone()
            if sample_info_result and sample_info_result[0] is not None:
                min_sample, max_sample, unique_samples = sample_info_result
            else:
                min_sample, max_sample, unique_samples = 0, 0, 0
            
            # 检查模型预测记录
            cursor.execute("SELECT COUNT(*) FROM individual_model_predictions")
            model_predictions_result = cursor.fetchone()
            model_predictions = model_predictions_result[0] if model_predictions_result else 0
            
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables_result = cursor.fetchall()
            tables = [row[0] for row in tables_result] if tables_result else []
            
            conn.close()
            
            # 显示检查结果
            check_result = f"""📊 AI数据完整性检查报告
            
🧠 内存状态：
- 总学习样本：{memory_samples}
- 正确预测：{memory_correct}
- 当前准确率：{(memory_correct/memory_samples*100) if memory_samples > 0 else 0:.1f}%

💾 数据库状态：
- 预测记录数：{db_predictions}
- 正确预测数：{db_correct}
- 数据库准确率：{(db_correct/db_predictions*100) if db_predictions > 0 else 0:.1f}%

📈 样本信息：
- 最小样本编号：{min_sample}
- 最大样本编号：{max_sample}
- 唯一样本数：{unique_samples}
- 模型预测记录：{model_predictions}

🗃️ 数据库表：
- 存在的表：{', '.join(tables)}

✅ 数据一致性：
- 内存vs数据库：{'一致' if memory_samples == db_predictions else f'不一致！内存{memory_samples} vs 数据库{db_predictions}'}
- 正确预测：{'一致' if memory_correct == db_correct else f'不一致！内存{memory_correct} vs 数据库{db_correct}'}

💡 分析：
"""
            
            # 添加详细分析
            if memory_samples == 0 and db_predictions == 0:
                check_result += "• 暂无学习数据，这是正常的新系统状态\n"
                check_result += "• 建议：开始使用AI预测功能以积累学习数据"
            elif memory_samples > 0 and db_predictions == 0:
                check_result += "• 内存中有数据但数据库为空，可能是保存问题\n"
                check_result += "• 建议：检查数据库写入权限或重启程序"
            elif memory_samples == 0 and db_predictions > 0:
                check_result += "• 数据库有历史数据但内存中无数据\n"
                check_result += "• 建议：重启程序以加载历史数据"
            elif memory_samples == db_predictions and memory_samples > 0:
                check_result += "• 数据完整，AI正常工作\n"
                check_result += "• 系统状态良好"
            else:
                check_result += "• 数据存在不一致，可能需要修复\n"
                check_result += "• 建议：备份数据后重置AI模型"
            
            # 创建检查结果窗口
            result_window = tk.Toplevel(self.main_app.root)
            result_window.title("AI数据检查结果")
            result_window.geometry("600x700")
            result_window.resizable(True, True)
            result_window.transient(self.main_app.root)
            
            # 居中显示
            result_window.update_idletasks()
            x = (result_window.winfo_screenwidth() - result_window.winfo_width()) // 2
            y = (result_window.winfo_screenheight() - result_window.winfo_height()) // 2
            result_window.geometry(f"+{x}+{y}")
            
            # 结果显示
            text_widget = scrolledtext.ScrolledText(result_window, 
                                                   font=ModernStyle.FONTS['default'],
                                                   bg=ModernStyle.COLORS['surface'],
                                                   wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            text_widget.insert(tk.END, check_result)
            text_widget.config(state=tk.DISABLED)
            
            # 添加操作按钮
            btn_frame = tk.Frame(result_window, bg=ModernStyle.COLORS['surface'])
            btn_frame.pack(fill=tk.X, padx=20, pady=10)
            
            close_btn = tk.ttk.Button(btn_frame, text="关闭", 
                                  command=result_window.destroy,
                                  style='Primary.TButton')
            close_btn.pack(side=tk.RIGHT)
            
            if memory_samples != db_predictions and memory_samples > 0:
                sync_btn = tk.ttk.Button(btn_frame, text="同步数据", 
                                     command=lambda: self.sync_ai_data(result_window),
                                     style='Warning.TButton')
                sync_btn.pack(side=tk.RIGHT, padx=(0, 10))
            
        except Exception as e:
            error_msg = f"数据检查失败：{str(e)}\n\n错误详情：\n{str(e)}"
            messagebox.showerror("错误", error_msg)
            print(f"数据检查错误: {e}")
            import traceback
            traceback.print_exc()
    
    def sync_ai_data(self, parent_window):
        """同步AI数据"""
        try:
            if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
                messagebox.showerror("错误", "AI引擎不可用！")
                return
            
            result = messagebox.askyesno("确认同步", 
                                       "确定要同步AI数据吗？\n\n这将强制保存当前内存中的数据到数据库。",
                                       parent=parent_window)
            if result:
                # 强制保存模型状态
                self.local_ai.save_model_state()
                
                messagebox.showinfo("成功", "数据同步完成！", parent=parent_window)
                parent_window.destroy()
                
                # 重新检查数据
                self.main_app.root.after(1000, self.check_ai_data_integrity)
                
        except Exception as e:
            messagebox.showerror("错误", f"数据同步失败：{str(e)}", parent=parent_window)

    def export_ai_data(self):
        """导出AI数据"""
        if not AI_AVAILABLE or not hasattr(self, 'local_ai') or self.local_ai is None:
            messagebox.showwarning("警告", "AI引擎不可用，无法导出数据！")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="导出AI数据"
        )
        
        if filename:
            try:
                # 获取AI统计数据
                stats = self.local_ai.get_learning_stats()
                
                # 准备导出数据
                export_data = {
                    'ai_stats': stats,
                    'export_time': datetime.now().isoformat(),
                    'app_version': '3.1',
                    'ai_engine': 'local_scikit_learn'
                }
                
                # 写入文件
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("成功", f"AI数据已导出到：\n{filename}")
                self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI数据已导出到: {filename}")
                
            except Exception as e:
                messagebox.showerror("错误", f"导出AI数据失败：{str(e)}")
    
    def get_drift_count(self):
        """获取概念漂移次数"""
        try:
            if AI_AVAILABLE and hasattr(self, 'local_ai') and self.local_ai is not None:
                conn = sqlite3.connect(self.local_ai.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM drift_events")
                count = cursor.fetchone()[0]
                conn.close()
                return count
        except Exception:
            pass
        return 0

    def get_recent_drift(self):
        """获取最近的漂移时间"""
        try:
            if AI_AVAILABLE and hasattr(self, 'local_ai') and self.local_ai is not None:
                conn = sqlite3.connect(self.local_ai.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT timestamp FROM drift_events ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    dt = datetime.fromisoformat(result[0])
                    return dt.strftime('%m-%d %H:%M')
                else:
                    return "无"
        except Exception:
            pass
        return "无"

    def clear_all_data_and_reset_ai(self):
        """清空所有数据并重置AI引擎（用于测试时的完全重置）"""
        result = messagebox.askyesno("确认重置", 
                                   "⚠️ 这将清空所有数据和AI学习记录！\n\n" +
                                   "包括：\n" +
                                   "• 所有历史数据\n" +
                                   "• AI学习状态\n" +
                                   "• 预测历史\n" +
                                   "• 计分板数据\n\n" +
                                   "确定继续吗？")
        if result:
            try:
                # 清空主程序数据
                self.main_app.chart_data = []
                self.main_app.input_text.delete(1.0, tk.END)
                
                # 清空结果表格
                for item in self.main_app.result_tree.get_children():
                    self.main_app.result_tree.delete(item)
                
                # 重置计分板
                self.main_app.scoreboard_stats = {
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
                
                # 清空预测数据
                self.main_app.prediction_data = {
                    'hasPendingPrediction': False,
                    'predictedResult': None,
                    'predictionTime': None,
                    'predictionHistory': []
                }
                
                # 重置AI引擎
                self.reset_ai_to_match_main_data()
                
                # 清空预处理数据
                self.main_app.preprocessed_data = {
                    'features_matrix': {},
                    'tail_statistics': {},
                    'sliding_windows': {},
                    'metadata': {
                        'last_preprocessed_period_count': 0,
                        'last_update_time': None,
                        'data_hash': None,
                        'version': '1.0'
                    }
                }
                
                # 删除预处理文件
                if self.main_app.preprocessed_file.exists():
                    self.main_app.preprocessed_file.unlink()
                
                # 更新界面显示
                self.main_app.update_scoreboard()
                self.main_app.update_prediction_status()
                self.main_app.update_prediction_history_display()
                if hasattr(self.main_app, 'ui_updates.update_ai_status_display'):
                    self.main_app.ui_updates.update_ai_status_display()
                
                # 保存空数据
                self.main_app.data_manager.save_data()
                
                self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有数据已清空并重置")
                messagebox.showinfo("成功", "所有数据已清空并重置！")
                
            except Exception as e:
                messagebox.showerror("错误", f"重置失败：{str(e)}")