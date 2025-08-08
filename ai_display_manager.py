#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox, filedialog
import json
from datetime import datetime
from pathlib import Path

# 尝试导入matplotlib
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

# 检查AI可用性
try:
    import sklearn
    import sys
    import os
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from .local_ai_engine import UltimateOnlineAIEngine
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class AIDisplayManager:
    def __init__(self, main_app):
        self.main_app = main_app
    
    def update_learning_trend_chart(self):
        """更新学习趋势图"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self.main_app, 'learning_fig'):
            return
        
        try:
            self.main_app.learning_fig.clear()
            
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai:
                import sqlite3
                conn = sqlite3.connect(self.main_app.local_ai.db_path)
                cursor = conn.cursor()
                
                # 获取学习历史数据
                cursor.execute("""
                    SELECT sample_number, is_correct, timestamp
                    FROM online_predictions
                    WHERE is_correct IS NOT NULL
                    ORDER BY sample_number
                """)
                
                records = cursor.fetchall()
                conn.close()
                
                if records:
                    # 创建双y轴图表
                    ax1 = self.main_app.learning_fig.add_subplot(111)
                    ax2 = ax1.twinx()
                    
                    # 处理数据
                    sample_numbers = [r[0] for r in records]
                    accuracies = []
                    
                    # 计算滑动准确率
                    window_size = min(10, len(records))
                    for i in range(len(records)):
                        start_idx = max(0, i - window_size + 1)
                        window_records = records[start_idx:i+1]
                        correct_count = sum(1 for r in window_records if r[1] == 1)
                        accuracy = correct_count / len(window_records)
                        accuracies.append(accuracy)
                    
                    # 绘制准确率趋势线
                    ax1.plot(sample_numbers, accuracies, 'b-', linewidth=2, label='准确率趋势', alpha=0.8)
                    ax1.fill_between(sample_numbers, accuracies, alpha=0.3, color='blue')
                    ax1.set_xlabel('学习样本数', fontfamily='Microsoft YaHei')
                    ax1.set_ylabel('准确率', fontfamily='Microsoft YaHei', color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.set_ylim(0, 1)
                    ax1.grid(True, alpha=0.3)
                    
                    # 绘制预测结果散点
                    correct_samples = [s for s, r, _ in records if r == 1]
                    incorrect_samples = [s for s, r, _ in records if r == 0]
                    
                    ax2.scatter(correct_samples, [1]*len(correct_samples), 
                               color='green', alpha=0.6, s=20, label='预测正确')
                    ax2.scatter(incorrect_samples, [0]*len(incorrect_samples), 
                               color='red', alpha=0.6, s=20, label='预测错误')
                    ax2.set_ylabel('预测结果', fontfamily='Microsoft YaHei', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.set_yticks([0, 1])
                    ax2.set_yticklabels(['错误', '正确'])
                    
                    # 设置标题和图例
                    self.main_app.learning_fig.suptitle('AI学习趋势分析', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
                    
                    # 添加统计信息
                    total_predictions = len(records)
                    total_correct = sum(1 for r in records if r[1] == 1)
                    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
                    
                    stats_text = f'总预测: {total_predictions}次  正确: {total_correct}次  准确率: {overall_accuracy:.1%}'
                    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontfamily='Microsoft YaHei')
                
                else:
                    # 没有数据时显示提示
                    ax = self.main_app.learning_fig.add_subplot(111)
                    ax.text(0.5, 0.5, '暂无学习数据\n开始使用AI预测功能后\n这里将显示学习趋势图', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='gray', fontfamily='Microsoft YaHei',
                           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            
            # 调整布局并重绘
            self.main_app.learning_fig.tight_layout()
            self.main_app.learning_canvas.draw_idle()
        
        except Exception as e:
            print(f"更新学习趋势图失败: {e}")
    
    def update_model_performance_table(self, stats):
        """更新模型性能表格"""
        if not hasattr(self.main_app, 'model_performance_tree'):
            return
            
        # 清空现有数据
        for item in self.main_app.model_performance_tree.get_children():
            self.main_app.model_performance_tree.delete(item)
        
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai:
                # 获取真实的模型性能统计
                model_stats = self.main_app.local_ai.get_model_performance_stats()
                ensemble_weights = stats.get('ensemble_weights', {})
                
                if model_stats:
                    # 显示有真实数据的模型
                    for model_name, perf_data in model_stats.items():
                        total_pred = perf_data['total_predictions']
                        correct_pred = perf_data['correct_predictions']
                        accuracy = perf_data['accuracy']
                        # 安全获取权重值 - 处理字典类型的权重
                        weight_data = ensemble_weights.get(model_name, 0.0)
                        if isinstance(weight_data, dict):
                            weight = weight_data.get('total_weight', 0.0)
                        else:
                            weight = weight_data if weight_data is not None else 0.0
                        
                        # 模型状态评估
                        if accuracy > 0.7:
                            status = "优秀"
                        elif accuracy > 0.6:
                            status = "良好"
                        elif accuracy > 0.4:
                            status = "一般"
                        elif accuracy > 0.2:
                            status = "较差"
                        else:
                            status = "很差"
                        
                        # 如果预测次数太少，标记为学习中
                        if total_pred < 10:
                            status = "学习中"
                        
                        # 格式化模型名称
                        display_name = model_name.replace('_', ' ').title()
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."
                        
                        self.main_app.model_performance_tree.insert('', tk.END, values=[
                            display_name,
                            total_pred,
                            correct_pred,
                            f"{accuracy:.1%}",
                            f"{weight:.3f}",
                            status
                        ])
                
                else:
                    # 没有真实数据时显示占位信息
                    for model_name, weight_data in ensemble_weights.items():
                        display_name = model_name.replace('_', ' ').title()
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."
                    
                        # 安全获取权重值 - 处理字典类型的权重
                        if isinstance(weight_data, dict):
                            weight = weight_data.get('total_weight', 0.0)
                        else:
                            weight = weight_data if weight_data is not None else 0.0
                    
                        self.main_app.model_performance_tree.insert('', tk.END, values=[
                            display_name,
                            0,
                            0,
                            "0.0%",
                            f"{weight:.3f}",
                            "待训练"
                        ])
        
        except Exception as e:
            print(f"更新模型性能表格失败: {e}")
    
    def update_learning_history_table(self):
        """更新学习历史表格"""
        if not hasattr(self.main_app, 'learning_history_tree'):
            return
            
        # 清空现有数据
        for item in self.main_app.learning_history_tree.get_children():
            self.main_app.learning_history_tree.delete(item)
        
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai:
                import sqlite3
                conn = sqlite3.connect(self.main_app.local_ai.db_path)
                cursor = conn.cursor()
                
                # 获取所有学习记录（按时间倒序，最新的在前面）
                cursor.execute("""
                    SELECT timestamp, sample_number, predicted_tails, actual_tails, 
                           is_correct, model_used, drift_detected
                    FROM online_predictions
                    ORDER BY timestamp DESC
                """)
                
                records = cursor.fetchall()
                conn.close()
                
                for record in records:
                    timestamp, sample_num, pred_tails, actual_tails, is_correct, model_used, drift_detected = record
                    
                    # 格式化时间
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime('%m-%d %H:%M')
                    except:
                        time_str = timestamp[:16] if len(timestamp) > 16 else timestamp
                    
                    # 解析预测和实际尾数
                    try:
                        import json
                        pred_list = json.loads(pred_tails) if pred_tails else []
                        actual_list = json.loads(actual_tails) if actual_tails else []
                        
                        pred_str = ','.join(map(str, pred_list))
                        actual_str = ','.join(map(str, actual_list))
                    except:
                        pred_str = str(pred_tails)
                        actual_str = str(actual_tails)
                    
                    # 结果状态
                    result = "✅" if is_correct == 1 else "❌" if is_correct == 0 else "?"
                    
                    # 漂移检测
                    drift_str = "是" if drift_detected == 1 else "否"
                    
                    # 模型名称简化
                    model_display = model_used.replace('_', ' ').title() if model_used else "未知"
                    if len(model_display) > 20:
                        model_display = model_display[:17] + "..."
                    
                    self.main_app.learning_history_tree.insert('', tk.END, values=[
                        time_str,
                        sample_num or "?",
                        pred_str,
                        actual_str,
                        result,
                        model_display,
                        drift_str
                    ])
        
        except Exception as e:
            print(f"更新学习历史表格失败: {e}")
    
    def update_drift_analysis_display(self):
        """更新漂移分析显示"""
        try:
            drift_events_count = self.get_drift_count()
            if hasattr(self.main_app, 'drift_events_label'):
                self.main_app.drift_events_label.config(text=f"{drift_events_count}次")
            
            # 获取最近的漂移时间
            recent_drift = self.get_recent_drift()
            if hasattr(self.main_app, 'recent_drift_label'):
                self.main_app.recent_drift_label.config(text=recent_drift)
            
            # 更新漂移历史表格
            self.update_drift_history_table()
        
        except Exception as e:
            print(f"更新漂移分析显示失败: {e}")
    
    def get_recent_drift(self):
        """获取最近的漂移时间"""
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai is not None:
                import sqlite3
                conn = sqlite3.connect(self.main_app.local_ai.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT timestamp FROM drift_events ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    from datetime import datetime
                    dt = datetime.fromisoformat(result[0])
                    return dt.strftime('%m-%d %H:%M')
                else:
                    return "无"
        except Exception:
            pass
        return "无"
    
    def get_drift_count(self):
        """获取漂移次数"""
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai is not None:
                import sqlite3
                conn = sqlite3.connect(self.main_app.local_ai.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM drift_events")
                result = cursor.fetchone()
                conn.close()
                return result[0] if result else 0
        except Exception:
            pass
        return 0
    
    def update_drift_history_table(self):
        """更新漂移历史表格"""
        if not hasattr(self.main_app, 'drift_history_tree'):
            return
            
        # 清空现有数据
        for item in self.main_app.drift_history_tree.get_children():
            self.main_app.drift_history_tree.delete(item)
        
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai is not None:
                import sqlite3
                conn = sqlite3.connect(self.main_app.local_ai.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT timestamp, detector_name, drift_type, confidence, action_taken
                    FROM drift_events
                    ORDER BY timestamp DESC
                """)
                
                records = cursor.fetchall()
                conn.close()
                
                for record in records:
                    timestamp, detector_name, drift_type, confidence, action_taken = record
                    
                    # 格式化时间
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime('%m-%d %H:%M:%S')
                    except:
                        time_str = timestamp[:19] if len(timestamp) > 19 else timestamp
                    
                    self.main_app.drift_history_tree.insert('', tk.END, values=[
                        time_str,
                        detector_name or "未知",
                        drift_type or "概念漂移",
                        f"{confidence:.3f}" if confidence else "?",
                        action_taken or "自适应调整"
                    ])
        
        except Exception as e:
            print(f"更新漂移历史表格失败: {e}")
    
    def update_detailed_report_display(self, stats):
        """更新详细报告显示"""
        if not hasattr(self.main_app, 'ai_detailed_report_text'):
            return
        
        # 启用编辑模式
        self.main_app.ai_detailed_report_text.config(state=tk.NORMAL)
        
        # 清空现有内容
        self.main_app.ai_detailed_report_text.delete(1.0, tk.END)
        
        try:
            basic_stats = stats.get('basic_stats', {})
            model_stats = stats.get('model_stats', {})
            ensemble_weights = stats.get('ensemble_weights', {})
            
            # 获取调试信息
            debug_info = stats.get('debug_info', {})
            
            report_text = f"""
🤖 AI学习详细报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 数据验证：
- 数据库预测记录：{debug_info.get('db_prediction_records', '未知')} 条
- 样本编号范围：{debug_info.get('sample_number_range', '未知')}
- 内存中总样本：{debug_info.get('memory_total_samples', '未知')} 个
- 内存中正确预测：{debug_info.get('memory_correct_predictions', '未知')} 个

📊 核心统计：
- 总学习样本：{basic_stats.get('total_samples_seen', 0)} 个
- 正确预测：{basic_stats.get('correct_predictions', 0)} 次
- 当前准确率：{basic_stats.get('current_accuracy', 0):.2%}
- 初始化状态：{'已初始化' if basic_stats.get('is_initialized', False) else '未初始化'}

🎯 模型构成：
- River模型：{model_stats.get('river_models', 0)} 个
- scikit-multiflow模型：{model_stats.get('sklearn_models', 0)} 个
- Deep-River模型：{model_stats.get('deep_models', 0)} 个
- 总模型数：{model_stats.get('total_models', 0)} 个
- 漂移检测器：{len(stats.get('drift_detectors', []))} 个

⚖️ 模型权重排名："""
            
            # 按权重排序显示模型 - 修复字典比较问题
            try:
                sorted_weights = sorted(ensemble_weights.items(), 
                                    key=lambda x: x[1].get('total_weight', 0.0) if isinstance(x[1], dict) else 0.0, 
                                    reverse=True)
            except Exception as sort_error:
                print(f"排序权重时出错: {sort_error}")
                sorted_weights = list(ensemble_weights.items())
            for i, (model_name, weight_data) in enumerate(sorted_weights[:10], 1):
                model_display = model_name.replace('_', ' ').title()
                # 安全获取权重值 - 处理字典类型的权重
                if isinstance(weight_data, dict):
                    weight = weight_data.get('total_weight', 0.0)
                else:
                    weight = weight_data if weight_data is not None else 0.0
                report_text += f"\n{i:2d}. {model_display:<25} 权重: {weight:.4f}"
            
            # 学习表现分析
            if basic_stats.get('total_samples_seen', 0) > 0:
                # 安全计算稳定模型数量
                stable_models_count = 0
                for w in ensemble_weights.values():
                    if isinstance(w, dict):
                        weight = w.get('total_weight', 0.0)
                    else:
                        weight = w if w is not None else 0.0
                    
                    if weight > 0.05:
                        stable_models_count += 1
                
                report_text += f"""

📈 学习表现分析：
- 学习效率：{'优秀' if basic_stats.get('current_accuracy', 0) > 0.6 else '良好' if basic_stats.get('current_accuracy', 0) > 0.4 else '需改进'}
- 样本利用率：{basic_stats.get('total_samples_seen', 0) / max(1, len(self.main_app.chart_data)) * 100:.1f}%
- 模型稳定性：{'稳定' if stable_models_count >= 5 else '一般'}
"""
            
            # 概念漂移分析
            drift_count = self.get_drift_count()
            if drift_count > 0:
                report_text += f"""
🚨 概念漂移分析：
- 检测到漂移事件：{drift_count} 次
- 漂移频率：{drift_count / max(1, basic_stats.get('total_samples_seen', 1)) * 100:.2f}%
- 最近漂移：{self.get_recent_drift()}
- 适应能力：{'强' if drift_count > 0 else '未测试'}
"""
            else:
                report_text += f"""
🚨 概念漂移分析：
- 检测到漂移事件：0 次
- 数据稳定性：良好
- 适应能力：待观察
"""
            
            # 改进建议
            report_text += f"""
💡 改进建议："""
            
            if basic_stats.get('total_samples_seen', 0) < 50:
                report_text += f"\n• 增加训练数据：当前样本数较少，建议积累更多学习样本"
            
            if basic_stats.get('current_accuracy', 0) < 0.5:
                report_text += f"\n• 提升预测准确率：当前准确率偏低，AI仍在学习中"
            elif basic_stats.get('current_accuracy', 0) > 0.7:
                report_text += f"\n• 维持良好表现：当前准确率较高，继续保持"
            
            # 找出表现差的模型 - 安全处理权重数据类型
            poor_models = []
            for name, weight_data in ensemble_weights.items():
                if isinstance(weight_data, dict):
                    weight = weight_data.get('total_weight', 0.0)
                else:
                    weight = weight_data if weight_data is not None else 0.0
                
                if weight < 0.03:
                    poor_models.append(name)
            
            if poor_models:
                report_text += f"\n• 优化弱势模型：{len(poor_models)} 个模型权重较低，考虑调整"
            
            if drift_count == 0 and basic_stats.get('total_samples_seen', 0) > 30:
                report_text += f"\n• 数据多样性：建议增加更多样化的数据以测试适应能力"
            
            report_text += f"""

🔮 使用建议：
- 持续使用AI预测功能，让AI从实际结果中学习
- 定期查看学习进展，了解AI的改进情况
- 在数据分布发生变化时，AI会自动检测并适应
- 建议积累至少100个学习样本以获得稳定性能

📝 技术说明：
- 采用在线学习算法，无需离线训练
- 支持概念漂移检测和自适应调整
- 使用集成学习提高预测稳定性
- 实时更新模型权重以优化性能
"""
            
            self.main_app.ai_detailed_report_text.insert(tk.END, report_text)
        
        except Exception as e:
            error_text = f"生成详细报告时出错：{str(e)}\n\n请检查AI系统状态或联系技术支持。"
            self.main_app.ai_detailed_report_text.insert(tk.END, error_text)
        
        # 设置为只读
        self.main_app.ai_detailed_report_text.config(state=tk.DISABLED)
    
    def refresh_detailed_report(self):
        """刷新详细报告"""
        try:
            if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai is not None:
                stats = self.main_app.local_ai.get_comprehensive_stats()
                self.update_detailed_report_display(stats)
                self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI详细报告已刷新")
            else:
                messagebox.showwarning("警告", "AI引擎不可用，无法刷新报告")
        except Exception as e:
            print(f"刷新详细报告时出错: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"刷新报告失败：{str(e)}")

    def trigger_update_after_learning(self):
        """在AI学习后触发更新（简化版，避免循环调用）"""
        try:
            print("🔄 AI学习后触发显示更新...")
            
            # 添加调用保护
            if hasattr(self, '_updating_after_learning') and self._updating_after_learning:
                return
            
            self._updating_after_learning = True
            
            try:
                # 更新AI相关显示（减少定时器使用）
                self.update_learning_trend_chart()
                
                # 更新模型性能表格
                if AI_AVAILABLE and hasattr(self.main_app, 'local_ai') and self.main_app.local_ai is not None:
                    stats = self.main_app.local_ai.get_comprehensive_stats()
                    self.update_model_performance_table(stats)
                
                # 更新学习历史表格
                self.update_learning_history_table()
                
                # 更新漂移分析显示
                self.update_drift_analysis_display()
                
                # 只进行一次投资详情更新
                if hasattr(self.main_app, 'investment_manager') and self.main_app.investment_manager is not None:
                    print("🔄 更新投资详情...")
                    self.main_app.investment_manager.refresh_investment_details()
                
                print("✅ AI学习后显示更新完成")
                
            finally:
                # 重置调用保护标志
                self._updating_after_learning = False
                
        except Exception as e:
            self._updating_after_learning = False
            print(f"AI学习后触发更新失败: {e}")

    def export_detailed_report(self):
        """导出详细报告"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                title="导出AI详细报告"
            )
            
            if filename:
                report_content = self.main_app.ai_detailed_report_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                messagebox.showinfo("成功", f"AI详细报告已导出到：\n{filename}")
                self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI报告已导出到: {filename}")
        
        except Exception as e:
            messagebox.showerror("错误", f"导出报告失败：{str(e)}")
    
    def reset_ai_model(self):
        """重置AI模型"""
        if not AI_AVAILABLE or not hasattr(self.main_app, 'local_ai') or self.main_app.local_ai is None:
            messagebox.showerror("错误", "AI引擎不可用！\n\n请确保已安装scikit-learn依赖包。")
            return
        
        result = messagebox.askyesno("确认重置", 
                                   "确定要重置AI模型吗？\n\n这将删除所有学习数据和模型，\n操作不可撤销！")
        if result:
            try:
                # 重置AI模型
                reset_success = self.main_app.local_ai.reset_model()
            
                if reset_success:
                    # 清除预测结果
                    self.main_app.ai_prediction_result = None
                
                    # 重置主程序中的相关数据
                    self.main_app.chart_data = []
                
                    # 更新显示
                    self.main_app.ui_updates.update_ai_status_display()
                    self.main_app.ui_updates.update_ai_progress_display()
                    if hasattr(self.main_app, 'ui_updates') and hasattr(self.main_app.ui_updates, 'update_learning_progress_display'):
                        self.main_app.ui_updates.update_learning_progress_display()
                
                    messagebox.showinfo("成功", "🔄 AI模型已完全重置！\n\n• 所有学习数据已清除\n• 所有模型已重新初始化\n• 数据库记录已清空\n\n可以重新开始训练")
                    self.main_app.add_to_history(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI模型已完全重置")
                else:
                    messagebox.showerror("错误", "AI模型重置失败！\n\n请检查控制台输出获取详细错误信息。")
            
            except Exception as e:
                messagebox.showerror("错误", f"重置AI模型失败：{str(e)}\n\n请确保AI引擎正常运行。")
                import traceback
                traceback.print_exc()
    
    def sync_ai_data(self, parent_window):
        """同步AI数据"""
        try:
            if not AI_AVAILABLE or not hasattr(self.main_app, 'local_ai') or self.main_app.local_ai is None:
                messagebox.showerror("错误", "AI引擎不可用！")
                return
            
            result = messagebox.askyesno("确认同步", 
                                       "确定要同步AI数据吗？\n\n这将强制保存当前内存中的数据到数据库。",
                                       parent=parent_window)
            if result:
                # 强制保存模型状态
                self.main_app.local_ai.save_model_state()
                
                messagebox.showinfo("成功", "数据同步完成！", parent=parent_window)
                parent_window.destroy()
                
                # 重新检查数据
                self.main_app.root.after(1000, self.main_app.check_ai_data_integrity)
                
        except Exception as e:
            messagebox.showerror("错误", f"数据同步失败：{str(e)}", parent=parent_window)
    
    def force_update_all_ai_displays(self):
        """强制更新所有AI显示组件"""
        try:
            if not AI_AVAILABLE or not hasattr(self.main_app, 'local_ai') or self.main_app.local_ai is None:
                return
        
            # 分步骤更新，避免UI阻塞
            def update_step_1():
                try:
                    self.main_app.ui_updates.update_ai_status_display()
                except Exception as e:
                    pass
                # 安排下一步
                self.main_app.root.after(100, update_step_2)
        
            def update_step_2():
                try:
                    self.main_app.ui_updates.update_ai_prediction_display()
                except Exception as e:
                    pass
                # 安排下一步
                self.main_app.root.after(100, update_step_3)
        
            def update_step_3():
                try:
                    self.main_app.ui_updates.update_learning_progress_display()
                except Exception as e:
                    pass
                # 安排下一步
                self.main_app.root.after(100, update_step_4)
        
            def update_step_4():
                try:
                    if hasattr(self.main_app, 'ai_progress_text'):
                        self.main_app.ui_updates.update_ai_progress_display()
                except Exception as e:
                    pass
                # 安排下一步
                self.main_app.root.after(100, update_step_5)
            
            def update_step_5():
                try:
                    # 刷新投资详情
                    if hasattr(self.main_app, 'investment_manager') and self.main_app.investment_manager is not None:
                        self.main_app.investment_manager.refresh_investment_details()
                except Exception as e:
                    pass
        
            # 开始第一步
            update_step_1()
        
        except Exception as e:
            pass
    
    def ensure_ai_ui_components_initialized(self):
        """确保AI相关UI组件已初始化"""
        try:
            print("🔍 检查AI UI组件初始化状态...")
            
            # 检查学习概览组件是否存在
            required_components = [
                'total_samples_label',
                'current_accuracy_label',
                'best_model_label', 
                'drift_count_label'
            ]
            
            missing_components = []
            for component in required_components:
                if not hasattr(self.main_app, component):
                    missing_components.append(component)
            
            if missing_components:
                print(f"⚠️ 检测到缺少UI组件: {missing_components}")
                print("💡 建议：请手动切换到AI助手页面的学习概览标签页")
                
                # 尝试切换到AI助手标签页（如果notebook可用）
                if hasattr(self.main_app, 'notebook') and self.main_app.notebook:
                    try:
                        # 查找AI助手选项卡的索引
                        tab_count = self.main_app.notebook.index("end")
                        for i in range(tab_count):
                            tab_text = self.main_app.notebook.tab(i, "text")
                            if "AI助手" in tab_text:
                                print(f"🔄 自动切换到AI助手选项卡（索引{i}）")
                                self.main_app.notebook.select(i)
                                break
                    except Exception as e:
                        print(f"❌ 自动切换选项卡失败: {e}")
                
                return False
            else:
                print("✅ 所有AI UI组件已初始化")
                return True
                
        except Exception as e:
            print(f"❌ 检查UI组件初始化失败: {e}")
            return False
    
    def check_ai_ui_components_status(self):
        """检查AI相关UI组件的状态"""
        
        # 定义需要检查的组件
        ui_components = {
            'total_samples_label': '总学习样本标签',
            'current_accuracy_label': '当前准确率标签',
            'best_model_label': '最佳模型标签',
            'drift_count_label': '概念漂移标签',
            'drift_events_label': '漂移事件标签',
            'recent_drift_label': '最近漂移标签',
            'model_performance_tree': '模型性能表格',
            'learning_history_tree': '学习历史表格',
            'drift_history_tree': '漂移历史表格'
        }
        
        missing_components = []
        
        for attr_name, display_name in ui_components.items():
            if not hasattr(self.main_app, attr_name):
                missing_components.append(display_name)
        
        return len(missing_components) == 0