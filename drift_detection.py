# drift_detection.py - 概念漂移检测模块

from typing import List, Dict, Any
from datetime import datetime

class DriftDetectionManager:
    """概念漂移检测管理器"""
    
    def __init__(self, ai_config, db_manager):
        self.ai_config = ai_config
        self.db_manager = db_manager
        
    def handle_concept_drift(self, drift_detectors: List[str], ensemble_weights: Dict, 
                           learning_config: Dict, deep_learning_module=None):
        """处理概念漂移"""
        print(f"🚨 检测到概念漂移！检测器: {drift_detectors}")
        
        # 记录漂移事件
        for detector in drift_detectors:
            self.db_manager.save_drift_record(detector, 'concept_drift', 1.0, 'adaptive_response')
        
        # 概念漂移响应策略
        if len(drift_detectors) >= 2:  # 多个检测器同时报告漂移
            print("🔄 执行强化适应策略")
            # 重置表现较差的模型权重
            self.reset_poor_performing_models(ensemble_weights)
        else:
            print("⚡ 执行轻量适应策略")
            # 仅调整学习率或权重
            self.adjust_learning_parameters(learning_config, deep_learning_module)
        
        # 重置漂移检测器
        for detector_name in drift_detectors:
            # 这里需要访问实际的漂移检测器实例
            # 由于漂移检测器在主引擎中，这个方法需要在主引擎中调用
            pass
    
    def handle_advanced_concept_drift(self, drift_info: Dict, feature_selector=None, 
                                    feature_weighter=None, ensemble_weights: Dict = None,
                                    deep_learning_module=None):
        """处理高级概念漂移"""
        print("🔧 执行高级概念漂移响应策略...")
        
        # 分析漂移类型和严重程度
        individual_detectors = drift_info.get('individual_detectors', {})
        
        # 统计检测到漂移的检测器数量
        drift_count = sum(1 for detector_info in individual_detectors.values() 
                         if detector_info.get('detected', False))
        
        total_detectors = len(individual_detectors)
        drift_severity = drift_count / total_detectors if total_detectors > 0 else 0.0
        
        print(f"   漂移严重程度: {drift_severity:.2f} ({drift_count}/{total_detectors}个检测器)")
        
        # 根据严重程度采取不同策略
        if drift_severity >= 0.75:  # 严重漂移
            print("   🚨 执行严重漂移响应策略")
            self._severe_drift_response(feature_selector, feature_weighter, ensemble_weights, deep_learning_module)
            
        elif drift_severity >= 0.5:  # 中等漂移
            print("   ⚠️ 执行中等漂移响应策略")
            self._moderate_drift_response(feature_selector, feature_weighter, ensemble_weights, deep_learning_module)
            
        else:  # 轻微漂移
            print("   ℹ️ 执行轻微漂移响应策略")
            self._mild_drift_response(feature_selector, feature_weighter, ensemble_weights)
        
        # 记录漂移事件
        self.db_manager.save_drift_record('advanced_ensemble', 'concept_drift', drift_severity, 
                                        f'severity_{drift_severity:.2f}_response')
    
    def _severe_drift_response(self, feature_selector=None, feature_weighter=None, 
                             ensemble_weights: Dict = None, deep_learning_module=None):
        """严重漂移响应"""
        # 1. 重置特征处理组件
        if feature_selector and hasattr(self.ai_config, 'DynamicFeatureSelector'):
            try:
                # 创建新的特征选择器实例（需要在主引擎中更新引用）
                print("   🔄 需要重置特征处理组件（在主引擎中执行）")
            except Exception as e:
                print(f"   ❌ 重置特征处理组件失败: {e}")
        
        # 2. 大幅降低表现差的模型权重
        if ensemble_weights:
            for model_key, weight_info in ensemble_weights.items():
                if len(weight_info.get('performance_history', [])) >= 10:
                    recent_performance = weight_info['performance_history'][-10:]
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    if avg_performance < 0.3:
                        weight_info['weight'] *= 0.3  # 大幅降低权重
                        print(f"   📉 大幅降低模型 {model_key} 权重")
        
        # 3. 增加学习率（如果支持）
        self._adjust_learning_rates(factor=1.5, deep_learning_module=deep_learning_module, 
                                   feature_weighter=feature_weighter)
    
    def _moderate_drift_response(self, feature_selector=None, feature_weighter=None, 
                               ensemble_weights: Dict = None, deep_learning_module=None):
        """中等漂移响应"""
        # 1. 调整特征选择比例
        if feature_selector and hasattr(feature_selector, 'selection_ratio'):
            feature_selector.selection_ratio = min(0.95, feature_selector.selection_ratio + 0.05)
            print(f"   🎯 调整特征选择比例到 {feature_selector.selection_ratio:.2f}")
        
        # 2. 适度调整模型权重
        if ensemble_weights:
            for model_key, weight_info in ensemble_weights.items():
                if len(weight_info.get('performance_history', [])) >= 5:
                    recent_performance = weight_info['performance_history'][-5:]
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    if avg_performance < 0.4:
                        weight_info['weight'] *= 0.7  # 适度降低权重
                        print(f"   📊 适度调整模型 {model_key} 权重")
        
        # 3. 适度调整学习率
        self._adjust_learning_rates(factor=1.2, deep_learning_module=deep_learning_module, 
                                   feature_weighter=feature_weighter)
    
    def _mild_drift_response(self, feature_selector=None, feature_weighter=None, 
                           ensemble_weights: Dict = None):
        """轻微漂移响应"""
        # 1. 增加特征更新频率
        if feature_selector and hasattr(feature_selector, 'update_frequency'):
            feature_selector.update_frequency = max(5, feature_selector.update_frequency - 2)
            print(f"   ⚡ 增加特征更新频率到 {feature_selector.update_frequency}")
        
        # 2. 轻微调整权重学习率
        if feature_weighter and hasattr(feature_weighter, 'learning_rate'):
            feature_weighter.learning_rate = min(0.02, feature_weighter.learning_rate * 1.1)
            print(f"   🔧 调整特征权重学习率到 {feature_weighter.learning_rate:.4f}")
        
        # 3. 轻微调整模型权重
        if ensemble_weights:
            for model_key, weight_info in ensemble_weights.items():
                recent_updates = weight_info.get('performance_history', [])[-3:]
                if recent_updates and sum(recent_updates) == 0:  # 最近3次都预测错误
                    weight_info['weight'] *= 0.9
                    print(f"   📈 轻微调整模型 {model_key} 权重")
    
    def _adjust_learning_rates(self, factor: float, deep_learning_module=None, feature_weighter=None):
        """调整学习率（如果模型支持）"""
        adjusted_count = 0
        
        # 调整PyTorch深度学习模型的学习率
        if deep_learning_module and hasattr(deep_learning_module, 'models') and deep_learning_module.models:
            try:
                if hasattr(deep_learning_module, 'optimizers'):
                    for model_name, optimizer in deep_learning_module.optimizers.items():
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] *= factor
                            new_lr = param_group['lr']
                            print(f"   📚 调整 {model_name} 学习率: {old_lr:.6f} → {new_lr:.6f}")
                            adjusted_count += 1
            except Exception as e:
                print(f"   ⚠️ 调整PyTorch学习率失败: {e}")
        
        # 调整特征处理组件的学习率
        if feature_weighter and hasattr(feature_weighter, 'learning_rate'):
            try:
                old_lr = feature_weighter.learning_rate
                feature_weighter.learning_rate *= factor
                feature_weighter.learning_rate = min(0.05, max(0.001, feature_weighter.learning_rate))
                print(f"   🎯 调整特征权重学习率: {old_lr:.6f} → {feature_weighter.learning_rate:.6f}")
                adjusted_count += 1
            except Exception as e:
                print(f"   ⚠️ 调整特征权重学习率失败: {e}")
        
        if adjusted_count > 0:
            print(f"   ✅ 成功调整了 {adjusted_count} 个组件的学习率")
        else:
            print("   ℹ️ 没有可调整的学习率参数")
    
    def reset_poor_performing_models(self, ensemble_weights: Dict):
        """重置表现较差的模型"""
        reset_count = 0
        for model_key, weight_info in ensemble_weights.items():
            performance_history = weight_info.get('performance_history', [])
            
            if len(performance_history) >= 10:
                recent_accuracy = sum(performance_history[-10:]) / 10
                
                if recent_accuracy < 0.3:  # 表现很差
                    # 减少权重
                    old_weight = weight_info['weight']
                    weight_info['weight'] = max(weight_info['weight'] * 0.5, 0.01)
                    reset_count += 1
                    print(f"   📉 降低模型 {model_key} 权重: {old_weight:.4f} → {weight_info['weight']:.4f}")
        
        if reset_count > 0:
            print(f"   ✅ 重置了 {reset_count} 个表现较差的模型")
        else:
            print("   ℹ️ 没有发现需要重置的模型")
    
    def adjust_learning_parameters(self, learning_config: Dict, deep_learning_module=None):
        """调整学习参数"""
        try:
            # 轻微调整概念漂移敏感度
            old_sensitivity = learning_config.get('drift_sensitivity', 0.005)
            learning_config['drift_sensitivity'] = min(old_sensitivity * 1.1, 0.01)
            print(f"   🎛️ 调整漂移敏感度: {old_sensitivity:.4f} → {learning_config['drift_sensitivity']:.4f}")
            
            # 如果有深度学习模块，调整其学习参数
            if deep_learning_module and hasattr(deep_learning_module, 'update_learning_rate'):
                try:
                    for model_name in deep_learning_module.models.keys():
                        deep_learning_module.update_learning_rate(model_name, factor=1.1)
                        print(f"   📚 调整深度学习模型 {model_name} 学习率")
                except Exception as e:
                    print(f"   ⚠️ 调整深度学习参数失败: {e}")
            
        except Exception as e:
            print(f"   ❌ 调整学习参数失败: {e}")
    
    def reset_drift_detectors(self, drift_detectors: Dict):
        """重置漂移检测器"""
        reset_count = 0
        for detector_name, detector in drift_detectors.items():
            try:
                if hasattr(detector, 'reset'):
                    detector.reset()
                    reset_count += 1
                    print(f"   🔄 重置漂移检测器: {detector_name}")
            except Exception as e:
                print(f"   ⚠️ 重置漂移检测器 {detector_name} 失败: {e}")
        
        if reset_count > 0:
            print(f"   ✅ 成功重置了 {reset_count} 个漂移检测器")
        else:
            print("   ℹ️ 没有可重置的漂移检测器")
    
    def get_drift_detection_stats(self) -> Dict:
        """获取漂移检测统计"""
        try:
            drift_events_count = self.db_manager.get_drift_events_count()
            
            # 可以添加更多统计信息
            stats = {
                'total_drift_events': drift_events_count,
                'detection_enabled': True,
                'response_strategies': ['severe', 'moderate', 'mild'],
                'last_check': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"获取漂移检测统计失败: {e}")
            return {
                'total_drift_events': 0,
                'detection_enabled': False,
                'error': str(e)
            }
    
    def analyze_drift_patterns(self, ensemble_weights: Dict) -> Dict:
        """分析漂移模式"""
        try:
            analysis = {
                'models_with_declining_performance': [],
                'models_with_stable_performance': [],
                'models_with_improving_performance': [],
                'overall_stability': 'unknown'
            }
            
            declining_count = 0
            stable_count = 0
            improving_count = 0
            
            for model_key, weight_info in ensemble_weights.items():
                performance_history = weight_info.get('performance_history', [])
                
                if len(performance_history) >= 10:
                    recent_performance = performance_history[-5:]
                    older_performance = performance_history[-10:-5]
                    
                    if recent_performance and older_performance:
                        recent_avg = sum(recent_performance) / len(recent_performance)
                        older_avg = sum(older_performance) / len(older_performance)
                        
                        performance_change = recent_avg - older_avg
                        
                        if performance_change < -0.1:  # 下降超过10%
                            analysis['models_with_declining_performance'].append({
                                'model': model_key,
                                'change': performance_change,
                                'recent_performance': recent_avg
                            })
                            declining_count += 1
                        elif performance_change > 0.1:  # 上升超过10%
                            analysis['models_with_improving_performance'].append({
                                'model': model_key,
                                'change': performance_change,
                                'recent_performance': recent_avg
                            })
                            improving_count += 1
                        else:
                            analysis['models_with_stable_performance'].append({
                                'model': model_key,
                                'change': performance_change,
                                'recent_performance': recent_avg
                            })
                            stable_count += 1
            
            # 分析整体稳定性
            total_models = declining_count + stable_count + improving_count
            if total_models > 0:
                if declining_count / total_models > 0.5:
                    analysis['overall_stability'] = 'declining'
                elif improving_count / total_models > 0.3:
                    analysis['overall_stability'] = 'improving'
                else:
                    analysis['overall_stability'] = 'stable'
            
            analysis['summary'] = {
                'total_analyzed': total_models,
                'declining_models': declining_count,
                'stable_models': stable_count,
                'improving_models': improving_count
            }
            
            return analysis
            
        except Exception as e:
            print(f"分析漂移模式失败: {e}")
            return {'error': str(e)}