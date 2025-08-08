# feature_engineering.py - 特征工程模块

import numpy as np
import math
from typing import List, Dict, Any

class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self, ai_config, feature_selector=None, feature_combiner=None, 
                 timeseries_enhancer=None, feature_weighter=None, feature_assessor=None):
        self.ai_config = ai_config
        
        # 智能特征处理组件
        self.feature_selector = feature_selector
        self.feature_combiner = feature_combiner
        self.timeseries_enhancer = timeseries_enhancer
        self.feature_weighter = feature_weighter
        self.feature_assessor = feature_assessor
        
        # 特征处理统计
        self.feature_processing_stats = {
            'enhancement_count': 0,
            'selection_updates': 0,
            'quality_assessments': 0,
            'last_quality_report': {}
        }
    
    def extract_enhanced_features(self, data_list: List[Dict], current_index: int = 0, main_app_ref=None) -> np.ndarray:
        """提取增强特征（使用智能特征处理组件）"""
    
        # 添加数据量诊断
        data_count = len(data_list) if data_list else 0
        if data_count < 30:
            print(f"🔍 AI特征提取：接收到{data_count}期数据（数据量较少）")
    
        # 尝试从主应用获取预处理特征
        if main_app_ref:
            try:
                preprocessed_features = main_app_ref.extract_preprocessed_features(current_index)
                if preprocessed_features is not None and len(preprocessed_features) == 60:
                    return self._apply_intelligent_feature_processing(preprocessed_features)
            except Exception as e:
                print(f"使用预处理特征失败，回退到原方法: {e}")
    
        # 回退到原始特征提取方法
        try:
            basic_features = self._extract_features_fallback(data_list, current_index)
            if basic_features is None or len(basic_features) == 0:
                print(f"⚠️ 回退特征提取返回空值，使用默认特征")
                basic_features = np.zeros(60, dtype=float)
            return self._apply_intelligent_feature_processing(basic_features)
        except Exception as fallback_e:
            print(f"❌ 回退特征提取也失败: {fallback_e}")
            # 最后的安全网：返回固定的零特征
            return np.zeros(60, dtype=float)
    
    def _extract_features_fallback(self, data_list: List[Dict], current_index: int = 0) -> np.ndarray:
        """回退的特征提取方法，确保数据安全性"""
        try:
            if not data_list or len(data_list) == 0:
                print(f"⚠️ 回退特征提取：数据为空，返回零特征")
                return np.zeros(60, dtype=float)
            
            features = []
            
            # 基础频率特征（10维）
            for tail in range(10):
                try:
                    count = 0
                    total_periods = min(15, len(data_list))
                    for i in range(total_periods):
                        if tail in data_list[i].get('tails', []):
                            count += 1
                    frequency = count / total_periods if total_periods > 0 else 0.0
                    
                    # 确保频率在合理范围内
                    if math.isnan(frequency) or math.isinf(frequency):
                        frequency = 0.0
                    else:
                        frequency = max(0.0, min(1.0, frequency))
                    
                    features.append(frequency)
                except Exception as e:
                    print(f"   ⚠️ 计算尾数{tail}频率失败: {e}")
                    features.append(0.0)
            
            # 短期频率特征（10维）
            for tail in range(10):
                try:
                    count = 0
                    total_periods = min(5, len(data_list))
                    for i in range(total_periods):
                        if tail in data_list[i].get('tails', []):
                            count += 1
                    frequency = count / total_periods if total_periods > 0 else 0.0
                    
                    if math.isnan(frequency) or math.isinf(frequency):
                        frequency = 0.0
                    else:
                        frequency = max(0.0, min(1.0, frequency))
                    
                    features.append(frequency)
                except Exception as e:
                    print(f"   ⚠️ 计算尾数{tail}短期频率失败: {e}")
                    features.append(0.0)
            
            # 连续性特征（10维）
            for tail in range(10):
                try:
                    consecutive = 0
                    for period in data_list[:10]:
                        if tail in period.get('tails', []):
                            consecutive += 1
                        else:
                            break
                    
                    consecutive_ratio = consecutive / 10.0
                    if math.isnan(consecutive_ratio) or math.isinf(consecutive_ratio):
                        consecutive_ratio = 0.0
                    else:
                        consecutive_ratio = max(0.0, min(1.0, consecutive_ratio))
                    
                    features.append(consecutive_ratio)
                except Exception as e:
                    print(f"   ⚠️ 计算尾数{tail}连续性失败: {e}")
                    features.append(0.0)
            
            # 间隔特征（10维）
            for tail in range(10):
                try:
                    last_seen = -1
                    for i, period in enumerate(data_list[:15]):
                        if tail in period.get('tails', []):
                            last_seen = i
                            break
                    
                    if last_seen >= 0:
                        interval_ratio = (last_seen + 1) / 15.0
                    else:
                        interval_ratio = 1.0  # 很久没出现
                    
                    if math.isnan(interval_ratio) or math.isinf(interval_ratio):
                        interval_ratio = 1.0
                    else:
                        interval_ratio = max(0.0, min(1.0, interval_ratio))
                    
                    features.append(interval_ratio)
                except Exception as e:
                    print(f"   ⚠️ 计算尾数{tail}间隔失败: {e}")
                    features.append(1.0)
            
            # 趋势特征（10维）
            for tail in range(10):
                try:
                    if len(data_list) >= 10:
                        mid = 5
                        early_count = sum(1 for period in data_list[mid:10] if tail in period.get('tails', []))
                        late_count = sum(1 for period in data_list[:mid] if tail in period.get('tails', []))
                        trend = (late_count - early_count) / 5.0
                        
                        if math.isnan(trend) or math.isinf(trend):
                            trend = 0.0
                        else:
                            trend = max(-1.0, min(1.0, trend))
                    else:
                        trend = 0.0
                    
                    features.append(trend)
                except Exception as e:
                    print(f"   ⚠️ 计算尾数{tail}趋势失败: {e}")
                    features.append(0.0)
            
            # 统计特征（10维）
            try:
                tail_counts_per_period = []
                for period in data_list[:15]:
                    count = len(period.get('tails', []))
                    tail_counts_per_period.append(count)
                
                if tail_counts_per_period:
                    mean_count = np.mean(tail_counts_per_period)
                    std_count = np.std(tail_counts_per_period)
                    min_count = np.min(tail_counts_per_period)
                    max_count = np.max(tail_counts_per_period)
                    median_count = np.median(tail_counts_per_period)
                    
                    # 验证统计值
                    stats = [mean_count, std_count, min_count, max_count, median_count]
                    for i, stat in enumerate(stats):
                        if math.isnan(stat) or math.isinf(stat):
                            stats[i] = 0.0
                        else:
                            stats[i] = float(stat)
                    
                    features.extend(stats)
                else:
                    features.extend([0.0] * 5)
            except Exception as e:
                print(f"   ⚠️ 计算统计特征失败: {e}")
                features.extend([0.0] * 5)
            
            # 补充特征到60维
            while len(features) < 60:
                features.append(0.0)
            
            # 确保只有60维
            features = features[:60]
            
            # 转换为numpy数组并验证
            features_array = np.array(features, dtype=float)
            
            # 最终验证
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                print(f"⚠️ 回退特征提取仍包含无效值，使用零数组")
                features_array = np.zeros(60, dtype=float)
            
            print(f"✅ 回退特征提取完成，维度: {features_array.shape}")
            return features_array
            
        except Exception as e:
            print(f"❌ 回退特征提取失败: {e}")
            return np.zeros(60, dtype=float)
    
    def create_model_specific_features(self, base_features, data_list):
        """为每个模型创建专属特征"""
        model_features = {}
    
        # 提取历史数据用于特征工程
        latest_tails = set(data_list[0].get('tails', [])) if data_list else set()
    
        # 为每个River模型创建专属特征
        model_configs = [
            ('hoeffding_tree', self._create_decision_tree_features),
            ('hoeffding_adaptive', self._create_adaptive_tree_features),
            ('logistic', self._create_logistic_features),
            ('naive_bayes', self._create_independence_features),
            ('naive_bayes_multinomial', self._create_count_features),
            ('naive_bayes_gaussian', self._create_distribution_features),
            ('naive_bayes_mixed', self._create_mixed_features),
            ('bagging', self._create_stability_features),
            ('adaboost', self._create_error_correction_features),
            ('bagging_nb', self._create_probabilistic_stability_features),
            ('bagging_lr', self._create_linear_stability_features),
            ('pattern_matcher_strict', self._create_pattern_matching_features)
        ]
    
        for model_key, feature_creator in model_configs:
            # 直接使用模型键名，不需要复杂的匹配逻辑
            try:
                features = feature_creator(base_features, data_list)
                # 为不同的前缀版本都创建特征
                model_features[f'local_{model_key}'] = features
                model_features[f'river_{model_key}'] = features
                model_features[model_key] = features
            except Exception as e:
                print(f"为模型 {model_key} 创建特征失败: {e}")
                # 使用默认特征
                default_features = {f'feature_{i}': base_features[i] for i in range(min(30, len(base_features)))}
                model_features[f'local_{model_key}'] = default_features
                model_features[f'river_{model_key}'] = default_features
                model_features[model_key] = default_features
    
        return model_features
    
    def _create_decision_tree_features(self, base_features, data_list):
        """为决策树创建分割导向的特征"""
        features = {}
        
        # 创建明显的分割特征
        for tail in range(10):
            # 二元分割特征：是否在最新期出现
            features[f'is_in_latest_{tail}'] = 1.0 if (data_list and tail in data_list[0].get('tails', [])) else 0.0
            
            # 频率分割特征：高频/中频/低频
            recent_count = sum(1 for i in range(min(10, len(data_list))) if tail in data_list[i].get('tails', []))
            if recent_count >= 6:
                features[f'freq_category_{tail}'] = 2.0  # 高频
            elif recent_count >= 3:
                features[f'freq_category_{tail}'] = 1.0  # 中频
            else:
                features[f'freq_category_{tail}'] = 0.0  # 低频
            
            # 间隔分割特征
            last_appearance = -1
            for i, period in enumerate(data_list):
                if tail in period.get('tails', []):
                    last_appearance = i
                    break
            
            if last_appearance == 0:
                features[f'gap_category_{tail}'] = 0.0  # 刚出现
            elif last_appearance <= 2:
                features[f'gap_category_{tail}'] = 1.0  # 近期出现
            elif last_appearance <= 5:
                features[f'gap_category_{tail}'] = 2.0  # 中期出现
            else:
                features[f'gap_category_{tail}'] = 3.0  # 远期或未出现
        
        return features
    
    def _create_adaptive_tree_features(self, base_features, data_list):
        """为自适应树创建变化检测特征"""
        features = {}
        
        for tail in range(10):
            # 趋势变化特征
            recent_5 = [1 if tail in data_list[i].get('tails', []) else 0 for i in range(min(5, len(data_list)))]
            older_5 = [1 if tail in data_list[i].get('tails', []) else 0 for i in range(5, min(10, len(data_list)))]
            
            recent_avg = sum(recent_5) / len(recent_5) if recent_5 else 0
            older_avg = sum(older_5) / len(older_5) if older_5 else 0
            
            features[f'trend_change_{tail}'] = recent_avg - older_avg
            
            # 波动性特征
            if len(recent_5) > 1:
                variance = sum((x - recent_avg) ** 2 for x in recent_5) / len(recent_5)
                features[f'volatility_{tail}'] = variance
            else:
                features[f'volatility_{tail}'] = 0.0
            
            # 适应性信号
            changes = sum(1 for i in range(1, len(recent_5)) if recent_5[i] != recent_5[i-1])
            features[f'adaptation_signal_{tail}'] = changes / max(1, len(recent_5) - 1)
        
        return features
    
    def _create_logistic_features(self, base_features, data_list):
        """为逻辑回归创建线性可分特征"""
        features = {}
        
        for tail in range(10):
            # 线性趋势特征
            appearances = []
            for i, period in enumerate(data_list[:15]):
                appearances.append(1 if tail in period.get('tails', []) else 0)
            
            if len(appearances) > 1:
                # 计算线性趋势斜率
                n = len(appearances)
                x_sum = sum(range(n))
                y_sum = sum(appearances)
                xy_sum = sum(i * appearances[i] for i in range(n))
                x2_sum = sum(i * i for i in range(n))
                
                denominator = n * x2_sum - x_sum * x_sum
                if denominator != 0:
                    slope = (n * xy_sum - x_sum * y_sum) / denominator
                    features[f'linear_trend_{tail}'] = slope
                else:
                    features[f'linear_trend_{tail}'] = 0.0
            else:
                features[f'linear_trend_{tail}'] = 0.0
            
            # 加权频率（线性权重）
            weighted_sum = sum((15 - i) * (1 if tail in data_list[i].get('tails', []) else 0) 
                             for i in range(min(15, len(data_list))))
            weight_total = sum(15 - i for i in range(min(15, len(data_list))))
            features[f'weighted_freq_{tail}'] = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        return features
    
    def _create_independence_features(self, base_features, data_list):
        """为独立性假设创建特征"""
        features = {}
        
        for tail in range(10):
            # 每个尾数独立的出现概率
            total_periods = len(data_list)
            appearances = sum(1 for period in data_list if tail in period.get('tails', []))
            features[f'independent_prob_{tail}'] = appearances / total_periods if total_periods > 0 else 0.1
            
            # 条件独立特征（忽略其他尾数的影响）
            features[f'marginal_freq_{tail}'] = appearances / 10.0 if total_periods > 0 else 0.1
            
            # 先验概率特征
            features[f'prior_belief_{tail}'] = 0.1  # 均匀先验
        
        return features
    
    def _create_count_features(self, base_features, data_list):
        """为多项式朴素贝叶斯创建计数特征"""
        features = {}
        
        for tail in range(10):
            # 原始计数特征
            count_5 = sum(1 for i in range(min(5, len(data_list))) if tail in data_list[i].get('tails', []))
            count_10 = sum(1 for i in range(min(10, len(data_list))) if tail in data_list[i].get('tails', []))
            count_20 = sum(1 for i in range(min(20, len(data_list))) if tail in data_list[i].get('tails', []))
            
            features[f'count_5_{tail}'] = float(count_5)
            features[f'count_10_{tail}'] = float(count_10)
            features[f'count_20_{tail}'] = float(count_20)
            
            # 计数比例特征
            features[f'count_ratio_{tail}'] = count_5 / max(1, count_10)
        
        return features
    
    def _create_distribution_features(self, base_features, data_list):
        """为高斯朴素贝叶斯创建分布特征"""
        features = {}
        
        for tail in range(10):
            # 时序分布特征
            positions = []  # 该尾数出现的位置
            for i, period in enumerate(data_list[:20]):
                if tail in period.get('tails', []):
                    positions.append(i)
            
            if positions:
                mean_position = sum(positions) / len(positions)
                features[f'mean_position_{tail}'] = mean_position / 20.0
                
                if len(positions) > 1:
                    variance = sum((p - mean_position) ** 2 for p in positions) / len(positions)
                    features[f'position_variance_{tail}'] = variance / 400.0  # 归一化
                else:
                    features[f'position_variance_{tail}'] = 0.5
            else:
                features[f'mean_position_{tail}'] = 1.0  # 表示很久没出现
                features[f'position_variance_{tail}'] = 1.0
        
        return features
    
    def _create_mixed_features(self, base_features, data_list):
        """为混合朴素贝叶斯创建混合类型特征"""
        features = {}
        
        for tail in range(10):
            # 混合二元特征
            features[f'binary_recent_{tail}'] = 1.0 if (data_list and tail in data_list[0].get('tails', [])) else 0.0
            
            # 混合连续特征
            recent_freq = sum(1 for i in range(min(8, len(data_list))) if tail in data_list[i].get('tails', []))
            features[f'continuous_freq_{tail}'] = recent_freq / 8.0
            
            # 混合分类特征  
            if recent_freq >= 5:
                features[f'category_{tail}'] = 2.0  # 热门
            elif recent_freq >= 2:
                features[f'category_{tail}'] = 1.0  # 普通
            else:
                features[f'category_{tail}'] = 0.0  # 冷门
        
        return features
    
    def _create_stability_features(self, base_features, data_list):
        """为基础装袋创建稳定性特征"""
        features = {}
        
        for tail in range(10):
            # 稳定性指标：方差
            recent_appearances = [1 if tail in data_list[i].get('tails', []) else 0 
                                for i in range(min(12, len(data_list)))]
            if len(recent_appearances) > 1:
                mean_val = sum(recent_appearances) / len(recent_appearances)
                variance = sum((x - mean_val) ** 2 for x in recent_appearances) / len(recent_appearances)
                features[f'stability_{tail}'] = 1.0 - variance  # 低方差 = 高稳定性
            else:
                features[f'stability_{tail}'] = 0.5
            
            # 一致性特征
            features[f'consistency_{tail}'] = mean_val if len(recent_appearances) > 0 else 0.5
        
        return features
    
    def _create_error_correction_features(self, base_features, data_list):
        """为AdaBoost创建错误修正特征"""
        features = {}
        
        for tail in range(10):
            # 错误修正信号：与预期偏差
            expected_freq = 0.1  # 理论期望频率
            actual_freq = sum(1 for period in data_list[:10] if tail in period.get('tails', [])) / min(10, len(data_list))
            
            features[f'error_signal_{tail}'] = abs(actual_freq - expected_freq)
            features[f'correction_need_{tail}'] = max(0, expected_freq - actual_freq)  # 需要向上修正
            
            # 累积错误
            features[f'cumulative_error_{tail}'] = (actual_freq - expected_freq) ** 2
        
        return features
    
    def _create_probabilistic_stability_features(self, base_features, data_list):
        """为朴素贝叶斯装袋创建概率稳定性特征"""
        features = {}
        
        # 结合概率和稳定性
        for tail in range(10):
            # 概率特征
            prob = sum(1 for period in data_list[:8] if tail in period.get('tails', [])) / min(8, len(data_list))
            features[f'prob_{tail}'] = prob
            
            # 稳定概率
            segments = []
            for start in range(0, min(12, len(data_list)), 3):
                segment = data_list[start:start+3]
                segment_prob = sum(1 for period in segment if tail in period.get('tails', [])) / len(segment)
                segments.append(segment_prob)
            
            if len(segments) > 1:
                prob_variance = sum((p - prob) ** 2 for p in segments) / len(segments)
                features[f'prob_stability_{tail}'] = 1.0 - prob_variance
            else:
                features[f'prob_stability_{tail}'] = 0.5
        
        return features
    
    def _create_linear_stability_features(self, base_features, data_list):
        """为逻辑回归装袋创建线性稳定性特征"""
        features = {}
        
        for tail in range(10):
            # 线性特征
            weights = [0.8, 0.6, 0.4, 0.2]  # 递减权重  
            weighted_sum = 0
            weight_total = 0
            
            for i, weight in enumerate(weights):
                if i < len(data_list):
                    if tail in data_list[i].get('tails', []):
                        weighted_sum += weight
                    weight_total += weight
            
            features[f'linear_weighted_{tail}'] = weighted_sum / weight_total if weight_total > 0 else 0
            
            # 线性稳定性
            features[f'linear_stability_{tail}'] = 1.0 - abs(0.5 - features[f'linear_weighted_{tail}'])
        
        return features
    
    def _create_pattern_matching_features(self, base_features, data_list):
        """为历史模式匹配创建特征"""
        features = {}
        
        # 保持原有特征格式
        for i in range(len(base_features)):
            features[f'feature_{i}'] = base_features[i] if i < len(base_features) else 0.0
        
        return features
    
    def _apply_intelligent_feature_processing(self, features: np.ndarray) -> np.ndarray:
        """应用智能特征处理流水线"""
        try:
            # 确保输入是一维numpy数组
            if isinstance(features, (list, tuple)):
                # 先清理列表中的无效值
                cleaned_list = []
                for item in features:
                    try:
                        if item is None:
                            cleaned_list.append(0.0)
                        elif isinstance(item, (int, float)):
                            if math.isnan(item) or math.isinf(item):
                                cleaned_list.append(0.0)
                            else:
                                cleaned_list.append(float(item))
                        else:
                            cleaned_list.append(float(item))
                    except (ValueError, TypeError):
                        cleaned_list.append(0.0)
                features = np.array(cleaned_list, dtype=float)
            else:
                try:
                    features = np.array(features, dtype=float)
                except (ValueError, TypeError):
                    print(f"⚠️ 无法转换特征为numpy数组，使用零数组: {type(features)}")
                    features = np.zeros(60, dtype=float)
        
            # 确保是一维数组
            if features.ndim > 1:
                features = features.flatten()
        
            # 清理数组中的无效值
            if len(features) > 0:
                # 替换NaN和无穷大值
                nan_mask = np.isnan(features)
                inf_mask = np.isinf(features)
                features[nan_mask] = 0.0
                features[inf_mask] = 0.0
                
                # 确保数据类型正确
                features = features.astype(float)
                
                # 最终验证
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"⚠️ 特征数组仍包含无效值，重新初始化")
                    features = np.zeros(len(features), dtype=float)
            else:
                print(f"⚠️ 特征数组为空，使用默认长度60")
                features = np.zeros(60, dtype=float)
        
            print(f"🔧 输入特征维度: {features.shape}, 类型: {features.dtype}, 有效值数量: {np.sum(~np.isnan(features))}")
        
            # 1. 时序特征增强
            if self.timeseries_enhancer:
                enhanced_features = self.timeseries_enhancer.enhance_features(features)
            else:
                enhanced_features = features
        
            # 检查和修复增强特征
            if not isinstance(enhanced_features, np.ndarray):
                enhanced_features = np.array(enhanced_features, dtype=float)
            if enhanced_features.ndim > 1:
                enhanced_features = enhanced_features.flatten()
            enhanced_features = enhanced_features.astype(float)
        
            print(f"🔧 增强后特征维度: {enhanced_features.shape}")
        
            # 2. 特征交互组合
            if self.feature_combiner:
                combined_features = self.feature_combiner.create_interaction_features(enhanced_features)
            else:
                combined_features = enhanced_features
        
            # 检查和修复组合特征
            if not isinstance(combined_features, np.ndarray):
                combined_features = np.array(combined_features, dtype=float)
            if combined_features.ndim > 1:
                combined_features = combined_features.flatten()
            combined_features = combined_features.astype(float)
        
            print(f"🔧 组合后特征维度: {combined_features.shape}")
        
            # 3. 动态特征选择
            if self.feature_selector:
                selected_features, selected_indices = self.feature_selector.select_features(combined_features)
            else:
                selected_features = combined_features
        
            # 检查和修复选择特征
            if not isinstance(selected_features, np.ndarray):
                selected_features = np.array(selected_features, dtype=float)
            if selected_features.ndim > 1:
                selected_features = selected_features.flatten()
            selected_features = selected_features.astype(float)
        
            print(f"🔧 选择后特征维度: {selected_features.shape}")
        
            # 4. 自适应特征加权
            if self.feature_weighter:
                weighted_features = self.feature_weighter.apply_weights(selected_features)
            else:
                weighted_features = selected_features
        
            # 检查和修复加权特征
            if not isinstance(weighted_features, np.ndarray):
                weighted_features = np.array(weighted_features, dtype=float)
            if weighted_features.ndim > 1:
                weighted_features = weighted_features.flatten()
            weighted_features = weighted_features.astype(float)
        
            print(f"🔧 加权后特征维度: {weighted_features.shape}")
        
            # 5. 特征质量评估（每10次执行一次）
            self.feature_processing_stats['enhancement_count'] += 1
            if self.feature_processing_stats['enhancement_count'] % 10 == 0 and self.feature_assessor:
                current_accuracy = 0.5  # 默认准确率，实际使用时需要传入
                quality_report = self.feature_assessor.assess_feature_quality(weighted_features, current_accuracy)
                self.feature_processing_stats['last_quality_report'] = quality_report
                self.feature_processing_stats['quality_assessments'] += 1
            
                # 根据质量报告调整参数
                if quality_report['overall_quality'] < 0.5:
                    print(f"⚠️ 特征质量较低({quality_report['overall_quality']:.3f})，建议: {quality_report['recommendations'][:2]}")
        
            # 确保输出维度一致（填充或截断到目标维度）
            target_dim = 60  # 保持与原始系统兼容
            if len(weighted_features) > target_dim:
                final_features = weighted_features[:target_dim]
            else:
                padding_size = target_dim - len(weighted_features)
                if padding_size > 0:
                    padding = np.zeros(padding_size, dtype=float)
                    final_features = np.concatenate([weighted_features, padding])
                else:
                    final_features = weighted_features
        
            # 最终检查
            final_features = final_features.astype(float)
            print(f"🔧 最终特征维度: {final_features.shape}")
        
            return final_features
        
        except Exception as e:
            print(f"智能特征处理失败: {e}")
            import traceback
            traceback.print_exc()
        
            # 回退到基础特征
            try:
                if len(features) >= 60:
                    return features[:60].astype(float)
                else:
                    padding_size = 60 - len(features)
                    padding = np.zeros(padding_size, dtype=float)
                    return np.concatenate([features.astype(float), padding])
            except Exception as fallback_error:
                print(f"回退处理也失败: {fallback_error}")
                return np.zeros(60, dtype=float)
    
    def extract_tail_specific_features(self, data_list, tail):
        """提取特定尾数的特征"""
        if not data_list:
            return {}
        
        features = {}
        
        # 最近5期出现情况
        recent_5_appearances = []
        for i, period in enumerate(data_list[:5]):
            appeared = tail in period.get('tails', [])
            recent_5_appearances.append(appeared)
        features['recent_5_pattern'] = recent_5_appearances
        features['recent_5_count'] = sum(recent_5_appearances)
        
        # 最近10期统计
        recent_10_count = sum(1 for period in data_list[:10] if tail in period.get('tails', []))
        features['recent_10_count'] = recent_10_count
        features['recent_10_frequency'] = recent_10_count / min(10, len(data_list))
        
        # 连续性分析
        consecutive_count = 0
        for period in data_list:
            if tail in period.get('tails', []):
                consecutive_count += 1
            else:
                break
        features['consecutive_appearances'] = consecutive_count
        
        # 距离上次出现的间隔
        last_appearance = -1
        for i, period in enumerate(data_list):
            if tail in period.get('tails', []):
                last_appearance = i
                break
        features['last_appearance_distance'] = last_appearance
        
        # 在最新期中的状态
        features['in_latest_period'] = tail in data_list[0].get('tails', [])
        
        return features
    
    def update_components_with_learning_result(self, features, accuracy):
        """使用学习结果更新智能特征处理组件"""
        try:
            # 更新特征选择器
            if self.feature_selector:
                self.feature_selector.update_feature_importance(features, accuracy)
            
            # 更新特征交互组合器
            if self.feature_combiner:
                self.feature_combiner.update_interaction_scores(features, accuracy)
            
            # 更新自适应权重器
            if self.feature_weighter:
                self.feature_weighter.update_weights(features, accuracy)
            
            # 统计更新
            self.feature_processing_stats['selection_updates'] += 1
            
        except Exception as e:
            print(f"更新智能特征处理组件失败: {e}")