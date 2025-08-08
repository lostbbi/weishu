#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
冷门挖掘器预测模型集 - 科研级完整实现
专门针对"杀多赔少"策略中被忽视的冷门尾数组合挖掘
"""

import numpy as np
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 数据结构定义
@dataclass
class ColdnessProfile:
    """冷门度档案数据结构"""
    tail: int
    current_absence_length: int      # 当前缺席长度
    max_absence_length: int          # 历史最大缺席长度
    avg_absence_length: float        # 平均缺席长度
    total_appearances: int           # 总出现次数
    appearance_frequency: float      # 出现频率
    coldness_index: float           # 综合冷门指数
    revival_probability: float       # 复出概率
    last_appearance_period: int      # 上次出现期数索引

@dataclass
class RevivalSignal:
    """复出信号数据结构"""
    tail: int
    signal_strength: float          # 信号强度 0-1
    signal_type: str               # 'cyclic', 'compensation', 'pattern_break'
    expected_timing: int           # 预期复出时机
    confidence: float              # 置信度
    supporting_evidence: List[str] # 支持证据

class ColdnessLevel(Enum):
    """冷门程度等级"""
    EXTREMELY_COLD = 5    # 极度冷门
    VERY_COLD = 4        # 非常冷门
    MODERATELY_COLD = 3  # 中等冷门
    SLIGHTLY_COLD = 2    # 轻微冷门
    NEUTRAL = 1          # 中性
    WARM = 0             # 温热

class UnpopularDigger:
    """
    冷门挖掘器 - 科研级完整实现
    
    核心功能：
    1. 多维度冷门分析算法
    2. 长期冷门追踪系统
    3. 复出时机预测模型
    4. 反热门策略生成
    5. 动态冷门度评估
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化冷门挖掘器"""
        self.config = config or self._get_default_config()
        
        # 核心分析组件
        self.coldness_analyzer = ColdnessAnalyzer()
        self.revival_predictor = RevivalPredictor()
        self.anti_hot_strategist = AntiHotStrategist()
        self.pattern_detector = ColdPatternDetector()
        
        # 历史数据存储
        self.coldness_profiles = {}  # 每个尾数的冷门档案
        self.revival_history = deque(maxlen=self.config['revival_history_window'])
        self.prediction_outcomes = deque(maxlen=self.config['outcome_tracking_window'])
        
        # 学习状态
        self.total_predictions = 0
        self.successful_revivals = 0
        self.model_confidence = 0.5
        self.adaptation_rate = self.config['adaptation_rate']
        
        # 多时间尺度分析窗口
        self.analysis_windows = {
            'immediate': deque(maxlen=5),      # 最近5期
            'short_term': deque(maxlen=15),    # 短期15期
            'medium_term': deque(maxlen=40),   # 中期40期
            'long_term': deque(maxlen=100),    # 长期100期
        }
        
        # 冷门挖掘策略库
        self.digging_strategies = self._initialize_digging_strategies()
        
        print(f"🔍 冷门挖掘器初始化完成")
        print(f"   - 分析窗口: {len(self.analysis_windows)}个时间尺度")
        print(f"   - 挖掘策略: {len(self.digging_strategies)}种")
        print(f"   - 适应性学习率: {self.adaptation_rate}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'revival_history_window': 200,
            'outcome_tracking_window': 100,
            'adaptation_rate': 0.12,
            'coldness_threshold': 0.7,
            'revival_confidence_threshold': 0.6,
            'max_absence_tracking': 50,
            'pattern_matching_sensitivity': 0.8,
            'anti_hot_aggressiveness': 0.75,
            'cyclic_analysis_depth': 3,
            'compensation_psychology_weight': 0.4,
            'pattern_break_weight': 0.35,
            'frequency_deviation_weight': 0.25,
        }
    
    def predict(self, candidate_tails: List[int], historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        在筛选后的候选尾数中进行冷门挖掘预测
        
        Args:
            candidate_tails: 经过三大定律筛选的候选尾数
            historical_context: 历史上下文数据
            
        Returns:
            预测结果字典
        """
        prediction_start_time = datetime.now()
        
        if not candidate_tails:
            return {
                'success': False,
                'recommended_tails': [],
                'confidence': 0.0,
                'message': 'no_candidate_tails_provided',
                'coldness_analysis': {}
            }
        
        if len(historical_context) < 10:
            return {
                'success': False,
                'recommended_tails': [],
                'confidence': 0.0,
                'message': 'insufficient_historical_data',
                'coldness_analysis': {}
            }
        
        print(f"🔍 冷门挖掘器开始分析 {len(candidate_tails)} 个候选尾数: {candidate_tails}")
        
        # 更新所有分析窗口
        if historical_context:
            latest_period = historical_context[0]  # 最新期在前
            for window in self.analysis_windows.values():
                window.appendleft(latest_period)  # 添加到左边（最新）
        
        # === 更新冷门档案 ===
        self._update_coldness_profiles(historical_context)
        
        # === 候选尾数冷门度分析 ===
        candidate_coldness_analysis = {}
        for tail in candidate_tails:
            coldness_analysis = self.coldness_analyzer.analyze_tail_coldness(
                tail, historical_context, self.analysis_windows
            )
            candidate_coldness_analysis[tail] = coldness_analysis
        
        # === 复出时机预测 ===
        revival_predictions = {}
        for tail in candidate_tails:
            if tail in candidate_coldness_analysis:
                coldness_data = candidate_coldness_analysis[tail]
                revival_pred = self.revival_predictor.predict_revival_timing(
                    tail, coldness_data, historical_context
                )
                revival_predictions[tail] = revival_pred
        
        # === 反热门策略分析 ===
        anti_hot_analysis = self.anti_hot_strategist.analyze_anti_hot_opportunities(
            candidate_tails, historical_context, candidate_coldness_analysis
        )
        
        # === 冷门模式检测 ===
        pattern_analysis = self.pattern_detector.detect_cold_patterns(
            candidate_tails, historical_context, candidate_coldness_analysis
        )
        
        # === 综合评分与推荐生成 ===
        final_recommendations = self._generate_final_recommendations(
            candidate_tails, candidate_coldness_analysis, revival_predictions,
            anti_hot_analysis, pattern_analysis, historical_context
        )
        
        # 更新学习状态
        self._update_learning_state(final_recommendations)
        
        # 记录预测历史
        prediction_record = {
            'timestamp': prediction_start_time,
            'candidate_tails': candidate_tails,
            'recommendations': final_recommendations,
            'coldness_analysis': candidate_coldness_analysis,
            'revival_predictions': revival_predictions,
            'anti_hot_analysis': anti_hot_analysis,
            'pattern_analysis': pattern_analysis
        }
        
        self.revival_history.append(prediction_record)
        
        prediction_duration = (datetime.now() - prediction_start_time).total_seconds()
        
        result = {
            'success': True,
            'model_name': 'UnpopularDigger',
            'recommended_tails': final_recommendations['recommended_tails'],
            'confidence': final_recommendations['confidence'],
            'coldness_analysis': candidate_coldness_analysis,
            'revival_signals': revival_predictions,
            'anti_hot_opportunities': anti_hot_analysis,
            'pattern_insights': pattern_analysis,
            'strategy_reasoning': final_recommendations['reasoning'],
            'prediction_quality': final_recommendations['quality_assessment'],
            'analysis_duration': prediction_duration,
            'total_candidates_analyzed': len(candidate_tails)
        }
        
        print(f"🔍 冷门挖掘完成，推荐 {len(final_recommendations['recommended_tails'])} 个冷门尾数")
        
        return result
    
    def _update_coldness_profiles(self, historical_context: List[Dict[str, Any]]):
        """更新所有尾数的冷门档案"""
        
        for tail in range(10):
            # 计算当前缺席长度
            current_absence = 0
            for i, period in enumerate(historical_context):
                if tail not in period.get('tails', []):
                    current_absence += 1
                else:
                    break
            
            # 计算历史统计
            total_appearances = 0
            absence_lengths = []
            current_streak = 0
            is_absent = True
            
            for period in reversed(historical_context):  # 从最旧到最新
                if tail in period.get('tails', []):
                    total_appearances += 1
                    if is_absent and current_streak > 0:
                        absence_lengths.append(current_streak)
                        current_streak = 0
                    is_absent = False
                else:
                    if not is_absent:
                        current_streak = 1
                        is_absent = True
                    else:
                        current_streak += 1
            
            # 添加当前缺席期
            if is_absent and current_streak > 0:
                absence_lengths.append(current_streak)
            
            # 计算统计指标
            max_absence = max(absence_lengths) if absence_lengths else current_absence
            avg_absence = np.mean(absence_lengths) if absence_lengths else current_absence
            appearance_freq = total_appearances / len(historical_context) if historical_context else 0
            
            # 找到上次出现位置
            last_appearance_period = -1
            for i, period in enumerate(historical_context):
                if tail in period.get('tails', []):
                    last_appearance_period = i
                    break
            
            # 计算冷门指数
            coldness_index = self._calculate_coldness_index(
                current_absence, max_absence, avg_absence, appearance_freq, len(historical_context)
            )
            
            # 计算复出概率
            revival_probability = self._calculate_revival_probability(
                current_absence, avg_absence, appearance_freq
            )
            
            # 更新档案
            self.coldness_profiles[tail] = ColdnessProfile(
                tail=tail,
                current_absence_length=current_absence,
                max_absence_length=max_absence,
                avg_absence_length=avg_absence,
                total_appearances=total_appearances,
                appearance_frequency=appearance_freq,
                coldness_index=coldness_index,
                revival_probability=revival_probability,
                last_appearance_period=last_appearance_period
            )
    
    def _calculate_coldness_index(self, current_absence: int, max_absence: int, 
                                 avg_absence: float, appearance_freq: float, 
                                 total_periods: int) -> float:
        """计算综合冷门指数"""
        
        # 当前缺席权重
        absence_weight = min(1.0, current_absence / 20.0) * 0.4
        
        # 频率权重（低频率 = 冷门）
        frequency_weight = (1.0 - appearance_freq) * 0.35
        
        # 异常缺席权重
        if avg_absence > 0:
            abnormal_absence = max(0, (current_absence - avg_absence) / avg_absence)
            abnormal_weight = min(1.0, abnormal_absence) * 0.25
        else:
            abnormal_weight = 0.5
        
        coldness_index = absence_weight + frequency_weight + abnormal_weight
        return min(1.0, max(0.0, coldness_index))
    
    def _calculate_revival_probability(self, current_absence: int, 
                                     avg_absence: float, appearance_freq: float) -> float:
        """计算复出概率"""
        
        if avg_absence <= 0:
            return 0.5
        
        # 基于平均缺席长度的期望复出概率
        expected_revival = min(1.0, current_absence / (avg_absence * 1.5))
        
        # 基于历史频率的基础概率
        base_probability = appearance_freq
        
        # 补偿效应：缺席越久，复出概率越高
        compensation_factor = min(1.0, current_absence / 15.0)
        
        # 综合复出概率
        revival_prob = (expected_revival * 0.4 + base_probability * 0.3 + compensation_factor * 0.3)
        
        return min(0.95, max(0.05, revival_prob))
    
    def _generate_final_recommendations(self, candidate_tails: List[int], 
                                      coldness_analysis: Dict[int, Dict], 
                                      revival_predictions: Dict[int, Dict],
                                      anti_hot_analysis: Dict[str, Any], 
                                      pattern_analysis: Dict[str, Any],
                                      historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成最终推荐"""
        
        # 为每个候选尾数计算综合评分
        tail_scores = {}
        
        for tail in candidate_tails:
            score_components = {
                'coldness_score': 0.0,
                'revival_score': 0.0,
                'anti_hot_score': 0.0,
                'pattern_score': 0.0
            }
            
            # 冷门度评分
            if tail in coldness_analysis:
                coldness_data = coldness_analysis[tail]
                coldness_level = coldness_data.get('coldness_level', ColdnessLevel.NEUTRAL)
                
                coldness_mapping = {
                    ColdnessLevel.EXTREMELY_COLD: 1.0,
                    ColdnessLevel.VERY_COLD: 0.8,
                    ColdnessLevel.MODERATELY_COLD: 0.6,
                    ColdnessLevel.SLIGHTLY_COLD: 0.4,
                    ColdnessLevel.NEUTRAL: 0.2,
                    ColdnessLevel.WARM: 0.0
                }
                
                score_components['coldness_score'] = coldness_mapping.get(coldness_level, 0.2)
            
            # 复出时机评分
            if tail in revival_predictions:
                revival_data = revival_predictions[tail]
                revival_strength = revival_data.get('revival_strength', 0.0)
                timing_score = revival_data.get('timing_score', 0.0)
                
                score_components['revival_score'] = (revival_strength * 0.6 + timing_score * 0.4)
            
            # 反热门机会评分
            anti_hot_opportunities = anti_hot_analysis.get('tail_opportunities', {})
            if tail in anti_hot_opportunities:
                opportunity_data = anti_hot_opportunities[tail]
                score_components['anti_hot_score'] = opportunity_data.get('opportunity_strength', 0.0)
            
            # 冷门模式评分
            pattern_signals = pattern_analysis.get('tail_pattern_signals', {})
            if tail in pattern_signals:
                pattern_data = pattern_signals[tail]
                score_components['pattern_score'] = pattern_data.get('pattern_strength', 0.0)
            
            # 综合评分
            weights = self.config
            total_score = (
                score_components['coldness_score'] * 0.35 +
                score_components['revival_score'] * 0.3 +
                score_components['anti_hot_score'] * 0.2 +
                score_components['pattern_score'] * 0.15
            )
            
            tail_scores[tail] = {
                'total_score': total_score,
                'components': score_components,
                'coldness_profile': self.coldness_profiles.get(tail)
            }
        
        # 选择评分最高的尾数
        if not tail_scores:
            return {
                'recommended_tails': [],
                'confidence': 0.0,
                'reasoning': 'no_valid_candidates',
                'quality_assessment': 'poor'
            }
        
        # 排序并选择top尾数
        sorted_tails = sorted(tail_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        # 动态选择推荐数量
        recommendation_count = self._determine_recommendation_count(sorted_tails, candidate_tails)
        
        recommended_tails = [tail for tail, _ in sorted_tails[:recommendation_count]]
        
        # 计算置信度
        confidence = self._calculate_prediction_confidence(sorted_tails, anti_hot_analysis, pattern_analysis)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(recommended_tails, tail_scores, coldness_analysis, revival_predictions)
        
        # 质量评估
        quality = self._assess_prediction_quality(confidence, len(recommended_tails), tail_scores)
        
        return {
            'recommended_tails': recommended_tails,
            'confidence': confidence,
            'reasoning': reasoning,
            'quality_assessment': quality,
            'tail_scores': tail_scores,
            'selection_details': {
                'total_candidates': len(candidate_tails),
                'candidates_scored': len(tail_scores),
                'recommendation_count': recommendation_count,
                'top_score': sorted_tails[0][1]['total_score'] if sorted_tails else 0.0
            }
        }
    
    def _determine_recommendation_count(self, sorted_tails: List[Tuple], candidate_tails: List[int]) -> int:
        """动态确定推荐数量"""
        
        if not sorted_tails:
            return 0
        
        # 基于评分差异动态调整
        top_score = sorted_tails[0][1]['total_score']
        
        if top_score > 0.8:
            # 有明显的高分冷门尾数
            return 1
        elif top_score > 0.6:
            # 中等冷门程度，可以推荐1-2个
            count = 1
            if len(sorted_tails) > 1 and sorted_tails[1][1]['total_score'] > 0.5:
                count = 2
            return count
        else:
            # 冷门程度一般，最多推荐2个
            return min(2, len(sorted_tails))
    
    def _calculate_prediction_confidence(self, sorted_tails: List[Tuple], 
                                       anti_hot_analysis: Dict, pattern_analysis: Dict) -> float:
        """计算预测置信度"""
        
        if not sorted_tails:
            return 0.0
        
        confidence_factors = []
        
        # 评分质量因子
        top_score = sorted_tails[0][1]['total_score']
        score_quality = min(1.0, top_score)
        confidence_factors.append(score_quality)
        
        # 评分一致性因子
        if len(sorted_tails) > 1:
            scores = [item[1]['total_score'] for item in sorted_tails]
            score_std = np.std(scores)
            consistency = max(0.0, 1.0 - score_std)
            confidence_factors.append(consistency)
        else:
            confidence_factors.append(0.8)
        
        # 反热门信号强度
        anti_hot_strength = anti_hot_analysis.get('overall_strength', 0.5)
        confidence_factors.append(anti_hot_strength)
        
        # 模式信号强度
        pattern_strength = pattern_analysis.get('overall_pattern_strength', 0.5)
        confidence_factors.append(pattern_strength)
        
        # 历史成功率
        historical_success = self.successful_revivals / max(self.total_predictions, 1)
        confidence_factors.append(historical_success)
        
        # 数据充足性
        data_sufficiency = min(1.0, len(self.revival_history) / 50.0)
        confidence_factors.append(data_sufficiency)
        
        # 综合置信度
        overall_confidence = np.mean(confidence_factors)
        
        return min(0.95, max(0.05, overall_confidence))
    
    def _generate_reasoning(self, recommended_tails: List[int], tail_scores: Dict, 
                          coldness_analysis: Dict, revival_predictions: Dict) -> str:
        """生成推理说明"""
        
        reasoning_parts = []
        
        if not recommended_tails:
            return "无有效的冷门挖掘机会"
        
        reasoning_parts.append(f"🔍 冷门挖掘推荐: {recommended_tails}")
        
        for tail in recommended_tails:
            if tail in tail_scores:
                score_data = tail_scores[tail]
                total_score = score_data['total_score']
                components = score_data['components']
                
                # 分析主要推荐原因
                max_component = max(components.items(), key=lambda x: x[1])
                reason_type = max_component[0]
                reason_strength = max_component[1]
                
                if reason_type == 'coldness_score':
                    reasoning_parts.append(f"尾数{tail}: 冷门度高({reason_strength:.2f})")
                elif reason_type == 'revival_score':
                    reasoning_parts.append(f"尾数{tail}: 复出时机到({reason_strength:.2f})")
                elif reason_type == 'anti_hot_score':
                    reasoning_parts.append(f"尾数{tail}: 反热门机会({reason_strength:.2f})")
                elif reason_type == 'pattern_score':
                    reasoning_parts.append(f"尾数{tail}: 冷门模式信号({reason_strength:.2f})")
                
                # 添加冷门档案信息
                if tail in self.coldness_profiles:
                    profile = self.coldness_profiles[tail]
                    reasoning_parts.append(f"  缺席{profile.current_absence_length}期, 冷门指数{profile.coldness_index:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _assess_prediction_quality(self, confidence: float, recommendation_count: int, 
                                 tail_scores: Dict) -> str:
        """评估预测质量"""
        
        if confidence > 0.8 and recommendation_count > 0:
            return "excellent"
        elif confidence > 0.7 and recommendation_count > 0:
            return "good"
        elif confidence > 0.6:
            return "moderate"
        elif confidence > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _update_learning_state(self, recommendations: Dict):
        """更新学习状态"""
        
        self.total_predictions += 1
        
        # 基于推荐质量更新模型置信度
        quality = recommendations.get('quality_assessment', 'moderate')
        quality_scores = {
            'excellent': 0.9,
            'good': 0.75,
            'moderate': 0.6,
            'fair': 0.45,
            'poor': 0.3
        }
        
        quality_score = quality_scores.get(quality, 0.6)
        
        # 指数移动平均更新
        self.model_confidence = (
            self.model_confidence * (1 - self.adaptation_rate) +
            quality_score * self.adaptation_rate
        )
        
        self.model_confidence = min(0.95, max(0.05, self.model_confidence))
    
    def learn_from_outcome(self, prediction_result: Dict[str, Any], 
                          actual_outcome: List[int]) -> Dict[str, Any]:
        """从结果中学习"""
        
        recommended_tails = prediction_result.get('recommended_tails', [])
        
        # 评估推荐准确性
        successful_recommendations = len(set(recommended_tails).intersection(set(actual_outcome)))
        total_recommendations = len(recommended_tails)
        
        if total_recommendations > 0:
            recommendation_accuracy = successful_recommendations / total_recommendations
        else:
            recommendation_accuracy = 0.0
        
        # 特别评估冷门复出的成功率
        revival_success = 0.0
        if recommended_tails:
            for tail in recommended_tails:
                if tail in actual_outcome and tail in self.coldness_profiles:
                    profile = self.coldness_profiles[tail]
                    # 如果是真正的冷门尾数成功复出
                    if profile.coldness_index > 0.6:
                        revival_success += 1.0
            
            revival_success = revival_success / len(recommended_tails)
        
        # 更新成功统计
        if recommendation_accuracy > 0.6:
            self.successful_revivals += 1
        
        # 记录学习结果
        outcome_record = {
            'timestamp': datetime.now(),
            'prediction': prediction_result,
            'actual_outcome': actual_outcome,
            'recommendation_accuracy': recommendation_accuracy,
            'revival_success_rate': revival_success,
            'successful_tails': list(set(recommended_tails).intersection(set(actual_outcome))),
            'failed_tails': list(set(recommended_tails) - set(actual_outcome))
        }
        
        self.prediction_outcomes.append(outcome_record)
        
        # 自适应调整
        self._adaptive_adjustment(recommendation_accuracy, revival_success)
        
        return {
            'learning_success': True,
            'recommendation_accuracy': recommendation_accuracy,
            'revival_success_rate': revival_success,
            'successful_predictions': self.successful_revivals,
            'total_predictions': self.total_predictions,
            'model_confidence': self.model_confidence,
            'cold_revival_rate': revival_success
        }
    
    def _adaptive_adjustment(self, accuracy: float, revival_success: float):
        """自适应调整模型参数"""
        
        # 调整冷门阈值
        if accuracy > 0.8 and revival_success > 0.7:
            # 表现很好，可以提高冷门要求
            self.config['coldness_threshold'] = min(0.9, self.config['coldness_threshold'] + 0.02)
        elif accuracy < 0.4:
            # 表现不好，降低冷门要求
            self.config['coldness_threshold'] = max(0.5, self.config['coldness_threshold'] - 0.02)
        
        # 调整反热门激进程度
        if revival_success > 0.6:
            # 冷门挖掘成功，可以更激进
            self.config['anti_hot_aggressiveness'] = min(0.9, self.config['anti_hot_aggressiveness'] + 0.03)
        elif revival_success < 0.3:
            # 冷门挖掘不成功，变保守
            self.config['anti_hot_aggressiveness'] = max(0.5, self.config['anti_hot_aggressiveness'] - 0.03)
    
    def _initialize_digging_strategies(self) -> Dict[str, Dict]:
        """初始化冷门挖掘策略库"""
        
        return {
            'extreme_cold_hunter': {
                'type': 'coldness_focused',
                'min_coldness_index': 0.8,
                'description': '极度冷门猎手策略',
                'suitable_conditions': ['long_absence', 'low_frequency']
            },
            'cyclic_revival_tracker': {
                'type': 'timing_focused',
                'cycle_sensitivity': 0.7,
                'description': '周期性复出追踪策略',
                'suitable_conditions': ['regular_patterns', 'cyclic_behavior']
            },
            'compensation_psychological': {
                'type': 'psychology_focused',
                'compensation_weight': 0.8,
                'description': '补偿心理利用策略',
                'suitable_conditions': ['extreme_absence', 'player_expectations']
            },
            'anti_hot_contrarian': {
                'type': 'contrarian_focused',
                'contrarian_strength': 0.9,
                'description': '反热门对抗策略',
                'suitable_conditions': ['hot_number_dominance', 'market_bias']
            },
            'pattern_break_seeker': {
                'type': 'pattern_focused',
                'break_sensitivity': 0.75,
                'description': '模式打破寻求策略',
                'suitable_conditions': ['established_patterns', 'pattern_fatigue']
            }
        }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        
        success_rate = self.successful_revivals / self.total_predictions if self.total_predictions > 0 else 0
        
        # 冷门档案统计
        coldness_distribution = defaultdict(int)
        for profile in self.coldness_profiles.values():
            if profile.coldness_index >= 0.8:
                coldness_distribution['extremely_cold'] += 1
            elif profile.coldness_index >= 0.6:
                coldness_distribution['very_cold'] += 1
            elif profile.coldness_index >= 0.4:
                coldness_distribution['moderately_cold'] += 1
            else:
                coldness_distribution['neutral_warm'] += 1
        
        # 最近预测统计
        recent_outcomes = list(self.prediction_outcomes)[-10:]
        recent_accuracy = np.mean([outcome['recommendation_accuracy'] for outcome in recent_outcomes]) if recent_outcomes else 0
        recent_revival_success = np.mean([outcome['revival_success_rate'] for outcome in recent_outcomes]) if recent_outcomes else 0
        
        return {
            'model_name': 'UnpopularDigger',
            'total_predictions': self.total_predictions,
            'successful_revivals': self.successful_revivals,
            'success_rate': success_rate,
            'model_confidence': self.model_confidence,
            'recent_accuracy': recent_accuracy,
            'recent_revival_success_rate': recent_revival_success,
            'coldness_distribution': dict(coldness_distribution),
            'strategy_count': len(self.digging_strategies),
            'analysis_windows': {k: len(v) for k, v in self.analysis_windows.items()},
            'prediction_history_length': len(self.prediction_outcomes),
            'config_status': {
                'coldness_threshold': self.config['coldness_threshold'],
                'anti_hot_aggressiveness': self.config['anti_hot_aggressiveness'],
                'adaptation_rate': self.config['adaptation_rate']
            }
        }

# 辅助分析组件实现

class ColdnessAnalyzer:
    """冷门度分析器"""
    
    def analyze_tail_coldness(self, tail: int, historical_context: List[Dict[str, Any]], 
                             analysis_windows: Dict[str, deque]) -> Dict[str, Any]:
        """分析尾数冷门度"""
        
        # 多时间尺度分析
        multi_scale_analysis = {}
        
        for window_name, window_data in analysis_windows.items():
            if not window_data:
                continue
                
            window_list = list(window_data)
            
            # 计算该窗口内的统计
            appearances = sum(1 for period in window_list if tail in period.get('tails', []))
            total_periods = len(window_list)
            
            if total_periods > 0:
                frequency = appearances / total_periods
                absence_ratio = 1.0 - frequency
                
                # 计算当前缺席长度
                current_absence = 0
                for period in window_list:
                    if tail not in period.get('tails', []):
                        current_absence += 1
                    else:
                        break
                
                multi_scale_analysis[window_name] = {
                    'frequency': frequency,
                    'absence_ratio': absence_ratio,
                    'current_absence': current_absence,
                    'appearances': appearances,
                    'total_periods': total_periods
                }
            else:
                multi_scale_analysis[window_name] = {
                    'frequency': 0.0,
                    'absence_ratio': 1.0,
                    'current_absence': 0,
                    'appearances': 0,
                    'total_periods': 0
                }
        
        # 综合冷门度评估
        coldness_level = self._determine_coldness_level(multi_scale_analysis)
        overall_coldness_score = self._calculate_overall_coldness(multi_scale_analysis)
        
        return {
            'tail': tail,
            'coldness_level': coldness_level,
            'overall_coldness_score': overall_coldness_score,
            'multi_scale_analysis': multi_scale_analysis,
            'analysis_timestamp': datetime.now()
        }
    
    def _determine_coldness_level(self, analysis: Dict[str, Dict]) -> ColdnessLevel:
        """确定冷门等级"""
        
        # 使用长期数据判断
        long_term_data = analysis.get('long_term', {})
        medium_term_data = analysis.get('medium_term', {})
        
        if not long_term_data and not medium_term_data:
            return ColdnessLevel.NEUTRAL
        
        # 优先使用长期数据
        primary_data = long_term_data if long_term_data else medium_term_data
        
        absence_ratio = primary_data.get('absence_ratio', 0.5)
        current_absence = primary_data.get('current_absence', 0)
        
        if absence_ratio >= 0.9 and current_absence >= 15:
            return ColdnessLevel.EXTREMELY_COLD
        elif absence_ratio >= 0.8 and current_absence >= 10:
            return ColdnessLevel.VERY_COLD
        elif absence_ratio >= 0.7 and current_absence >= 6:
            return ColdnessLevel.MODERATELY_COLD
        elif absence_ratio >= 0.6 and current_absence >= 3:
            return ColdnessLevel.SLIGHTLY_COLD
        elif absence_ratio < 0.3:
            return ColdnessLevel.WARM
        else:
            return ColdnessLevel.NEUTRAL
    
    def _calculate_overall_coldness(self, analysis: Dict[str, Dict]) -> float:
        """计算综合冷门度分数"""
        
        scores = []
        weights = {
            'immediate': 0.15,
            'short_term': 0.25,
            'medium_term': 0.35,
            'long_term': 0.25
        }
        
        for window_name, weight in weights.items():
            if window_name in analysis:
                data = analysis[window_name]
                absence_ratio = data.get('absence_ratio', 0.5)
                current_absence = data.get('current_absence', 0)
                
                # 综合评分
                window_score = (absence_ratio * 0.7 + min(1.0, current_absence / 10.0) * 0.3)
                scores.append(window_score * weight)
        
        return sum(scores) if scores else 0.5

class RevivalPredictor:
    """复出时机预测器"""
    
    def predict_revival_timing(self, tail: int, coldness_data: Dict[str, Any], 
                              historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测复出时机"""
        
        # 获取冷门度数据
        coldness_level = coldness_data.get('coldness_level', ColdnessLevel.NEUTRAL)
        multi_scale = coldness_data.get('multi_scale_analysis', {})
        
        # 分析历史复出模式
        revival_patterns = self._analyze_historical_revivals(tail, historical_context)
        
        # 计算复出强度
        revival_strength = self._calculate_revival_strength(coldness_level, revival_patterns, multi_scale)
        
        # 计算时机得分
        timing_score = self._calculate_timing_score(tail, historical_context, revival_patterns)
        
        # 生成复出信号
        revival_signals = self._generate_revival_signals(tail, revival_strength, timing_score, revival_patterns)
        
        return {
            'tail': tail,
            'revival_strength': revival_strength,
            'timing_score': timing_score,
            'revival_signals': revival_signals,
            'historical_patterns': revival_patterns,
            'expected_revival_period': self._estimate_revival_period(revival_patterns, timing_score)
        }
    
    def _analyze_historical_revivals(self, tail: int, historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析历史复出模式"""
        
        revival_intervals = []
        last_appearance = -1
        
        for i, period in enumerate(historical_context):
            if tail in period.get('tails', []):
                if last_appearance >= 0:
                    interval = i - last_appearance
                    revival_intervals.append(interval)
                last_appearance = i
        
        if revival_intervals:
            avg_interval = np.mean(revival_intervals)
            std_interval = np.std(revival_intervals)
            min_interval = min(revival_intervals)
            max_interval = max(revival_intervals)
        else:
            avg_interval = 10.0
            std_interval = 5.0
            min_interval = 1
            max_interval = 20
        
        return {
            'revival_intervals': revival_intervals,
            'avg_interval': avg_interval,
            'std_interval': std_interval,
            'min_interval': min_interval,
            'max_interval': max_interval,
            'pattern_consistency': 1.0 - (std_interval / max(avg_interval, 1.0)) if avg_interval > 0 else 0.0
        }
    
    def _calculate_revival_strength(self, coldness_level: ColdnessLevel, 
                                   revival_patterns: Dict, multi_scale: Dict) -> float:
        """计算复出强度"""
        
        # 基于冷门等级的基础强度
        level_strength_map = {
            ColdnessLevel.EXTREMELY_COLD: 0.9,
            ColdnessLevel.VERY_COLD: 0.75,
            ColdnessLevel.MODERATELY_COLD: 0.6,
            ColdnessLevel.SLIGHTLY_COLD: 0.45,
            ColdnessLevel.NEUTRAL: 0.3,
            ColdnessLevel.WARM: 0.1
        }
        
        base_strength = level_strength_map.get(coldness_level, 0.3)
        
        # 基于历史模式的调整
        pattern_consistency = revival_patterns.get('pattern_consistency', 0.5)
        pattern_adjustment = pattern_consistency * 0.2
        
        # 基于多时间尺度的调整
        long_term_data = multi_scale.get('long_term', {})
        if long_term_data:
            long_term_absence = long_term_data.get('absence_ratio', 0.5)
            long_term_adjustment = long_term_absence * 0.15
        else:
            long_term_adjustment = 0.0
        
        final_strength = base_strength + pattern_adjustment + long_term_adjustment
        return min(1.0, max(0.0, final_strength))
    
    def _calculate_timing_score(self, tail: int, historical_context: List[Dict[str, Any]], 
                               revival_patterns: Dict) -> float:
        """计算时机得分"""
        
        if not historical_context:
            return 0.5
        
        # 计算自上次出现的间隔
        periods_since_last = 0
        for period in historical_context:
            if tail not in period.get('tails', []):
                periods_since_last += 1
            else:
                break
        
        # 基于平均间隔计算时机成熟度
        avg_interval = revival_patterns.get('avg_interval', 10.0)
        
        if avg_interval > 0:
            timing_maturity = min(1.0, periods_since_last / avg_interval)
        else:
            timing_maturity = 0.5
        
        # 过度延迟惩罚
        if periods_since_last > avg_interval * 2:
            timing_maturity *= 0.8
        
        return timing_maturity
    
    def _generate_revival_signals(self, tail: int, revival_strength: float, 
                                 timing_score: float, revival_patterns: Dict) -> List[RevivalSignal]:
        """生成复出信号"""
        
        signals = []
        
        # 强度信号
        if revival_strength > 0.7:
            signals.append(RevivalSignal(
                tail=tail,
                signal_strength=revival_strength,
                signal_type='high_coldness_revival',
                expected_timing=int(revival_patterns.get('avg_interval', 10) * 0.8),
                confidence=revival_strength * 0.9,
                supporting_evidence=['extremely_cold_status', 'long_absence']
            ))
        
        # 时机信号
        if timing_score > 0.8:
            signals.append(RevivalSignal(
                tail=tail,
                signal_strength=timing_score,
                signal_type='timing_maturity',
                expected_timing=1,
                confidence=timing_score * 0.8,
                supporting_evidence=['interval_analysis', 'timing_maturity']
            ))
        
        # 模式信号
        pattern_consistency = revival_patterns.get('pattern_consistency', 0.0)
        if pattern_consistency > 0.6:
            signals.append(RevivalSignal(
                tail=tail,
                signal_strength=pattern_consistency,
                signal_type='pattern_revival',
                expected_timing=int(revival_patterns.get('avg_interval', 10)),
                confidence=pattern_consistency * 0.7,
                supporting_evidence=['historical_pattern', 'cyclic_behavior']
            ))
        
        return signals
    
    def _estimate_revival_period(self, revival_patterns: Dict, timing_score: float) -> int:
        """估计复出期数"""
        
        avg_interval = revival_patterns.get('avg_interval', 10.0)
        
        if timing_score > 0.8:
            return 1  # 即将复出
        elif timing_score > 0.6:
            return int(avg_interval * 0.3)
        else:
            return int(avg_interval * 0.7)

class AntiHotStrategist:
    """反热门策略师"""
    
    def analyze_anti_hot_opportunities(self, candidate_tails: List[int], 
                                     historical_context: List[Dict[str, Any]], 
                                     coldness_analysis: Dict[int, Dict]) -> Dict[str, Any]:
        """分析反热门机会"""
        
        if len(historical_context) < 5:
            return {'overall_strength': 0.0, 'tail_opportunities': {}}
        
        # 识别当前热门尾数
        hot_tails = self._identify_hot_tails(historical_context)
        
        # 分析反热门机会
        tail_opportunities = {}
        
        for tail in candidate_tails:
            if tail not in hot_tails:  # 只分析非热门的候选尾数
                opportunity_strength = self._calculate_anti_hot_opportunity(
                    tail, hot_tails, historical_context, coldness_analysis.get(tail, {})
                )
                
                if opportunity_strength > 0.3:
                    tail_opportunities[tail] = {
                        'opportunity_strength': opportunity_strength,
                        'anti_hot_reasoning': self._generate_anti_hot_reasoning(tail, hot_tails, opportunity_strength)
                    }
        
        # 计算整体反热门强度
        overall_strength = np.mean(list(opp['opportunity_strength'] for opp in tail_opportunities.values())) if tail_opportunities else 0.0
        
        return {
            'overall_strength': overall_strength,
            'hot_tails_identified': hot_tails,
            'tail_opportunities': tail_opportunities,
            'anti_hot_strategy': self._recommend_anti_hot_strategy(hot_tails, tail_opportunities)
        }
    
    def _identify_hot_tails(self, historical_context: List[Dict[str, Any]]) -> Set[int]:
        """识别热门尾数"""
        
        # 分析最近10期
        recent_periods = historical_context[:10] if len(historical_context) >= 10 else historical_context
        
        tail_counts = defaultdict(int)
        for period in recent_periods:
            for tail in period.get('tails', []):
                tail_counts[tail] += 1
        
        total_periods = len(recent_periods)
        hot_threshold = total_periods * 0.6  # 60%以上出现率视为热门
        
        hot_tails = {tail for tail, count in tail_counts.items() if count >= hot_threshold}
        
        return hot_tails
    
    def _calculate_anti_hot_opportunity(self, tail: int, hot_tails: Set[int], 
                                      historical_context: List[Dict[str, Any]], 
                                      coldness_data: Dict) -> float:
        """计算反热门机会强度"""
        
        opportunity_factors = []
        
        # 冷门度因子
        coldness_score = coldness_data.get('overall_coldness_score', 0.5)
        opportunity_factors.append(coldness_score * 0.4)
        
        # 热门对比因子
        if hot_tails:
            hot_contrast = len(hot_tails) / 10.0  # 热门尾数占比
            opportunity_factors.append(hot_contrast * 0.3)
        else:
            opportunity_factors.append(0.0)
        
        # 反向心理因子
        reverse_psychology = self._calculate_reverse_psychology_factor(tail, hot_tails, historical_context)
        opportunity_factors.append(reverse_psychology * 0.3)
        
        return sum(opportunity_factors)
    
    def _calculate_reverse_psychology_factor(self, tail: int, hot_tails: Set[int], 
                                           historical_context: List[Dict[str, Any]]) -> float:
        """计算反向心理因子"""
        
        # 如果热门尾数很多，冷门尾数的反向心理价值增加
        hot_dominance = len(hot_tails) / 10.0
        
        # 如果该尾数长期被忽视，反向价值增加
        recent_5_periods = historical_context[:5] if len(historical_context) >= 5 else historical_context
        recent_appearances = sum(1 for period in recent_5_periods if tail in period.get('tails', []))
        
        neglect_factor = 1.0 - (recent_appearances / max(len(recent_5_periods), 1))
        
        return (hot_dominance * 0.6 + neglect_factor * 0.4)
    
    def _generate_anti_hot_reasoning(self, tail: int, hot_tails: Set[int], strength: float) -> str:
        """生成反热门推理"""
        
        if strength > 0.8:
            return f"尾数{tail}完全避开热门区域{hot_tails}，具有强反热门价值"
        elif strength > 0.6:
            return f"尾数{tail}与热门{hot_tails}形成对比，适合反热门策略"
        else:
            return f"尾数{tail}有一定反热门机会"
    
    def _recommend_anti_hot_strategy(self, hot_tails: Set[int], opportunities: Dict) -> str:
        """推荐反热门策略"""
        
        if len(hot_tails) >= 4:
            return "aggressive_anti_hot"  # 激进反热门
        elif len(hot_tails) >= 2:
            return "moderate_anti_hot"    # 温和反热门
        else:
            return "minimal_anti_hot"     # 轻微反热门

class ColdPatternDetector:
    """冷门模式检测器"""
    
    def detect_cold_patterns(self, candidate_tails: List[int], 
                           historical_context: List[Dict[str, Any]], 
                           coldness_analysis: Dict[int, Dict]) -> Dict[str, Any]:
        """检测冷门模式"""
        
        if len(historical_context) < 15:
            return {'overall_pattern_strength': 0.0, 'tail_pattern_signals': {}}
        
        tail_pattern_signals = {}
        
        for tail in candidate_tails:
            pattern_signals = self._analyze_tail_patterns(tail, historical_context, coldness_analysis.get(tail, {}))
            
            if pattern_signals['overall_strength'] > 0.3:
                tail_pattern_signals[tail] = pattern_signals
        
        # 计算整体模式强度
        overall_strength = np.mean([signals['overall_strength'] for signals in tail_pattern_signals.values()]) if tail_pattern_signals else 0.0
        
        return {
            'overall_pattern_strength': overall_strength,
            'tail_pattern_signals': tail_pattern_signals,
            'detected_pattern_types': self._summarize_pattern_types(tail_pattern_signals)
        }
    
    def _analyze_tail_patterns(self, tail: int, historical_context: List[Dict[str, Any]], 
                              coldness_data: Dict) -> Dict[str, Any]:
        """分析单个尾数的模式"""
        
        pattern_strengths = {}
        
        # 周期性模式检测
        cyclic_strength = self._detect_cyclic_patterns(tail, historical_context)
        pattern_strengths['cyclic'] = cyclic_strength
        
        # 补偿模式检测
        compensation_strength = self._detect_compensation_patterns(tail, historical_context)
        pattern_strengths['compensation'] = compensation_strength
        
        # 断裂模式检测
        break_strength = self._detect_break_patterns(tail, historical_context)
        pattern_strengths['break'] = break_strength
        
        # 综合模式强度
        overall_strength = np.mean(list(pattern_strengths.values()))
        
        return {
            'overall_strength': overall_strength,
            'pattern_breakdown': pattern_strengths,
            'dominant_pattern': max(pattern_strengths.items(), key=lambda x: x[1])[0] if pattern_strengths else 'none'
        }
    
    def _detect_cyclic_patterns(self, tail: int, historical_context: List[Dict[str, Any]]) -> float:
        """检测周期性模式"""
        
        # 记录出现位置
        appearance_positions = []
        for i, period in enumerate(historical_context):
            if tail in period.get('tails', []):
                appearance_positions.append(i)
        
        if len(appearance_positions) < 3:
            return 0.0
        
        # 计算间隔
        intervals = [appearance_positions[i+1] - appearance_positions[i] for i in range(len(appearance_positions)-1)]
        
        if not intervals:
            return 0.0
        
        # 检测周期性
        interval_consistency = 1.0 - (np.std(intervals) / max(np.mean(intervals), 1.0))
        
        return max(0.0, interval_consistency)
    
    def _detect_compensation_patterns(self, tail: int, historical_context: List[Dict[str, Any]]) -> float:
        """检测补偿模式"""
        
        # 寻找长期缺席后的复出模式
        absence_revival_pairs = []
        
        current_absence = 0
        for period in historical_context:
            if tail not in period.get('tails', []):
                current_absence += 1
            else:
                if current_absence > 5:  # 缺席5期以上后复出
                    absence_revival_pairs.append(current_absence)
                current_absence = 0
        
        if len(absence_revival_pairs) < 2:
            return 0.0
        
        # 检测补偿模式的一致性
        avg_absence = np.mean(absence_revival_pairs)
        consistency = 1.0 - (np.std(absence_revival_pairs) / max(avg_absence, 1.0))
        
        return max(0.0, consistency)
    
    def _detect_break_patterns(self, tail: int, historical_context: List[Dict[str, Any]]) -> float:
        """检测断裂模式"""
        
        # 检测是否打破了某种既定模式
        if len(historical_context) < 10:
            return 0.0
        
        # 分析前半段和后半段的出现模式
        mid_point = len(historical_context) // 2
        first_half = historical_context[mid_point:]
        second_half = historical_context[:mid_point]
        
        first_appearances = sum(1 for period in first_half if tail in period.get('tails', []))
        second_appearances = sum(1 for period in second_half if tail in period.get('tails', []))
        
        first_freq = first_appearances / len(first_half) if first_half else 0
        second_freq = second_appearances / len(second_half) if second_half else 0
        
        # 频率变化幅度
        frequency_change = abs(first_freq - second_freq)
        
        return min(1.0, frequency_change * 2.0)
    
    def _summarize_pattern_types(self, tail_signals: Dict) -> List[str]:
        """总结检测到的模式类型"""
        
        pattern_types = set()
        
        for tail_data in tail_signals.values():
            dominant_pattern = tail_data.get('dominant_pattern', 'none')
            if dominant_pattern != 'none':
                pattern_types.add(dominant_pattern)
        
        return list(pattern_types)