#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
反向心理学预测模型集 - 科研级完整实现
专门针对"杀多赔少"策略的反向心理操控
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
class PopularitySignal:
    """大众偏好信号数据结构"""
    timestamp: datetime
    popular_tails: List[int]       # 热门尾数
    popularity_scores: Dict[int, float]  # 每个尾数的热门度分数
    crowd_sentiment: str           # 'bullish', 'bearish', 'neutral'
    trap_probability: float        # 陷阱概率
    confidence: float             # 置信度

@dataclass
class ReversalStrategy:
    """反向策略数据结构"""
    strategy_type: str            # 'avoid_hot', 'chase_cold', 'break_pattern'
    target_tails: List[int]       # 目标尾数
    avoidance_tails: List[int]    # 避开尾数
    reversal_strength: float     # 反向强度
    expected_effectiveness: float # 预期有效性
    reasoning: str               # 策略理由

class PopularityLevel(Enum):
    """热门程度等级"""
    EXTREMELY_HOT = 5     # 极度热门
    VERY_HOT = 4         # 非常热门
    MODERATELY_HOT = 3   # 中等热门
    NEUTRAL = 2          # 中性
    COLD = 1             # 冷门
    EXTREMELY_COLD = 0   # 极度冷门

class ReversePsychologyPredictor:
    """
    反向心理学预测模型 - 科研级完整实现
    
    核心功能：
    1. 大众偏好模式分析
    2. 热门陷阱识别
    3. 反向选择策略生成
    4. 群体心理逆向工程
    5. 动态反向强度调整
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化反向心理学预测器"""
        self.config = config or self._get_default_config()
        
        # 核心分析组件
        self.popularity_analyzer = PopularityAnalyzer()
        self.trap_detector = HotNumberTrapDetector()
        self.reversal_strategist = ReversalStrategist()
        self.crowd_psychology_engine = CrowdPsychologyEngine()
        
        # 历史数据存储
        self.popularity_history = deque(maxlen=self.config['max_history_length'])
        self.reversal_outcomes = deque(maxlen=self.config['outcome_tracking_window'])
        self.strategy_performance = {}
        
        # 学习状态
        self.total_predictions = 0
        self.successful_reversals = 0
        self.model_confidence = 0.5
        self.adaptation_rate = self.config['adaptation_rate']
        
        # 多时间尺度分析
        self.analysis_windows = {
            'immediate': deque(maxlen=3),      # 最近3期
            'short_term': deque(maxlen=10),    # 短期10期
            'medium_term': deque(maxlen=30),   # 中期30期
            'long_term': deque(maxlen=100),    # 长期100期
        }
        
        # 反向策略库
        self.strategy_library = self._initialize_strategy_library()
        
        print(f"🔄 反向心理学预测器初始化完成")
        print(f"   - 分析窗口: {len(self.analysis_windows)}个时间尺度")
        print(f"   - 策略库: {len(self.strategy_library)}种反向策略")
        print(f"   - 适应性学习率: {self.adaptation_rate}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_length': 200,
            'outcome_tracking_window': 100,
            'adaptation_rate': 0.15,
            'popularity_threshold': 0.7,
            'reversal_confidence_threshold': 0.6,
            'trap_detection_sensitivity': 0.75,
            'strategy_effectiveness_threshold': 0.55,
            'crowd_psychology_weight': 0.3,
            'historical_pattern_weight': 0.4,
            'real_time_signal_weight': 0.3,
            'max_reversal_strength': 0.9,
            'min_reversal_strength': 0.1,
        }
    
    def predict(self, period_data: Dict[str, Any], 
               historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成反向心理学预测
        
        Args:
            period_data: 当期数据
            historical_context: 历史上下文数据
            
        Returns:
            预测结果字典
        """
        prediction_start_time = datetime.now()
        
        # 更新所有分析窗口
        for window in self.analysis_windows.values():
            window.append(period_data)
        
        # === 大众偏好分析 ===
        popularity_analysis = self.popularity_analyzer.analyze_crowd_preferences(
            historical_context, self.analysis_windows
        )
        
        # === 热门陷阱检测 ===
        trap_analysis = self.trap_detector.detect_popularity_traps(
            period_data, historical_context, popularity_analysis
        )
        
        # === 群体心理分析 ===
        psychology_analysis = self.crowd_psychology_engine.analyze_group_psychology(
            historical_context, popularity_analysis, trap_analysis
        )
        
        # === 反向策略生成 ===
        reversal_strategies = self.reversal_strategist.generate_reversal_strategies(
            popularity_analysis, trap_analysis, psychology_analysis, historical_context
        )
        
        # === 策略优化与选择 ===
        optimal_strategy = self._select_optimal_strategy(
            reversal_strategies, historical_context, psychology_analysis
        )
        
        # === 预测结果综合 ===
        prediction_result = self._synthesize_prediction_result(
            optimal_strategy, popularity_analysis, trap_analysis, 
            psychology_analysis, historical_context
        )
        
        # 更新学习状态
        self._update_learning_state(prediction_result)
        
        # 记录预测历史
        self.popularity_history.append({
            'timestamp': prediction_start_time,
            'popularity_analysis': popularity_analysis,
            'trap_analysis': trap_analysis,
            'psychology_analysis': psychology_analysis,
            'selected_strategy': optimal_strategy,
            'prediction_result': prediction_result
        })
        
        prediction_duration = (datetime.now() - prediction_start_time).total_seconds()
        prediction_result['analysis_duration'] = prediction_duration
        
        return prediction_result
    
    def _select_optimal_strategy(self, reversal_strategies: List[ReversalStrategy], 
                                historical_context: List[Dict[str, Any]], 
                                psychology_analysis: Dict[str, Any]) -> ReversalStrategy:
        """选择最优反向策略"""
        
        if not reversal_strategies:
            # 生成默认策略
            return ReversalStrategy(
                strategy_type='defensive',
                target_tails=[],
                avoidance_tails=[],
                reversal_strength=0.5,
                expected_effectiveness=0.4,
                reasoning='无明显反向信号，采用保守策略'
            )
        
        # 评估每个策略的有效性
        strategy_scores = []
        
        for strategy in reversal_strategies:
            effectiveness_score = self._evaluate_strategy_effectiveness(
                strategy, historical_context, psychology_analysis
            )
            
            # 历史性能加权
            historical_performance = self.strategy_performance.get(
                strategy.strategy_type, 0.5
            )
            
            # 当前市场适配度
            market_fit_score = self._calculate_market_fit_score(
                strategy, psychology_analysis
            )
            
            # 综合评分
            total_score = (
                effectiveness_score * 0.4 +
                historical_performance * 0.3 +
                market_fit_score * 0.3
            )
            
            strategy_scores.append((strategy, total_score))
        
        # 选择评分最高的策略
        best_strategy, best_score = max(strategy_scores, key=lambda x: x[1])
        
        # 动态调整策略强度
        best_strategy.reversal_strength = self._adjust_reversal_strength(
            best_strategy, best_score, psychology_analysis
        )
        
        return best_strategy
    
    def _evaluate_strategy_effectiveness(self, strategy: ReversalStrategy, 
                                       historical_context: List[Dict[str, Any]], 
                                       psychology_analysis: Dict[str, Any]) -> float:
        """评估策略有效性"""
        
        base_effectiveness = strategy.expected_effectiveness
        
        # 历史验证
        historical_accuracy = self._validate_strategy_historically(
            strategy, historical_context[-20:] if len(historical_context) >= 20 else historical_context
        )
        
        # 心理学一致性
        psychology_consistency = self._check_psychology_consistency(
            strategy, psychology_analysis
        )
        
        # 市场时机
        timing_score = self._evaluate_market_timing(
            strategy, historical_context
        )
        
        # 综合评估
        effectiveness = (
            base_effectiveness * 0.3 +
            historical_accuracy * 0.4 +
            psychology_consistency * 0.2 +
            timing_score * 0.1
        )
        
        return min(1.0, max(0.0, effectiveness))
    
    def _validate_strategy_historically(self, strategy: ReversalStrategy, 
                                      historical_data: List[Dict[str, Any]]) -> float:
        """历史验证策略有效性"""
        
        if len(historical_data) < 5:
            return 0.5
        
        correct_predictions = 0
        total_validations = 0
        
        # 模拟历史应用策略
        for i in range(len(historical_data) - 1):
            current_period = historical_data[i]
            next_period = historical_data[i + 1]
            
            # 检查策略是否在当期适用
            if self._is_strategy_applicable(strategy, current_period, historical_data[:i+1]):
                total_validations += 1
                
                # 检查下期结果是否符合策略预期
                if strategy.target_tails:
                    # 目标尾数策略
                    next_tails = set(next_period.get('tails', []))
                    predicted_tails = set(strategy.target_tails)
                    
                    # 计算命中率
                    hits = len(next_tails.intersection(predicted_tails))
                    if hits > 0:
                        correct_predictions += 1
                
                if strategy.avoidance_tails:
                    # 避开尾数策略
                    next_tails = set(next_period.get('tails', []))
                    avoid_tails = set(strategy.avoidance_tails)
                    
                    # 如果成功避开了热门陷阱
                    avoided_traps = len(avoid_tails - next_tails)
                    if avoided_traps >= len(avoid_tails) * 0.7:
                        correct_predictions += 1
        
        return correct_predictions / total_validations if total_validations > 0 else 0.5
    
    def _is_strategy_applicable(self, strategy: ReversalStrategy, 
                              period_data: Dict[str, Any], 
                              context: List[Dict[str, Any]]) -> bool:
        """检查策略是否适用于特定期次"""
        
        if strategy.strategy_type == 'avoid_hot':
            # 避开热门策略：需要检测到热门数字
            return self._detect_hot_numbers_in_period(period_data, context)
        
        elif strategy.strategy_type == 'chase_cold':
            # 追逐冷门策略：需要检测到长期冷门
            return self._detect_cold_numbers_in_period(period_data, context)
        
        elif strategy.strategy_type == 'break_pattern':
            # 打破模式策略：需要检测到明显模式
            return self._detect_patterns_in_period(period_data, context)
        
        else:
            return True  # 默认策略总是适用
    
    def _detect_hot_numbers_in_period(self, period_data: Dict[str, Any], 
                                     context: List[Dict[str, Any]]) -> bool:
        """检测期次中是否存在热门数字"""
        
        if len(context) < 5:
            return False
        
        recent_context = context[-5:]
        tail_frequencies = defaultdict(int)
        
        for period in recent_context:
            for tail in period.get('tails', []):
                tail_frequencies[tail] += 1
        
        # 检查是否有热门数字（出现频率 >= 60%）
        hot_threshold = len(recent_context) * 0.6
        hot_tails = [tail for tail, freq in tail_frequencies.items() if freq >= hot_threshold]
        
        current_tails = set(period_data.get('tails', []))
        return any(tail in current_tails for tail in hot_tails)
    
    def _detect_cold_numbers_in_period(self, period_data: Dict[str, Any], 
                                      context: List[Dict[str, Any]]) -> bool:
        """检测期次中是否存在冷门数字"""
        
        if len(context) < 10:
            return False
        
        # 分析最近10期，找出长期未出现的尾数
        recent_context = context[-10:]
        tail_appearances = defaultdict(int)
        
        for period in recent_context:
            for tail in period.get('tails', []):
                tail_appearances[tail] += 1
        
        # 找出冷门尾数（出现次数 <= 20%）
        cold_threshold = len(recent_context) * 0.2
        cold_tails = [tail for tail in range(10) 
                     if tail_appearances[tail] <= cold_threshold]
        
        current_tails = set(period_data.get('tails', []))
        return any(tail in current_tails for tail in cold_tails)
    
    def _detect_patterns_in_period(self, period_data: Dict[str, Any], 
                                  context: List[Dict[str, Any]]) -> bool:
        """检测期次中是否存在明显模式"""
        
        if len(context) < 6:
            return False
        
        recent_context = context[-6:]
        
        # 检测连续性模式
        for tail in range(10):
            consecutive_count = 0
            for period in reversed(recent_context):
                if tail in period.get('tails', []):
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= 3:  # 连续3次以上出现
                return tail in period_data.get('tails', [])
        
        # 检测周期性模式
        for tail in range(10):
            positions = []
            for i, period in enumerate(recent_context):
                if tail in period.get('tails', []):
                    positions.append(i)
            
            if len(positions) >= 3:
                # 检查是否有规律间隔
                intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if len(set(intervals)) <= 2:  # 间隔模式较规律
                    return tail in period_data.get('tails', [])
        
        return False
    
    def _check_psychology_consistency(self, strategy: ReversalStrategy, 
                                    psychology_analysis: Dict[str, Any]) -> float:
        """检查策略与群体心理的一致性"""
        
        crowd_sentiment = psychology_analysis.get('crowd_sentiment', 'neutral')
        herd_behavior_strength = psychology_analysis.get('herd_behavior_strength', 0.5)
        
        consistency_score = 0.5
        
        if strategy.strategy_type == 'avoid_hot':
            # 避开热门策略与强从众行为一致
            if crowd_sentiment == 'bullish' and herd_behavior_strength > 0.7:
                consistency_score += 0.3
            elif herd_behavior_strength > 0.5:
                consistency_score += 0.2
        
        elif strategy.strategy_type == 'chase_cold':
            # 追逐冷门策略与反向思维一致
            if crowd_sentiment == 'bearish' and herd_behavior_strength < 0.3:
                consistency_score += 0.3
            elif herd_behavior_strength < 0.5:
                consistency_score += 0.2
        
        elif strategy.strategy_type == 'break_pattern':
            # 打破模式策略需要适中的群体行为
            if 0.3 <= herd_behavior_strength <= 0.7:
                consistency_score += 0.25
        
        # 考虑反向心理强度
        reversal_readiness = psychology_analysis.get('reversal_readiness', 0.5)
        if reversal_readiness > 0.6:
            consistency_score += 0.15
        elif reversal_readiness < 0.3:
            consistency_score -= 0.1
        
        return min(1.0, max(0.0, consistency_score))
    
    def _evaluate_market_timing(self, strategy: ReversalStrategy, 
                              historical_context: List[Dict[str, Any]]) -> float:
        """评估市场时机"""
        
        if len(historical_context) < 5:
            return 0.5
        
        timing_score = 0.5
        
        # 分析最近的市场波动性
        recent_periods = historical_context[-5:]
        tail_variability = self._calculate_tail_variability(recent_periods)
        
        if strategy.strategy_type == 'avoid_hot':
            # 热门避开策略在高波动性时更有效
            if tail_variability > 0.7:
                timing_score += 0.3
            elif tail_variability > 0.5:
                timing_score += 0.2
        
        elif strategy.strategy_type == 'chase_cold':
            # 冷门追逐策略在低波动性时更有效
            if tail_variability < 0.3:
                timing_score += 0.3
            elif tail_variability < 0.5:
                timing_score += 0.2
        
        elif strategy.strategy_type == 'break_pattern':
            # 模式打破策略在中等波动性时更有效
            if 0.4 <= tail_variability <= 0.6:
                timing_score += 0.25
        
        # 考虑策略使用频率（避免过度使用）
        recent_strategy_usage = self._calculate_recent_strategy_usage(
            strategy.strategy_type, historical_context
        )
        
        if recent_strategy_usage > 0.8:
            timing_score -= 0.2  # 过度使用降低有效性
        elif recent_strategy_usage < 0.2:
            timing_score += 0.1  # 稀缺使用提升有效性
        
        return min(1.0, max(0.0, timing_score))
    
    def _calculate_tail_variability(self, periods: List[Dict[str, Any]]) -> float:
        """计算尾数变异性"""
        
        if len(periods) < 2:
            return 0.5
        
        all_period_tails = []
        for period in periods:
            period_tails = set(period.get('tails', []))
            all_period_tails.append(period_tails)
        
        # 计算期间相似度
        similarities = []
        for i in range(len(all_period_tails) - 1):
            current_set = all_period_tails[i]
            next_set = all_period_tails[i + 1]
            
            if current_set or next_set:
                intersection = len(current_set.intersection(next_set))
                union = len(current_set.union(next_set))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        # 变异性是相似度的逆
        avg_similarity = np.mean(similarities) if similarities else 0.5
        variability = 1.0 - avg_similarity
        
        return variability
    
    def _calculate_recent_strategy_usage(self, strategy_type: str, 
                                       historical_context: List[Dict[str, Any]]) -> float:
        """计算最近策略使用频率"""
        
        # 这是一个简化实现，实际应该基于历史策略记录
        # 这里模拟基于策略类型的使用模式
        
        usage_patterns = {
            'avoid_hot': 0.4,      # 避开热门策略使用频率
            'chase_cold': 0.3,     # 追逐冷门策略使用频率
            'break_pattern': 0.2,  # 打破模式策略使用频率
            'defensive': 0.1       # 防御策略使用频率
        }
        
        return usage_patterns.get(strategy_type, 0.3)
    
    def _calculate_market_fit_score(self, strategy: ReversalStrategy, 
                                  psychology_analysis: Dict[str, Any]) -> float:
        """计算策略市场适配度"""
        
        market_conditions = psychology_analysis.get('market_conditions', {})
        crowd_consensus_strength = market_conditions.get('consensus_strength', 0.5)
        market_volatility = market_conditions.get('volatility', 0.5)
        
        fit_score = 0.5
        
        if strategy.strategy_type == 'avoid_hot':
            # 避开热门在强共识和高波动时适配度高
            fit_score += crowd_consensus_strength * 0.3
            fit_score += market_volatility * 0.2
        
        elif strategy.strategy_type == 'chase_cold':
            # 追逐冷门在弱共识和低波动时适配度高
            fit_score += (1.0 - crowd_consensus_strength) * 0.3
            fit_score += (1.0 - market_volatility) * 0.2
        
        elif strategy.strategy_type == 'break_pattern':
            # 打破模式在中等条件时适配度高
            consensus_fit = 1.0 - abs(crowd_consensus_strength - 0.5) * 2
            volatility_fit = 1.0 - abs(market_volatility - 0.5) * 2
            fit_score += consensus_fit * 0.25
            fit_score += volatility_fit * 0.25
        
        return min(1.0, max(0.0, fit_score))
    
    def _adjust_reversal_strength(self, strategy: ReversalStrategy, 
                                 strategy_score: float, 
                                 psychology_analysis: Dict[str, Any]) -> float:
        """动态调整反向强度"""
        
        base_strength = strategy.reversal_strength
        
        # 基于策略评分调整
        score_adjustment = (strategy_score - 0.5) * 0.4
        
        # 基于群体心理强度调整
        crowd_influence = psychology_analysis.get('crowd_influence_strength', 0.5)
        crowd_adjustment = crowd_influence * 0.3
        
        # 基于历史成功率调整
        historical_success = self.strategy_performance.get(strategy.strategy_type, 0.5)
        history_adjustment = (historical_success - 0.5) * 0.2
        
        # 基于当前模型置信度调整
        confidence_adjustment = (self.model_confidence - 0.5) * 0.1
        
        # 综合调整
        adjusted_strength = (
            base_strength + 
            score_adjustment + 
            crowd_adjustment + 
            history_adjustment + 
            confidence_adjustment
        )
        
        # 约束在配置范围内
        min_strength = self.config['min_reversal_strength']
        max_strength = self.config['max_reversal_strength']
        
        return min(max_strength, max(min_strength, adjusted_strength))
    
    def _synthesize_prediction_result(self, optimal_strategy: ReversalStrategy, 
                                    popularity_analysis: Dict[str, Any], 
                                    trap_analysis: Dict[str, Any], 
                                    psychology_analysis: Dict[str, Any], 
                                    historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """综合生成预测结果"""
        
        # 基于最优策略生成推荐
        if optimal_strategy.target_tails:
            recommended_tails = optimal_strategy.target_tails
        else:
            # 如果没有明确目标，基于反向逻辑推荐
            recommended_tails = self._generate_reverse_recommendations(
                popularity_analysis, trap_analysis, optimal_strategy
            )
        
        # 计算置信度
        confidence = self._calculate_prediction_confidence(
            optimal_strategy, popularity_analysis, psychology_analysis
        )
        
        # 生成避开建议
        avoid_tails = optimal_strategy.avoidance_tails or self._identify_avoid_tails(
            popularity_analysis, trap_analysis
        )
        
        # 生成详细分析
        detailed_analysis = self._generate_detailed_analysis(
            optimal_strategy, popularity_analysis, trap_analysis, 
            psychology_analysis, historical_context
        )
        
        return {
            'model_name': 'ReversePsychologyPredictor',
            'recommended_tails': recommended_tails,
            'avoid_tails': avoid_tails,
            'confidence': confidence,
            'reversal_strength': optimal_strategy.reversal_strength,
            'strategy_type': optimal_strategy.strategy_type,
            'reasoning': optimal_strategy.reasoning,
            'popularity_analysis': popularity_analysis,
            'trap_analysis': trap_analysis,
            'psychology_analysis': psychology_analysis,
            'detailed_analysis': detailed_analysis,
            'prediction_quality': self._assess_prediction_quality(confidence, optimal_strategy),
            'risk_assessment': self._assess_prediction_risk(optimal_strategy, psychology_analysis)
        }
    
    def _generate_reverse_recommendations(self, popularity_analysis: Dict[str, Any], 
                                        trap_analysis: Dict[str, Any], 
                                        strategy: ReversalStrategy) -> List[int]:
        """基于反向逻辑生成推荐"""
        
        popular_tails = set(popularity_analysis.get('popular_tails', []))
        trap_tails = set(trap_analysis.get('identified_traps', []))
        
        # 反向选择：避开热门和陷阱，选择冷门
        all_tails = set(range(10))
        avoid_set = popular_tails.union(trap_tails)
        
        candidate_tails = all_tails - avoid_set
        
        if not candidate_tails:
            # 如果所有尾数都被排除，选择风险最小的
            risk_scores = popularity_analysis.get('tail_risk_scores', {})
            candidate_tails = [min(risk_scores.keys(), key=lambda x: risk_scores[x])] if risk_scores else [0]
        
        # 根据策略强度决定推荐数量
        reversal_strength = strategy.reversal_strength
        if reversal_strength > 0.8:
            num_recommendations = 1  # 强反向：精准单选
        elif reversal_strength > 0.6:
            num_recommendations = 2  # 中强反向：双选
        else:
            num_recommendations = min(3, len(candidate_tails))  # 温和反向：多选
        
        # 选择推荐尾数
        if isinstance(candidate_tails, set):
            candidate_list = sorted(list(candidate_tails))
        else:
            candidate_list = list(candidate_tails)
        
        return candidate_list[:num_recommendations]
    
    def _identify_avoid_tails(self, popularity_analysis: Dict[str, Any], 
                            trap_analysis: Dict[str, Any]) -> List[int]:
        """识别需要避开的尾数"""
        
        # 合并热门尾数和陷阱尾数
        popular_tails = set(popularity_analysis.get('popular_tails', []))
        trap_tails = set(trap_analysis.get('identified_traps', []))
        high_risk_tails = set(popularity_analysis.get('high_risk_tails', []))
        
        avoid_set = popular_tails.union(trap_tails).union(high_risk_tails)
        
        return sorted(list(avoid_set))
    
    def _calculate_prediction_confidence(self, strategy: ReversalStrategy, 
                                       popularity_analysis: Dict[str, Any], 
                                       psychology_analysis: Dict[str, Any]) -> float:
        """计算预测置信度"""
        
        # 策略基础置信度
        base_confidence = strategy.expected_effectiveness
        
        # 群体心理一致性加成
        psychology_consistency = self._check_psychology_consistency(strategy, psychology_analysis)
        
        # 热门度信号强度
        popularity_strength = popularity_analysis.get('signal_strength', 0.5)
        
        # 历史模型表现
        model_performance = self.model_confidence
        
        # 数据充足性
        data_sufficiency = min(1.0, len(self.popularity_history) / 50.0)
        
        # 综合置信度
        confidence = (
            base_confidence * 0.3 +
            psychology_consistency * 0.25 +
            popularity_strength * 0.2 +
            model_performance * 0.15 +
            data_sufficiency * 0.1
        )
        
        return min(0.95, max(0.05, confidence))
    
    def _generate_detailed_analysis(self, strategy: ReversalStrategy, 
                                  popularity_analysis: Dict[str, Any], 
                                  trap_analysis: Dict[str, Any], 
                                  psychology_analysis: Dict[str, Any], 
                                  historical_context: List[Dict[str, Any]]) -> str:
        """生成详细分析报告"""
        
        analysis_parts = []
        
        # 策略分析
        analysis_parts.append(f"🔄 反向策略: {strategy.strategy_type}")
        analysis_parts.append(f"💪 反向强度: {strategy.reversal_strength:.2f}")
        analysis_parts.append(f"🎯 策略理由: {strategy.reasoning}")
        
        # 大众偏好分析
        popular_tails = popularity_analysis.get('popular_tails', [])
        if popular_tails:
            analysis_parts.append(f"📈 检测到热门尾数: {popular_tails}")
        
        # 陷阱分析
        trap_tails = trap_analysis.get('identified_traps', [])
        if trap_tails:
            analysis_parts.append(f"🕳️ 识别出陷阱尾数: {trap_tails}")
        
        # 群体心理分析
        crowd_sentiment = psychology_analysis.get('crowd_sentiment', 'neutral')
        herd_strength = psychology_analysis.get('herd_behavior_strength', 0.5)
        analysis_parts.append(f"👥 群体情绪: {crowd_sentiment}, 从众强度: {herd_strength:.2f}")
        
        # 反向逻辑说明
        if strategy.target_tails:
            analysis_parts.append(f"🎯 反向推荐: {strategy.target_tails} (逆向选择)")
        if strategy.avoidance_tails:
            analysis_parts.append(f"🚫 反向避开: {strategy.avoidance_tails} (热门陷阱)")
        
        return " | ".join(analysis_parts)
    
    def _assess_prediction_quality(self, confidence: float, strategy: ReversalStrategy) -> str:
        """评估预测质量"""
        
        if confidence > 0.8 and strategy.reversal_strength > 0.7:
            return "excellent"
        elif confidence > 0.7 and strategy.reversal_strength > 0.6:
            return "good"
        elif confidence > 0.6:
            return "moderate"
        elif confidence > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_prediction_risk(self, strategy: ReversalStrategy, 
                              psychology_analysis: Dict[str, Any]) -> str:
        """评估预测风险"""
        
        crowd_consensus = psychology_analysis.get('crowd_consensus_strength', 0.5)
        market_volatility = psychology_analysis.get('market_volatility', 0.5)
        
        if strategy.reversal_strength > 0.8 and crowd_consensus > 0.8:
            return "high"  # 强反向对抗强共识，风险高
        elif strategy.reversal_strength > 0.6 and market_volatility > 0.7:
            return "moderate-high"  # 中强反向在高波动环境
        elif strategy.reversal_strength < 0.4:
            return "low"  # 温和反向，风险低
        else:
            return "moderate"
    
    def _update_learning_state(self, prediction_result: Dict[str, Any]):
        """更新学习状态"""
        
        self.total_predictions += 1
        
        # 更新模型置信度（基于预测质量的自适应）
        prediction_quality = prediction_result.get('prediction_quality', 'moderate')
        quality_scores = {
            'excellent': 0.9,
            'good': 0.75,
            'moderate': 0.6,
            'fair': 0.45,
            'poor': 0.3
        }
        
        quality_score = quality_scores.get(prediction_quality, 0.6)
        
        # 使用指数移动平均更新置信度
        self.model_confidence = (
            self.model_confidence * (1 - self.adaptation_rate) +
            quality_score * self.adaptation_rate
        )
        
        # 确保置信度在合理范围内
        self.model_confidence = min(0.95, max(0.05, self.model_confidence))
    
    def learn_from_outcome(self, prediction_result: Dict[str, Any], 
                          actual_outcome: List[int]) -> Dict[str, Any]:
        """从结果中学习"""
        
        # 评估预测准确性
        recommended_tails = prediction_result.get('recommended_tails', [])
        avoid_tails = prediction_result.get('avoid_tails', [])
        
        # 推荐准确性
        recommendation_hits = len(set(recommended_tails).intersection(set(actual_outcome)))
        recommendation_accuracy = recommendation_hits / len(recommended_tails) if recommended_tails else 0
        
        # 避开准确性（成功避开了建议避开的尾数）
        avoided_successfully = len(set(avoid_tails) - set(actual_outcome))
        avoidance_accuracy = avoided_successfully / len(avoid_tails) if avoid_tails else 1.0
        
        # 综合准确性
        overall_accuracy = (recommendation_accuracy * 0.6 + avoidance_accuracy * 0.4)
        
        # 更新策略性能
        strategy_type = prediction_result.get('strategy_type', 'unknown')
        if strategy_type in self.strategy_performance:
            current_performance = self.strategy_performance[strategy_type]
            # 使用指数移动平均更新
            self.strategy_performance[strategy_type] = (
                current_performance * 0.8 + overall_accuracy * 0.2
            )
        else:
            self.strategy_performance[strategy_type] = overall_accuracy
        
        # 更新成功反向次数
        if overall_accuracy > 0.6:
            self.successful_reversals += 1
        
        # 记录结果
        outcome_record = {
            'timestamp': datetime.now(),
            'prediction': prediction_result,
            'actual_outcome': actual_outcome,
            'recommendation_accuracy': recommendation_accuracy,
            'avoidance_accuracy': avoidance_accuracy,
            'overall_accuracy': overall_accuracy,
            'strategy_type': strategy_type
        }
        
        self.reversal_outcomes.append(outcome_record)
        
        # 自适应调整
        self._adaptive_adjustment(overall_accuracy)
        
        return {
            'learning_success': True,
            'overall_accuracy': overall_accuracy,
            'recommendation_accuracy': recommendation_accuracy,
            'avoidance_accuracy': avoidance_accuracy,
            'updated_strategy_performance': self.strategy_performance.get(strategy_type, 0),
            'model_confidence': self.model_confidence,
            'successful_reversal_rate': self.successful_reversals / self.total_predictions if self.total_predictions > 0 else 0
        }
    
    def _adaptive_adjustment(self, accuracy: float):
        """自适应调整模型参数"""
        
        # 调整适应率
        if accuracy > 0.8:
            # 预测很准确，降低适应率以保持稳定
            self.adaptation_rate = max(0.05, self.adaptation_rate * 0.9)
        elif accuracy < 0.4:
            # 预测不准确，提高适应率以快速调整
            self.adaptation_rate = min(0.3, self.adaptation_rate * 1.1)
        
        # 调整配置参数
        if accuracy > 0.7:
            # 提高陷阱检测敏感性
            self.config['trap_detection_sensitivity'] = min(0.9, 
                self.config['trap_detection_sensitivity'] + 0.02)
        elif accuracy < 0.5:
            # 降低陷阱检测敏感性
            self.config['trap_detection_sensitivity'] = max(0.5, 
                self.config['trap_detection_sensitivity'] - 0.02)
    
    def _initialize_strategy_library(self) -> Dict[str, Dict]:
        """初始化策略库"""
        
        return {
            'avoid_hot_aggressive': {
                'type': 'avoid_hot',
                'reversal_strength': 0.9,
                'description': '激进避开热门策略',
                'suitable_conditions': ['strong_consensus', 'high_volatility']
            },
            'avoid_hot_moderate': {
                'type': 'avoid_hot',
                'reversal_strength': 0.7,
                'description': '温和避开热门策略',
                'suitable_conditions': ['moderate_consensus', 'normal_volatility']
            },
            'chase_cold_aggressive': {
                'type': 'chase_cold',
                'reversal_strength': 0.8,
                'description': '激进追逐冷门策略',
                'suitable_conditions': ['weak_consensus', 'low_volatility']
            },
            'chase_cold_moderate': {
                'type': 'chase_cold',
                'reversal_strength': 0.6,
                'description': '温和追逐冷门策略',
                'suitable_conditions': ['normal_consensus', 'low_volatility']
            },
            'break_pattern_strong': {
                'type': 'break_pattern',
                'reversal_strength': 0.8,
                'description': '强力打破模式策略',
                'suitable_conditions': ['clear_patterns', 'medium_volatility']
            },
            'break_pattern_subtle': {
                'type': 'break_pattern',
                'reversal_strength': 0.5,
                'description': '微妙打破模式策略',
                'suitable_conditions': ['subtle_patterns', 'low_volatility']
            },
            'defensive_conservative': {
                'type': 'defensive',
                'reversal_strength': 0.3,
                'description': '保守防御策略',
                'suitable_conditions': ['uncertain_market', 'high_risk']
            }
        }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        
        success_rate = self.successful_reversals / self.total_predictions if self.total_predictions > 0 else 0
        
        # 策略性能统计
        strategy_stats = {}
        for strategy_type, performance in self.strategy_performance.items():
            strategy_stats[strategy_type] = {
                'performance': performance,
                'usage_count': len([r for r in self.reversal_outcomes 
                                  if r.get('strategy_type') == strategy_type])
            }
        
        return {
            'model_name': 'ReversePsychologyPredictor',
            'total_predictions': self.total_predictions,
            'successful_reversals': self.successful_reversals,
            'success_rate': success_rate,
            'model_confidence': self.model_confidence,
            'adaptation_rate': self.adaptation_rate,
            'strategy_performance': strategy_stats,
            'analysis_windows': {k: len(v) for k, v in self.analysis_windows.items()},
            'popularity_history_length': len(self.popularity_history),
            'outcome_history_length': len(self.reversal_outcomes)
        }

# 辅助分析组件

class PopularityAnalyzer:
    """大众偏好分析器"""
    
    def analyze_crowd_preferences(self, historical_context: List[Dict[str, Any]], 
                                analysis_windows: Dict[str, deque]) -> Dict[str, Any]:
        """分析群体偏好模式"""
        
        # 多时间尺度热门度分析
        popularity_scores = self._calculate_multi_scale_popularity(historical_context, analysis_windows)
        
        # 识别热门尾数
        popular_tails = self._identify_popular_tails(popularity_scores)
        
        # 计算热门度信号强度
        signal_strength = self._calculate_signal_strength(popularity_scores)
        
        # 分析热门度趋势
        popularity_trend = self._analyze_popularity_trend(historical_context)
        
        return {
            'popularity_scores': popularity_scores,
            'popular_tails': popular_tails,
            'signal_strength': signal_strength,
            'popularity_trend': popularity_trend,
            'tail_risk_scores': self._calculate_tail_risk_scores(popularity_scores),
            'high_risk_tails': [tail for tail, score in popularity_scores.items() if score > 0.8]
        }
    
    def _calculate_multi_scale_popularity(self, historical_context: List[Dict[str, Any]], 
                                        analysis_windows: Dict[str, deque]) -> Dict[int, float]:
        """计算多时间尺度热门度"""
        
        popularity_scores = defaultdict(float)
        
        # 不同时间窗口的权重
        window_weights = {
            'immediate': 0.4,    # 最新3期权重最高
            'short_term': 0.3,   # 短期10期
            'medium_term': 0.2,  # 中期30期
            'long_term': 0.1     # 长期100期
        }
        
        for window_name, weight in window_weights.items():
            if window_name in analysis_windows and analysis_windows[window_name]:
                window_data = list(analysis_windows[window_name])
                window_scores = self._calculate_window_popularity(window_data)
                
                for tail, score in window_scores.items():
                    popularity_scores[tail] += score * weight
        
        return dict(popularity_scores)
    
    def _calculate_window_popularity(self, window_data: List[Dict[str, Any]]) -> Dict[int, float]:
        """计算窗口内热门度"""
        
        tail_counts = defaultdict(int)
        total_periods = len(window_data)
        
        if total_periods == 0:
            return {}
        
        for period in window_data:
            for tail in period.get('tails', []):
                tail_counts[tail] += 1
        
        # 计算相对频率作为热门度分数
        popularity_scores = {}
        for tail in range(10):
            frequency = tail_counts[tail] / total_periods
            popularity_scores[tail] = frequency
        
        return popularity_scores
    
    def _identify_popular_tails(self, popularity_scores: Dict[int, float]) -> List[int]:
        """识别热门尾数"""
        
        if not popularity_scores:
            return []
        
        # 计算动态阈值
        mean_score = np.mean(list(popularity_scores.values()))
        std_score = np.std(list(popularity_scores.values()))
        
        # 热门阈值：均值 + 0.5 * 标准差
        hot_threshold = mean_score + 0.5 * std_score
        
        popular_tails = [tail for tail, score in popularity_scores.items() if score > hot_threshold]
        
        return sorted(popular_tails)
    
    def _calculate_signal_strength(self, popularity_scores: Dict[int, float]) -> float:
        """计算热门度信号强度"""
        
        if not popularity_scores:
            return 0.0
        
        scores = list(popularity_scores.values())
        
        # 信号强度基于分数的方差
        variance = np.var(scores)
        max_variance = 0.25  # 理论最大方差（所有尾数要么0要么1）
        
        signal_strength = min(1.0, variance / max_variance)
        
        return signal_strength
    
    def _analyze_popularity_trend(self, historical_context: List[Dict[str, Any]]) -> str:
        """分析热门度趋势"""
        
        if len(historical_context) < 6:
            return 'insufficient_data'
        
        # 比较最近3期和之前3期的热门度分布
        recent_3 = historical_context[:3]
        previous_3 = historical_context[3:6]
        
        recent_scores = self._calculate_window_popularity(recent_3)
        previous_scores = self._calculate_window_popularity(previous_3)
        
        # 计算热门度变化
        trend_changes = []
        for tail in range(10):
            recent_score = recent_scores.get(tail, 0)
            previous_score = previous_scores.get(tail, 0)
            change = recent_score - previous_score
            trend_changes.append(change)
        
        avg_change = np.mean(trend_changes)
        
        if avg_change > 0.1:
            return 'increasing'
        elif avg_change < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_tail_risk_scores(self, popularity_scores: Dict[int, float]) -> Dict[int, float]:
        """计算尾数风险分数"""
        
        risk_scores = {}
        
        for tail, popularity in popularity_scores.items():
            # 风险分数基于热门度
            if popularity > 0.8:
                risk_score = 0.9  # 极高风险
            elif popularity > 0.6:
                risk_score = 0.7  # 高风险
            elif popularity > 0.4:
                risk_score = 0.5  # 中等风险
            elif popularity > 0.2:
                risk_score = 0.3  # 低风险
            else:
                risk_score = 0.1  # 极低风险
            
            risk_scores[tail] = risk_score
        
        return risk_scores

class HotNumberTrapDetector:
    """热门数字陷阱检测器"""
    
    def detect_popularity_traps(self, period_data: Dict[str, Any], 
                              historical_context: List[Dict[str, Any]], 
                              popularity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """检测热门数字陷阱"""
        
        # 识别陷阱尾数
        trap_tails = self._identify_trap_tails(historical_context, popularity_analysis)
        
        # 评估陷阱强度
        trap_strength = self._evaluate_trap_strength(trap_tails, popularity_analysis)
        
        # 分析陷阱类型
        trap_types = self._classify_trap_types(trap_tails, historical_context)
        
        return {
            'identified_traps': trap_tails,
            'trap_strength': trap_strength,
            'trap_types': trap_types,
            'trap_confidence': self._calculate_trap_confidence(trap_tails, popularity_analysis)
        }
    
    def _identify_trap_tails(self, historical_context: List[Dict[str, Any]], 
                           popularity_analysis: Dict[str, Any]) -> List[int]:
        """识别陷阱尾数"""
        
        popular_tails = popularity_analysis.get('popular_tails', [])
        popularity_scores = popularity_analysis.get('popularity_scores', {})
        
        trap_tails = []
        
        for tail in popular_tails:
            # 检查是否符合陷阱特征
            if self._is_trap_tail(tail, historical_context, popularity_scores):
                trap_tails.append(tail)
        
        return trap_tails
    
    def _is_trap_tail(self, tail: int, historical_context: List[Dict[str, Any]], 
                     popularity_scores: Dict[int, float]) -> bool:
        """判断是否为陷阱尾数"""
        
        # 陷阱特征1：短期内频繁出现
        if len(historical_context) >= 5:
            recent_5 = historical_context[:5]
            recent_count = sum(1 for period in recent_5 if tail in period.get('tails', []))
            if recent_count >= 4:  # 5期内出现4次或以上
                return True
        
        # 陷阱特征2：热门度异常高
        tail_popularity = popularity_scores.get(tail, 0)
        if tail_popularity > 0.8:
            return True
        
        # 陷阱特征3：与其他热门尾数聚集
        other_popular = [t for t, score in popularity_scores.items() 
                        if t != tail and score > 0.6]
        if len(other_popular) >= 2:
            return True
        
        return False
    
    def _evaluate_trap_strength(self, trap_tails: List[int], 
                              popularity_analysis: Dict[str, Any]) -> float:
        """评估陷阱强度"""
        
        if not trap_tails:
            return 0.0
        
        popularity_scores = popularity_analysis.get('popularity_scores', {})
        
        # 基于陷阱尾数的平均热门度
        trap_popularity = [popularity_scores.get(tail, 0) for tail in trap_tails]
        avg_trap_popularity = np.mean(trap_popularity) if trap_popularity else 0
        
        # 基于陷阱数量
        trap_count_factor = min(1.0, len(trap_tails) / 5.0)
        
        # 综合强度
        trap_strength = (avg_trap_popularity * 0.7 + trap_count_factor * 0.3)
        
        return trap_strength
    
    def _classify_trap_types(self, trap_tails: List[int], 
                           historical_context: List[Dict[str, Any]]) -> List[str]:
        """分类陷阱类型"""
        
        trap_types = []
        
        if not trap_tails:
            return trap_types
        
        # 频率陷阱：短期高频出现
        if self._detect_frequency_trap(trap_tails, historical_context):
            trap_types.append('frequency_trap')
        
        # 连续陷阱：连续多期出现
        if self._detect_consecutive_trap(trap_tails, historical_context):
            trap_types.append('consecutive_trap')
        
        # 聚集陷阱：多个热门数字同时出现
        if len(trap_tails) >= 3:
            trap_types.append('clustering_trap')
        
        # 对称陷阱：镜像对称数字
        if self._detect_symmetry_trap(trap_tails):
            trap_types.append('symmetry_trap')
        
        return trap_types
    
    def _detect_frequency_trap(self, trap_tails: List[int], 
                             historical_context: List[Dict[str, Any]]) -> bool:
        """检测频率陷阱"""
        
        if len(historical_context) < 8:
            return False
        
        recent_8 = historical_context[:8]
        
        for tail in trap_tails:
            count = sum(1 for period in recent_8 if tail in period.get('tails', []))
            if count >= 6:  # 8期内出现6次以上
                return True
        
        return False
    
    def _detect_consecutive_trap(self, trap_tails: List[int], 
                               historical_context: List[Dict[str, Any]]) -> bool:
        """检测连续陷阱"""
        
        if len(historical_context) < 4:
            return False
        
        for tail in trap_tails:
            consecutive_count = 0
            for period in historical_context:
                if tail in period.get('tails', []):
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= 3:  # 连续3期以上
                return True
        
        return False
    
    def _detect_symmetry_trap(self, trap_tails: List[int]) -> bool:
        """检测对称陷阱"""
        
        symmetry_pairs = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]
        
        for pair in symmetry_pairs:
            if pair[0] in trap_tails and pair[1] in trap_tails:
                return True
        
        return False
    
    def _calculate_trap_confidence(self, trap_tails: List[int], 
                                 popularity_analysis: Dict[str, Any]) -> float:
        """计算陷阱置信度"""
        
        if not trap_tails:
            return 0.0
        
        # 基于信号强度和陷阱数量
        signal_strength = popularity_analysis.get('signal_strength', 0.5)
        trap_count_factor = min(1.0, len(trap_tails) / 3.0)
        
        confidence = (signal_strength * 0.6 + trap_count_factor * 0.4)
        
        return confidence

class ReversalStrategist:
    """反向策略师"""
    
    def generate_reversal_strategies(self, popularity_analysis: Dict[str, Any], 
                                   trap_analysis: Dict[str, Any], 
                                   psychology_analysis: Dict[str, Any], 
                                   historical_context: List[Dict[str, Any]]) -> List[ReversalStrategy]:
        """生成反向策略"""
        
        strategies = []
        
        # 策略1：避开热门策略
        avoid_hot_strategy = self._create_avoid_hot_strategy(
            popularity_analysis, trap_analysis
        )
        if avoid_hot_strategy:
            strategies.append(avoid_hot_strategy)
        
        # 策略2：追逐冷门策略
        chase_cold_strategy = self._create_chase_cold_strategy(
            popularity_analysis, historical_context
        )
        if chase_cold_strategy:
            strategies.append(chase_cold_strategy)
        
        # 策略3：打破模式策略
        break_pattern_strategy = self._create_break_pattern_strategy(
            historical_context, psychology_analysis
        )
        if break_pattern_strategy:
            strategies.append(break_pattern_strategy)
        
        return strategies
    
    def _create_avoid_hot_strategy(self, popularity_analysis: Dict[str, Any], 
                                 trap_analysis: Dict[str, Any]) -> Optional[ReversalStrategy]:
        """创建避开热门策略"""
        
        popular_tails = popularity_analysis.get('popular_tails', [])
        trap_tails = trap_analysis.get('identified_traps', [])
        
        if not popular_tails and not trap_tails:
            return None
        
        avoidance_tails = list(set(popular_tails + trap_tails))
        target_tails = [tail for tail in range(10) if tail not in avoidance_tails]
        
        # 计算反向强度
        signal_strength = popularity_analysis.get('signal_strength', 0.5)
        trap_strength = trap_analysis.get('trap_strength', 0.0)
        reversal_strength = min(0.9, (signal_strength + trap_strength) / 2.0)
        
        # 预期有效性
        expected_effectiveness = 0.6 + reversal_strength * 0.3
        
        return ReversalStrategy(
            strategy_type='avoid_hot',
            target_tails=target_tails[:3],  # 最多推荐3个
            avoidance_tails=avoidance_tails,
            reversal_strength=reversal_strength,
            expected_effectiveness=expected_effectiveness,
            reasoning=f'检测到{len(popular_tails)}个热门尾数和{len(trap_tails)}个陷阱，采用避开策略'
        )
    
    def _create_chase_cold_strategy(self, popularity_analysis: Dict[str, Any], 
                                  historical_context: List[Dict[str, Any]]) -> Optional[ReversalStrategy]:
        """创建追逐冷门策略"""
        
        if len(historical_context) < 10:
            return None
        
        # 识别冷门尾数
        cold_tails = self._identify_cold_tails(historical_context)
        
        if not cold_tails:
            return None
        
        popularity_scores = popularity_analysis.get('popularity_scores', {})
        
        # 选择最冷门的尾数作为目标
        target_tails = sorted(cold_tails, key=lambda t: popularity_scores.get(t, 0))[:2]
        
        # 计算反向强度（基于冷门程度）
        avg_cold_score = np.mean([popularity_scores.get(t, 0) for t in cold_tails])
        reversal_strength = max(0.4, 1.0 - avg_cold_score * 2)
        
        # 预期有效性
        expected_effectiveness = 0.5 + (1.0 - avg_cold_score) * 0.3
        
        return ReversalStrategy(
            strategy_type='chase_cold',
            target_tails=target_tails,
            avoidance_tails=[],
            reversal_strength=reversal_strength,
            expected_effectiveness=expected_effectiveness,
            reasoning=f'识别到{len(cold_tails)}个冷门尾数，采用追逐冷门策略'
        )
    
    def _identify_cold_tails(self, historical_context: List[Dict[str, Any]]) -> List[int]:
        """识别冷门尾数"""
        
        # 分析最近15期
        analysis_window = min(15, len(historical_context))
        recent_data = historical_context[:analysis_window]
        
        tail_counts = defaultdict(int)
        for period in recent_data:
            for tail in period.get('tails', []):
                tail_counts[tail] += 1
        
        # 冷门阈值：出现次数少于期数的30%
        cold_threshold = analysis_window * 0.3
        
        cold_tails = [tail for tail in range(10) 
                     if tail_counts[tail] <= cold_threshold]
        
        return cold_tails
    
    def _create_break_pattern_strategy(self, historical_context: List[Dict[str, Any]], 
                                     psychology_analysis: Dict[str, Any]) -> Optional[ReversalStrategy]:
        """创建打破模式策略"""
        
        if len(historical_context) < 8:
            return None
        
        # 检测明显模式
        patterns = self._detect_obvious_patterns(historical_context)
        
        if not patterns:
            return None
        
        # 生成打破模式的目标
        pattern_breakers = self._generate_pattern_breakers(patterns, historical_context)
        
        if not pattern_breakers:
            return None
        
        # 计算反向强度
        pattern_strength = len(patterns) / 5.0  # 假设最多5种模式
        reversal_strength = min(0.8, pattern_strength)
        
        # 预期有效性
        expected_effectiveness = 0.55 + pattern_strength * 0.25
        
        return ReversalStrategy(
            strategy_type='break_pattern',
            target_tails=pattern_breakers,
            avoidance_tails=[],
            reversal_strength=reversal_strength,
            expected_effectiveness=expected_effectiveness,
            reasoning=f'检测到{len(patterns)}种明显模式，采用打破模式策略'
        )
    
    def _detect_obvious_patterns(self, historical_context: List[Dict[str, Any]]) -> List[str]:
        """检测明显模式"""
        
        patterns = []
        
        # 检测连续模式
        if self._detect_consecutive_patterns(historical_context):
            patterns.append('consecutive')
        
        # 检测交替模式
        if self._detect_alternating_patterns(historical_context):
            patterns.append('alternating')
        
        # 检测周期模式
        if self._detect_cyclic_patterns(historical_context):
            patterns.append('cyclic')
        
        return patterns
    
    def _detect_consecutive_patterns(self, historical_context: List[Dict[str, Any]]) -> bool:
        """检测连续模式"""
        
        for tail in range(10):
            consecutive_count = 0
            for period in historical_context[:6]:  # 检查最近6期
                if tail in period.get('tails', []):
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= 3:
                return True
        
        return False
    
    def _detect_alternating_patterns(self, historical_context: List[Dict[str, Any]]) -> bool:
        """检测交替模式"""
        
        if len(historical_context) < 6:
            return False
        
        for tail in range(10):
            # 检查1010或0101模式
            pattern = []
            for period in historical_context[:6]:
                pattern.append(1 if tail in period.get('tails', []) else 0)
            
            # 检查交替模式
            alternating = True
            for i in range(len(pattern) - 1):
                if pattern[i] == pattern[i + 1]:
                    alternating = False
                    break
            
            if alternating and len(pattern) >= 4:
                return True
        
        return False
    
    def _detect_cyclic_patterns(self, historical_context: List[Dict[str, Any]]) -> bool:
        """检测周期模式"""
        
        if len(historical_context) < 9:
            return False
        
        # 检测3期周期
        for tail in range(10):
            cycle_3_pattern = []
            for i in range(0, min(9, len(historical_context)), 3):
                if i < len(historical_context):
                    cycle_3_pattern.append(1 if tail in historical_context[i].get('tails', []) else 0)
            
            if len(cycle_3_pattern) >= 3 and len(set(cycle_3_pattern)) == 1:
                return True
        
        return False
    
    def _generate_pattern_breakers(self, patterns: List[str], 
                                 historical_context: List[Dict[str, Any]]) -> List[int]:
        """生成打破模式的目标尾数"""
        
        pattern_breakers = set()
        
        for pattern_type in patterns:
            if pattern_type == 'consecutive':
                # 打破连续模式：选择最近没有连续出现的尾数
                breakers = self._break_consecutive_pattern(historical_context)
                pattern_breakers.update(breakers)
            
            elif pattern_type == 'alternating':
                # 打破交替模式：选择打破交替规律的尾数
                breakers = self._break_alternating_pattern(historical_context)
                pattern_breakers.update(breakers)
            
            elif pattern_type == 'cyclic':
                # 打破周期模式：选择不符合周期的尾数
                breakers = self._break_cyclic_pattern(historical_context)
                pattern_breakers.update(breakers)
        
        return list(pattern_breakers)[:3]  # 最多返回3个
    
    def _break_consecutive_pattern(self, historical_context: List[Dict[str, Any]]) -> List[int]:
        """打破连续模式"""
        
        breakers = []
        
        for tail in range(10):
            # 检查该尾数是否在连续出现
            consecutive_count = 0
            for period in historical_context[:4]:
                if tail in period.get('tails', []):
                    consecutive_count += 1
                else:
                    break
            
            # 如果没有连续出现，可以作为打破者
            if consecutive_count == 0:
                breakers.append(tail)
        
        return breakers
    
    def _break_alternating_pattern(self, historical_context: List[Dict[str, Any]]) -> List[int]:
        """打破交替模式"""
        
        breakers = []
        
        if len(historical_context) < 2:
            return breakers
        
        last_period_tails = set(historical_context[0].get('tails', []))
        second_last_tails = set(historical_context[1].get('tails', []))
        
        # 选择在最近两期中都出现或都没出现的尾数（打破交替）
        for tail in range(10):
            last_appeared = tail in last_period_tails
            second_last_appeared = tail in second_last_tails
            
            if last_appeared == second_last_appeared:
                breakers.append(tail)
        
        return breakers
    
    def _break_cyclic_pattern(self, historical_context: List[Dict[str, Any]]) -> List[int]:
        """打破周期模式"""
        
        breakers = []
        
        # 简化实现：选择最近3期中出现频率适中的尾数
        if len(historical_context) >= 3:
            recent_3 = historical_context[:3]
            tail_counts = defaultdict(int)
            
            for period in recent_3:
                for tail in period.get('tails', []):
                    tail_counts[tail] += 1
            
            # 选择出现1-2次的尾数作为打破者
            for tail, count in tail_counts.items():
                if 1 <= count <= 2:
                    breakers.append(tail)
        
        return breakers

class CrowdPsychologyEngine:
    """群体心理分析引擎"""
    
    def analyze_group_psychology(self, historical_context: List[Dict[str, Any]], 
                                popularity_analysis: Dict[str, Any], 
                                trap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析群体心理状态"""
        
        # 分析从众行为强度
        herd_behavior_strength = self._analyze_herd_behavior(historical_context)
        
        # 分析群体情绪
        crowd_sentiment = self._analyze_crowd_sentiment(popularity_analysis, trap_analysis)
        
        # 分析反向准备度
        reversal_readiness = self._analyze_reversal_readiness(historical_context, popularity_analysis)
        
        # 分析市场条件
        market_conditions = self._analyze_market_conditions(historical_context)
        
        return {
            'herd_behavior_strength': herd_behavior_strength,
            'crowd_sentiment': crowd_sentiment,
            'reversal_readiness': reversal_readiness,
            'market_conditions': market_conditions,
            'crowd_influence_strength': self._calculate_crowd_influence(herd_behavior_strength, crowd_sentiment),
            'crowd_consensus_strength': self._calculate_consensus_strength(popularity_analysis),
            'market_volatility': market_conditions.get('volatility', 0.5)
        }
    
    def _analyze_herd_behavior(self, historical_context: List[Dict[str, Any]]) -> float:
        """分析从众行为强度"""
        
        if len(historical_context) < 5:
            return 0.5
        
        # 计算期间的相似度
        similarities = []
        recent_5 = historical_context[:5]
        
        for i in range(len(recent_5) - 1):
            current_tails = set(recent_5[i].get('tails', []))
            next_tails = set(recent_5[i + 1].get('tails', []))
            
            if current_tails or next_tails:
                similarity = len(current_tails.intersection(next_tails)) / len(current_tails.union(next_tails))
                similarities.append(similarity)
        
        # 从众强度基于相似度
        avg_similarity = np.mean(similarities) if similarities else 0.5
        herd_strength = avg_similarity
        
        return herd_strength
    
    def _analyze_crowd_sentiment(self, popularity_analysis: Dict[str, Any], 
                               trap_analysis: Dict[str, Any]) -> str:
        """分析群体情绪"""
        
        signal_strength = popularity_analysis.get('signal_strength', 0.5)
        trap_strength = trap_analysis.get('trap_strength', 0.0)
        
        if signal_strength > 0.7 and trap_strength > 0.5:
            return 'bullish'  # 乐观，追逐热门
        elif signal_strength < 0.3 and trap_strength < 0.3:
            return 'bearish'  # 悲观，避险
        else:
            return 'neutral'  # 中性
    
    def _analyze_reversal_readiness(self, historical_context: List[Dict[str, Any]], 
                                  popularity_analysis: Dict[str, Any]) -> float:
        """分析反向准备度"""
        
        # 基于热门度持续时间
        popularity_trend = popularity_analysis.get('popularity_trend', 'stable')
        
        readiness = 0.5
        
        if popularity_trend == 'increasing':
            # 热门度持续上升，反向准备度提高
            readiness += 0.3
        elif popularity_trend == 'decreasing':
            # 热门度下降，反向准备度降低
            readiness -= 0.2
        
        # 基于极端程度
        signal_strength = popularity_analysis.get('signal_strength', 0.5)
        if signal_strength > 0.8:
            readiness += 0.2  # 极端情况更容易反转
        
        return min(1.0, max(0.0, readiness))
    
    def _analyze_market_conditions(self, historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析市场条件"""
        
        if len(historical_context) < 8:
            return {'volatility': 0.5, 'consensus_strength': 0.5}
        
        # 计算波动性
        volatility = self._calculate_market_volatility(historical_context[:8])
        
        # 计算共识强度
        consensus_strength = self._calculate_consensus_strength_from_history(historical_context[:8])
        
        return {
            'volatility': volatility,
            'consensus_strength': consensus_strength
        }
    
    def _calculate_market_volatility(self, periods: List[Dict[str, Any]]) -> float:
        """计算市场波动性"""
        
        # 基于期间尾数数量的变化
        tail_counts = [len(period.get('tails', [])) for period in periods]
        
        if len(tail_counts) > 1:
            volatility = np.std(tail_counts) / np.mean(tail_counts) if np.mean(tail_counts) > 0 else 0
        else:
            volatility = 0.5
        
        return min(1.0, volatility)
    
    def _calculate_consensus_strength_from_history(self, periods: List[Dict[str, Any]]) -> float:
        """从历史数据计算共识强度"""
        
        if len(periods) < 3:
            return 0.5
        
        # 计算尾数出现的一致性
        tail_frequencies = defaultdict(int)
        total_periods = len(periods)
        
        for period in periods:
            for tail in period.get('tails', []):
                tail_frequencies[tail] += 1
        
        # 计算频率分布的均匀性
        frequencies = list(tail_frequencies.values())
        if frequencies:
            cv = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 0
            consensus = min(1.0, cv)  # 变异系数越高，共识越强
        else:
            consensus = 0.5
        
        return consensus
    
    def _calculate_crowd_influence(self, herd_strength: float, sentiment: str) -> float:
        """计算群体影响力强度"""
        
        sentiment_weights = {
            'bullish': 1.2,
            'bearish': 0.8,
            'neutral': 1.0
        }
        
        sentiment_weight = sentiment_weights.get(sentiment, 1.0)
        influence = herd_strength * sentiment_weight
        
        return min(1.0, influence)
    
    def _calculate_consensus_strength(self, popularity_analysis: Dict[str, Any]) -> float:
        """计算群体共识强度"""
        
        signal_strength = popularity_analysis.get('signal_strength', 0.5)
        popular_tails_count = len(popularity_analysis.get('popular_tails', []))
        
        # 共识强度基于信号强度和热门尾数集中度
        if popular_tails_count <= 2:
            concentration_factor = 1.0  # 高集中度
        elif popular_tails_count <= 4:
            concentration_factor = 0.7  # 中等集中度
        else:
            concentration_factor = 0.4  # 低集中度
        
        consensus = signal_strength * concentration_factor
        
        return consensus