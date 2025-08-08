#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import json

class MoneyFlowAnalyzer:
    """
    资金流向分析器 - 科研级定制版本
    
    核心理念：基于历史开奖模式推断虚拟资金流向，识别热门度和赔付压力，
    从而预测庄家可能的"杀多赔少"策略目标。
    
    主要功能：
    1. 虚拟资金流向分析 - 基于历史模式推断投注热度
    2. 热门度量化算法 - 多维度评估尾数受欢迎程度  
    3. 赔付压力评估机制 - 计算庄家面临的理论赔付风险
    4. 风险点识别系统 - 识别庄家最可能"杀掉"的热门尾数
    """
    
    def __init__(self):
        """初始化资金流向分析器"""
        print("💰 初始化资金流向分析器...")
        
        # === 核心分析参数 ===
        self.analysis_window = 30           # 分析窗口期数
        self.hot_threshold = 0.65          # 热门阈值
        self.pressure_threshold = 0.70     # 赔付压力阈值
        self.risk_threshold = 0.75         # 风险识别阈值
        
        # === 历史数据缓存 ===
        self.historical_patterns = {}      # 历史模式缓存
        self.flow_cache = deque(maxlen=100) # 资金流向缓存
        self.heat_trends = defaultdict(list) # 热度趋势记录
        
        # === 分析权重配置 ===
        self.weight_config = {
            'frequency_weight': 0.25,       # 频率权重
            'continuity_weight': 0.20,      # 连续性权重  
            'interval_weight': 0.15,        # 间隔权重
            'pattern_weight': 0.15,         # 模式权重
            'momentum_weight': 0.10,        # 动量权重
            'volatility_weight': 0.15       # 波动性权重
        }
        
        # === 赔付压力模型参数 ===
        self.payout_model = {
            'base_odds': 2.0,               # 基础赔率
            'popularity_factor': 1.5,       # 热门系数
            'frequency_multiplier': 1.8,    # 频率乘数
            'consecutive_bonus': 0.3,       # 连续奖励
            'pattern_bonus': 0.25          # 模式奖励
        }
        
        # === 风险识别算法参数 ===
        self.risk_model = {
            'kill_probability_base': 0.4,   # 基础"杀"概率
            'hot_penalty': 0.6,             # 热门惩罚
            'pressure_amplifier': 2.0,      # 压力放大器
            'pattern_danger': 0.8,          # 模式危险系数
            'crowd_following_risk': 0.7     # 从众跟风风险
        }
        
        # === 统计数据 ===
        self.analysis_stats = {
            'total_analyses': 0,
            'hot_spots_identified': 0,
            'pressure_points_found': 0,
            'successful_predictions': 0,
            'false_positives': 0,
            'accuracy_history': []
        }
        
        print("✅ 资金流向分析器初始化完成")
        print(f"   分析窗口: {self.analysis_window}期")
        print(f"   热门阈值: {self.hot_threshold}")
        print(f"   压力阈值: {self.pressure_threshold}")
        print(f"   风险阈值: {self.risk_threshold}")
    
    def analyze_money_flow(self, candidate_tails: List[int], historical_data: List[Dict]) -> Dict:
        """
        分析资金流向并预测庄家策略
        
        Args:
            candidate_tails: 经过三大定律筛选的候选尾数
            historical_data: 历史开奖数据（最新在前）
            
        Returns:
            分析结果字典
        """
        if not candidate_tails or not historical_data:
            return self._create_empty_result("输入数据不足")
        
        if len(historical_data) < 10:
            return self._create_empty_result("历史数据不足10期，无法进行有效分析")
        
        print(f"💰 开始资金流向分析...")
        print(f"   候选尾数: {candidate_tails}")
        print(f"   历史数据: {len(historical_data)}期")
        
        try:
            # === 第一阶段：虚拟资金流向推断 ===
            flow_analysis = self._analyze_virtual_money_flow(candidate_tails, historical_data)
            
            # === 第二阶段：热门度量化计算 ===
            popularity_analysis = self._calculate_popularity_metrics(candidate_tails, historical_data)
            
            # === 第三阶段：赔付压力评估 ===
            pressure_analysis = self._assess_payout_pressure(candidate_tails, historical_data, popularity_analysis)
            
            # === 第四阶段：风险点识别 ===
            risk_analysis = self._identify_risk_points(candidate_tails, pressure_analysis, flow_analysis)
            
            # === 第五阶段：策略建议生成 ===
            strategy_recommendations = self._generate_strategy_recommendations(
                candidate_tails, flow_analysis, popularity_analysis, pressure_analysis, risk_analysis
            )
            
            # === 综合分析结果 ===
            analysis_result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'candidate_tails': candidate_tails,
                'analysis_window': min(len(historical_data), self.analysis_window),
                
                # 核心分析结果
                'flow_analysis': flow_analysis,
                'popularity_analysis': popularity_analysis,
                'pressure_analysis': pressure_analysis,
                'risk_analysis': risk_analysis,
                'strategy_recommendations': strategy_recommendations,
                
                # 决策支持信息
                'recommended_tails': strategy_recommendations['recommended_tails'],
                'avoid_tails': strategy_recommendations['avoid_tails'],
                'confidence': strategy_recommendations['overall_confidence'],
                'reasoning': strategy_recommendations['reasoning'],
                
                # 详细数据（供调试和验证使用）
                'detailed_metrics': self._compile_detailed_metrics(
                    candidate_tails, flow_analysis, popularity_analysis, pressure_analysis, risk_analysis
                ),
                
                # 统计信息更新
                'analysis_stats': self._update_analysis_stats()
            }
            
            # 缓存分析结果
            self._cache_analysis_result(analysis_result)
            
            print(f"✅ 资金流向分析完成")
            print(f"   推荐尾数: {strategy_recommendations['recommended_tails']}")
            print(f"   避开尾数: {strategy_recommendations['avoid_tails']}")
            print(f"   整体置信度: {strategy_recommendations['overall_confidence']:.3f}")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 资金流向分析失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result(f"分析过程出错: {str(e)}")
    
    def _analyze_virtual_money_flow(self, candidate_tails: List[int], historical_data: List[Dict]) -> Dict:
        """
        分析虚拟资金流向
        
        基于历史开奖模式，推断各尾数的虚拟投注热度和资金流向趋势
        """
        print("   📊 执行虚拟资金流向分析...")
        
        flow_data = {}
        analysis_window = min(len(historical_data), self.analysis_window)
        recent_data = historical_data[:analysis_window]
        
        for tail in candidate_tails:
            # === 基础流向指标 ===
            appearances = [i for i, period in enumerate(recent_data) if tail in period.get('tails', [])]
            frequency = len(appearances) / analysis_window
            
            # === 流向强度计算 ===
            flow_strength = self._calculate_flow_strength(tail, recent_data, appearances)
            
            # === 流向趋势分析 ===
            flow_trend = self._analyze_flow_trend(tail, recent_data, appearances)
            
            # === 流向稳定性评估 ===
            flow_stability = self._assess_flow_stability(tail, recent_data, appearances)
            
            # === 流向动量计算 ===
            flow_momentum = self._calculate_flow_momentum(tail, recent_data, appearances)
            
            # === 综合流向评分 ===
            comprehensive_flow_score = (
                flow_strength * 0.3 +
                flow_trend * 0.25 +
                flow_stability * 0.2 +
                flow_momentum * 0.25
            )
            
            flow_data[tail] = {
                'frequency': frequency,
                'appearances': appearances,
                'flow_strength': flow_strength,
                'flow_trend': flow_trend,
                'flow_stability': flow_stability,
                'flow_momentum': flow_momentum,
                'comprehensive_score': comprehensive_flow_score,
                'flow_level': self._classify_flow_level(comprehensive_flow_score),
                'trend_direction': 'up' if flow_trend > 0.5 else 'down' if flow_trend < -0.5 else 'stable'
            }
        
        # === 相对流向分析 ===
        flow_rankings = self._calculate_relative_flow_rankings(flow_data)
        
        # === 流向集中度分析 ===
        flow_concentration = self._analyze_flow_concentration(flow_data)
        
        print(f"   ✓ 虚拟资金流向分析完成，检测到{len([t for t, d in flow_data.items() if d['flow_level'] == 'high'])}个高流向尾数")
        
        return {
            'individual_flows': flow_data,
            'flow_rankings': flow_rankings,
            'flow_concentration': flow_concentration,
            'analysis_window': analysis_window,
            'high_flow_tails': [tail for tail, data in flow_data.items() if data['flow_level'] == 'high'],
            'summary': self._summarize_flow_analysis(flow_data, flow_rankings, flow_concentration)
        }
    
    def _calculate_popularity_metrics(self, candidate_tails: List[int], historical_data: List[Dict]) -> Dict:
        """
        计算热门度量化指标
        
        多维度评估各尾数的受欢迎程度和市场热度
        """
        print("   🔥 执行热门度量化计算...")
        
        popularity_data = {}
        analysis_window = min(len(historical_data), self.analysis_window)
        recent_data = historical_data[:analysis_window]
        
        for tail in candidate_tails:
            # === 频率热度 ===
            frequency_heat = self._calculate_frequency_heat(tail, recent_data)
            
            # === 连续性热度 ===
            continuity_heat = self._calculate_continuity_heat(tail, recent_data)
            
            # === 间隔热度 ===
            interval_heat = self._calculate_interval_heat(tail, recent_data)
            
            # === 模式热度 ===
            pattern_heat = self._calculate_pattern_heat(tail, recent_data)
            
            # === 动量热度 ===
            momentum_heat = self._calculate_momentum_heat(tail, recent_data)
            
            # === 波动性热度 ===
            volatility_heat = self._calculate_volatility_heat(tail, recent_data)
            
            # === 综合热门度评分 ===
            comprehensive_popularity = (
                frequency_heat * self.weight_config['frequency_weight'] +
                continuity_heat * self.weight_config['continuity_weight'] +
                interval_heat * self.weight_config['interval_weight'] +
                pattern_heat * self.weight_config['pattern_weight'] +
                momentum_heat * self.weight_config['momentum_weight'] +
                volatility_heat * self.weight_config['volatility_weight']
            )
            
            popularity_data[tail] = {
                'frequency_heat': frequency_heat,
                'continuity_heat': continuity_heat,
                'interval_heat': interval_heat,
                'pattern_heat': pattern_heat,
                'momentum_heat': momentum_heat,
                'volatility_heat': volatility_heat,
                'comprehensive_popularity': comprehensive_popularity,
                'heat_level': self._classify_heat_level(comprehensive_popularity),
                'dominant_factor': self._identify_dominant_heat_factor(
                    frequency_heat, continuity_heat, interval_heat, 
                    pattern_heat, momentum_heat, volatility_heat
                )
            }
        
        # === 相对热度排名 ===
        popularity_rankings = self._calculate_popularity_rankings(popularity_data)
        
        # === 热度分布分析 ===
        heat_distribution = self._analyze_heat_distribution(popularity_data)
        
        print(f"   ✓ 热门度量化完成，识别到{len([t for t, d in popularity_data.items() if d['heat_level'] == 'hot'])}个热门尾数")
        
        return {
            'individual_popularity': popularity_data,
            'popularity_rankings': popularity_rankings,
            'heat_distribution': heat_distribution,
            'hot_tails': [tail for tail, data in popularity_data.items() if data['heat_level'] == 'hot'],
            'summary': self._summarize_popularity_analysis(popularity_data, popularity_rankings, heat_distribution)
        }
    
    def _assess_payout_pressure(self, candidate_tails: List[int], historical_data: List[Dict], 
                               popularity_analysis: Dict) -> Dict:
        """
        评估赔付压力
        
        基于热门度和历史模式，计算庄家面临的理论赔付压力
        """
        print("   💸 执行赔付压力评估...")
        
        pressure_data = {}
        
        for tail in candidate_tails:
            popularity_info = popularity_analysis['individual_popularity'][tail]
            
            # === 基础赔付风险 ===
            base_risk = self._calculate_base_payout_risk(tail, historical_data)
            
            # === 热门度压力 ===
            popularity_pressure = self._calculate_popularity_pressure(popularity_info)
            
            # === 频率压力 ===
            frequency_pressure = self._calculate_frequency_pressure(tail, historical_data)
            
            # === 连续性压力 ===
            continuity_pressure = self._calculate_continuity_pressure(tail, historical_data)
            
            # === 模式压力 ===
            pattern_pressure = self._calculate_pattern_pressure(tail, historical_data)
            
            # === 综合赔付压力 ===
            comprehensive_pressure = (
                base_risk * 0.2 +
                popularity_pressure * 0.25 +
                frequency_pressure * 0.2 +
                continuity_pressure * 0.15 +
                pattern_pressure * 0.2
            )
            
            # === 压力等级分类 ===
            pressure_level = self._classify_pressure_level(comprehensive_pressure)
            
            # === 杀号概率预测 ===
            kill_probability = self._predict_kill_probability(comprehensive_pressure, popularity_info)
            
            pressure_data[tail] = {
                'base_risk': base_risk,
                'popularity_pressure': popularity_pressure,
                'frequency_pressure': frequency_pressure,
                'continuity_pressure': continuity_pressure,
                'pattern_pressure': pattern_pressure,
                'comprehensive_pressure': comprehensive_pressure,
                'pressure_level': pressure_level,
                'kill_probability': kill_probability,
                'risk_factors': self._identify_pressure_risk_factors(
                    base_risk, popularity_pressure, frequency_pressure, 
                    continuity_pressure, pattern_pressure
                )
            }
        
        # === 压力排名 ===
        pressure_rankings = self._calculate_pressure_rankings(pressure_data)
        
        # === 整体压力分析 ===
        overall_pressure = self._analyze_overall_pressure(pressure_data)
        
        print(f"   ✓ 赔付压力评估完成，发现{len([t for t, d in pressure_data.items() if d['pressure_level'] == 'high'])}个高压力尾数")
        
        return {
            'individual_pressure': pressure_data,
            'pressure_rankings': pressure_rankings,
            'overall_pressure': overall_pressure,
            'high_pressure_tails': [tail for tail, data in pressure_data.items() if data['pressure_level'] == 'high'],
            'summary': self._summarize_pressure_analysis(pressure_data, pressure_rankings, overall_pressure)
        }
    
    def _identify_risk_points(self, candidate_tails: List[int], pressure_analysis: Dict, 
                             flow_analysis: Dict) -> Dict:
        """
        识别风险点
        
        综合资金流向和赔付压力，识别庄家最可能采取行动的风险点
        """
        print("   ⚠️ 执行风险点识别...")
        
        risk_data = {}
        
        for tail in candidate_tails:
            pressure_info = pressure_analysis['individual_pressure'][tail]
            flow_info = flow_analysis['individual_flows'][tail]
            
            # === 复合风险评分 ===
            compound_risk = self._calculate_compound_risk(pressure_info, flow_info)
            
            # === 时机风险评估 ===
            timing_risk = self._assess_timing_risk(tail, pressure_info, flow_info)
            
            # === 策略风险分析 ===
            strategy_risk = self._analyze_strategy_risk(tail, pressure_info, flow_info)
            
            # === 市场风险评估 ===
            market_risk = self._assess_market_risk(tail, pressure_info, flow_info)
            
            # === 综合风险评分 ===
            comprehensive_risk = (
                compound_risk * 0.3 +
                timing_risk * 0.25 +
                strategy_risk * 0.25 +
                market_risk * 0.2
            )
            
            # === 风险等级分类 ===
            risk_level = self._classify_risk_level(comprehensive_risk)
            
            # === 行动概率预测 ===
            action_probability = self._predict_action_probability(comprehensive_risk)
            
            risk_data[tail] = {
                'compound_risk': compound_risk,
                'timing_risk': timing_risk,
                'strategy_risk': strategy_risk,
                'market_risk': market_risk,
                'comprehensive_risk': comprehensive_risk,
                'risk_level': risk_level,
                'action_probability': action_probability,
                'risk_indicators': self._compile_risk_indicators(
                    compound_risk, timing_risk, strategy_risk, market_risk
                ),
                'warning_signals': self._detect_warning_signals(tail, pressure_info, flow_info)
            }
        
        # === 风险排名 ===
        risk_rankings = self._calculate_risk_rankings(risk_data)
        
        # === 系统性风险评估 ===
        systemic_risk = self._assess_systemic_risk(risk_data)
        
        print(f"   ✓ 风险点识别完成，发现{len([t for t, d in risk_data.items() if d['risk_level'] == 'high'])}个高风险尾数")
        
        return {
            'individual_risk': risk_data,
            'risk_rankings': risk_rankings,
            'systemic_risk': systemic_risk,
            'high_risk_tails': [tail for tail, data in risk_data.items() if data['risk_level'] == 'high'],
            'critical_risk_tails': [tail for tail, data in risk_data.items() if data['action_probability'] > 0.8],
            'summary': self._summarize_risk_analysis(risk_data, risk_rankings, systemic_risk)
        }
    
    def _generate_strategy_recommendations(self, candidate_tails: List[int], flow_analysis: Dict,
                                         popularity_analysis: Dict, pressure_analysis: Dict, 
                                         risk_analysis: Dict) -> Dict:
        """
        生成策略建议
        
        基于所有分析结果，生成最终的投注策略建议
        """
        print("   🎯 生成策略建议...")
        
        # === 收集所有分析数据 ===
        analysis_data = {}
        for tail in candidate_tails:
            analysis_data[tail] = {
                'flow': flow_analysis['individual_flows'][tail],
                'popularity': popularity_analysis['individual_popularity'][tail],
                'pressure': pressure_analysis['individual_pressure'][tail],
                'risk': risk_analysis['individual_risk'][tail]
            }
        
        # === 安全性评分计算 ===
        safety_scores = {}
        for tail in candidate_tails:
            data = analysis_data[tail]
            
            # 安全性 = 低风险 + 低压力 + 适中流向 + 适中热度
            safety_score = (
                (1.0 - data['risk']['comprehensive_risk']) * 0.4 +
                (1.0 - data['pressure']['comprehensive_pressure']) * 0.3 +
                min(data['flow']['comprehensive_score'], 1.0 - data['flow']['comprehensive_score']) * 0.2 +
                min(data['popularity']['comprehensive_popularity'], 1.0 - data['popularity']['comprehensive_popularity']) * 0.1
            )
            
            safety_scores[tail] = safety_score
        
        # === 机会性评分计算 ===
        opportunity_scores = {}
        for tail in candidate_tails:
            data = analysis_data[tail]
            
            # 机会性 = 低流向 + 低热度 + 低风险 + 历史表现
            opportunity_score = (
                (1.0 - data['flow']['comprehensive_score']) * 0.3 +
                (1.0 - data['popularity']['comprehensive_popularity']) * 0.3 +
                (1.0 - data['risk']['comprehensive_risk']) * 0.2 +
                self._calculate_historical_opportunity(tail) * 0.2
            )
            
            opportunity_scores[tail] = opportunity_score
        
        # === 策略分类 ===
        safe_choices = [tail for tail, score in safety_scores.items() if score > 0.6]
        opportunity_choices = [tail for tail, score in opportunity_scores.items() if score > 0.6]
        avoid_choices = [tail for tail in candidate_tails if 
                        risk_analysis['individual_risk'][tail]['risk_level'] == 'high' or
                        pressure_analysis['individual_pressure'][tail]['pressure_level'] == 'high']
        
        # === 最终推荐逻辑 ===
        if safe_choices and opportunity_choices:
            # 有既安全又有机会的选择
            recommended = list(set(safe_choices) & set(opportunity_choices))
            if not recommended:
                recommended = safe_choices[:2] + opportunity_choices[:2]
                recommended = list(set(recommended))[:3]
        elif safe_choices:
            # 只有安全选择
            recommended = safe_choices[:2]
        elif opportunity_choices:
            # 只有机会选择
            recommended = opportunity_choices[:2]
        else:
            # 都没有，选择风险最小的
            recommended = [min(candidate_tails, key=lambda t: risk_analysis['individual_risk'][t]['comprehensive_risk'])]
        
        # === 计算整体置信度 ===
        overall_confidence = self._calculate_overall_confidence(
            candidate_tails, analysis_data, recommended, avoid_choices
        )
        
        # === 生成推理说明 ===
        reasoning = self._generate_reasoning(
            candidate_tails, analysis_data, recommended, avoid_choices, 
            safe_choices, opportunity_choices, safety_scores, opportunity_scores
        )
        
        return {
            'recommended_tails': recommended,
            'avoid_tails': list(set(avoid_choices)),
            'safe_choices': safe_choices,
            'opportunity_choices': opportunity_choices,
            'safety_scores': safety_scores,
            'opportunity_scores': opportunity_scores,
            'overall_confidence': overall_confidence,
            'reasoning': reasoning,
            'strategy_type': self._determine_strategy_type(recommended, avoid_choices, safe_choices, opportunity_choices),
            'confidence_breakdown': self._breakdown_confidence(overall_confidence, analysis_data)
        }
    
    # === 辅助计算方法 ===
    
    def _calculate_flow_strength(self, tail: int, recent_data: List[Dict], appearances: List[int]) -> float:
        """计算流向强度"""
        if not appearances:
            return 0.0
        
        # 基于出现频率和时间衰减
        total_strength = 0.0
        for i, appearance_index in enumerate(appearances):
            # 时间权重：最近的权重更高
            time_weight = 1.0 - (appearance_index / len(recent_data))
            # 频率权重：连续出现加权
            freq_weight = 1.0 + (0.1 * i)
            total_strength += time_weight * freq_weight
        
        return min(total_strength / len(appearances), 1.0)
    
    def _analyze_flow_trend(self, tail: int, recent_data: List[Dict], appearances: List[int]) -> float:
        """分析流向趋势"""
        if len(appearances) < 2:
            return 0.0
        
        # 计算出现间隔的趋势
        intervals = []
        for i in range(1, len(appearances)):
            intervals.append(appearances[i] - appearances[i-1])
        
        if not intervals:
            return 0.0
        
        # 线性回归分析趋势
        n = len(intervals)
        x_values = list(range(n))
        y_values = intervals
        
        if n == 1:
            return 0.0
        
        # 简化的线性回归
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # 转换为0-1分数，负斜率表示间隔缩短（趋势向上）
        return max(0.0, min(1.0, 0.5 - slope * 0.1))
    
    def _assess_flow_stability(self, tail: int, recent_data: List[Dict], appearances: List[int]) -> float:
        """评估流向稳定性"""
        if len(appearances) < 3:
            return 0.5
        
        # 计算出现间隔的变异系数
        intervals = []
        for i in range(1, len(appearances)):
            intervals.append(appearances[i] - appearances[i-1])
        
        if not intervals:
            return 0.5
        
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.0
        
        variance = sum((interval - mean_interval) ** 2 for interval in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_interval
        
        # 变异系数越小，稳定性越高
        stability = 1.0 / (1.0 + cv)
        return min(max(stability, 0.0), 1.0)
    
    def _calculate_flow_momentum(self, tail: int, recent_data: List[Dict], appearances: List[int]) -> float:
        """计算流向动量"""
        if not appearances:
            return 0.0
        
        # 最近5期的动量
        recent_5 = recent_data[:5]
        recent_appearances = sum(1 for period in recent_5 if tail in period.get('tails', []))
        
        # 之前5期的对比
        if len(recent_data) >= 10:
            previous_5 = recent_data[5:10]
            previous_appearances = sum(1 for period in previous_5 if tail in period.get('tails', []))
        else:
            previous_appearances = len(appearances) * 0.5
        
        if previous_appearances == 0:
            momentum = 1.0 if recent_appearances > 0 else 0.0
        else:
            momentum = recent_appearances / previous_appearances
        
        return min(momentum / 2.0, 1.0)  # 归一化到0-1
    
    def _classify_flow_level(self, score: float) -> str:
        """分类流向等级"""
        if score > 0.7:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_relative_flow_rankings(self, flow_data: Dict) -> List[Tuple[int, float]]:
        """计算相对流向排名"""
        rankings = [(tail, data['comprehensive_score']) for tail, data in flow_data.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _analyze_flow_concentration(self, flow_data: Dict) -> Dict:
        """分析流向集中度"""
        scores = [data['comprehensive_score'] for data in flow_data.values()]
        if not scores:
            return {'concentration_index': 0.0, 'distribution': 'even'}
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        concentration_index = math.sqrt(variance) / (mean_score + 0.001)
        
        if concentration_index > 0.5:
            distribution = 'concentrated'
        elif concentration_index > 0.2:
            distribution = 'moderate'
        else:
            distribution = 'even'
        
        return {
            'concentration_index': concentration_index,
            'distribution': distribution,
            'mean_score': mean_score,
            'variance': variance
        }
    
    def _summarize_flow_analysis(self, flow_data: Dict, flow_rankings: List, flow_concentration: Dict) -> str:
        """总结流向分析"""
        high_flow_count = len([d for d in flow_data.values() if d['flow_level'] == 'high'])
        total_count = len(flow_data)
        concentration = flow_concentration['distribution']
        
        return f"检测到{high_flow_count}/{total_count}个高流向尾数，流向分布呈{concentration}状态"
    
    # === 热门度计算方法 ===
    
    def _calculate_frequency_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算频率热度"""
        appearances = sum(1 for period in recent_data if tail in period.get('tails', []))
        frequency = appearances / len(recent_data)
        
        # S型曲线转换，0.5附近变化最敏感
        if frequency <= 0.5:
            return 2 * frequency * frequency
        else:
            return 1 - 2 * (1 - frequency) * (1 - frequency)
    
    def _calculate_continuity_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算连续性热度"""
        max_consecutive = 0
        current_consecutive = 0
        
        for period in recent_data:
            if tail in period.get('tails', []):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # 连续3次以上开始快速升温
        if max_consecutive <= 1:
            return 0.1
        elif max_consecutive == 2:
            return 0.3
        elif max_consecutive == 3:
            return 0.6
        else:
            return min(0.9, 0.6 + (max_consecutive - 3) * 0.1)
    
    def _calculate_interval_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算间隔热度"""
        appearances = [i for i, period in enumerate(recent_data) if tail in period.get('tails', [])]
        
        if len(appearances) < 2:
            return 0.2
        
        intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
        avg_interval = sum(intervals) / len(intervals)
        
        # 间隔越短，热度越高
        heat = 1.0 / (1.0 + avg_interval * 0.2)
        return min(max(heat, 0.0), 1.0)
    
    def _calculate_pattern_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算模式热度"""
        # 检测规律性模式
        appearances = [1 if tail in period.get('tails', []) else 0 for period in recent_data]
        
        # 检测周期性
        pattern_strength = 0.0
        
        # 检测2-5期的周期模式
        for cycle in range(2, 6):
            if len(appearances) >= cycle * 2:
                matches = 0
                total_checks = 0
                
                for i in range(cycle, len(appearances)):
                    if appearances[i] == appearances[i - cycle]:
                        matches += 1
                    total_checks += 1
                
                if total_checks > 0:
                    cycle_strength = matches / total_checks
                    pattern_strength = max(pattern_strength, cycle_strength)
        
        return pattern_strength
    
    def _calculate_momentum_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算动量热度"""
        if len(recent_data) < 6:
            return 0.5
        
        # 最近3期 vs 之前3期
        recent_3 = sum(1 for period in recent_data[:3] if tail in period.get('tails', []))
        previous_3 = sum(1 for period in recent_data[3:6] if tail in period.get('tails', []))
        
        if previous_3 == 0:
            return 0.8 if recent_3 > 0 else 0.2
        
        momentum_ratio = recent_3 / previous_3
        
        # 对数函数平滑化
        heat = 0.5 + 0.3 * math.log(momentum_ratio + 0.1)
        return min(max(heat, 0.0), 1.0)
    
    def _calculate_volatility_heat(self, tail: int, recent_data: List[Dict]) -> float:
        """计算波动性热度"""
        appearances = [1 if tail in period.get('tails', []) else 0 for period in recent_data]
        
        if len(appearances) < 4:
            return 0.5
        
        # 计算滑动平均的变化
        window_size = 3
        moving_averages = []
        
        for i in range(len(appearances) - window_size + 1):
            avg = sum(appearances[i:i+window_size]) / window_size
            moving_averages.append(avg)
        
        if len(moving_averages) < 2:
            return 0.5
        
        # 计算变化率的平均绝对值
        changes = [abs(moving_averages[i] - moving_averages[i-1]) for i in range(1, len(moving_averages))]
        avg_change = sum(changes) / len(changes)
        
        # 波动性转换为热度（高波动 = 高关注度）
        volatility_heat = min(avg_change * 3, 1.0)
        return volatility_heat
    
    def _classify_heat_level(self, score: float) -> str:
        """分类热度等级"""
        if score > self.hot_threshold:
            return 'hot'
        elif score > 0.4:
            return 'warm'
        else:
            return 'cold'
    
    def _identify_dominant_heat_factor(self, freq: float, cont: float, inter: float, 
                                     patt: float, mom: float, vol: float) -> str:
        """识别主导热度因子"""
        factors = {
            'frequency': freq,
            'continuity': cont,
            'interval': inter,
            'pattern': patt,
            'momentum': mom,
            'volatility': vol
        }
        return max(factors.keys(), key=lambda k: factors[k])
    
    def _calculate_popularity_rankings(self, popularity_data: Dict) -> List[Tuple[int, float]]:
        """计算热门度排名"""
        rankings = [(tail, data['comprehensive_popularity']) for tail, data in popularity_data.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _analyze_heat_distribution(self, popularity_data: Dict) -> Dict:
        """分析热度分布"""
        scores = [data['comprehensive_popularity'] for data in popularity_data.values()]
        hot_count = len([s for s in scores if s > 0.65])
        warm_count = len([s for s in scores if 0.4 <= s <= 0.65])
        cold_count = len([s for s in scores if s < 0.4])
        
        return {
            'hot_count': hot_count,
            'warm_count': warm_count,
            'cold_count': cold_count,
            'distribution_type': 'concentrated' if hot_count <= 2 else 'dispersed'
        }
    
    def _summarize_popularity_analysis(self, popularity_data: Dict, popularity_rankings: List, 
                                     heat_distribution: Dict) -> str:
        """总结热门度分析"""
        hot_count = heat_distribution['hot_count']
        total_count = len(popularity_data)
        distribution_type = heat_distribution['distribution_type']
        
        return f"识别到{hot_count}/{total_count}个热门尾数，热度分布呈{distribution_type}态势"
    
    # === 压力评估方法 ===
    
    def _calculate_base_payout_risk(self, tail: int, historical_data: List[Dict]) -> float:
        """计算基础赔付风险"""
        # 基于历史频率的基础风险
        total_periods = min(len(historical_data), 50)  # 最多看50期
        appearances = sum(1 for period in historical_data[:total_periods] if tail in period.get('tails', []))
        
        frequency = appearances / total_periods
        expected_frequency = 0.4  # 假设期望频率为40%
        
        deviation = abs(frequency - expected_frequency)
        base_risk = min(deviation * 2, 1.0)
        
        return base_risk
    
    def _calculate_popularity_pressure(self, popularity_info: Dict) -> float:
        """计算热门度压力"""
        comprehensive_popularity = popularity_info['comprehensive_popularity']
        
        # 热门度越高，赔付压力越大
        if comprehensive_popularity > 0.8:
            return 0.9
        elif comprehensive_popularity > 0.6:
            return 0.7
        elif comprehensive_popularity > 0.4:
            return 0.4
        else:
            return 0.1
    
    def _calculate_frequency_pressure(self, tail: int, historical_data: List[Dict]) -> float:
        """计算频率压力"""
        recent_10 = historical_data[:10] if len(historical_data) >= 10 else historical_data
        appearances = sum(1 for period in recent_10 if tail in period.get('tails', []))
        
        frequency = appearances / len(recent_10)
        
        # 高频率带来高压力
        if frequency >= 0.8:
            return 0.95
        elif frequency >= 0.6:
            return 0.75
        elif frequency >= 0.4:
            return 0.5
        else:
            return 0.2
    
    def _calculate_continuity_pressure(self, tail: int, historical_data: List[Dict]) -> float:
        """计算连续性压力"""
        max_consecutive = 0
        current_consecutive = 0
        
        for period in historical_data[:20]:  # 检查最近20期
            if tail in period.get('tails', []):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # 连续出现带来巨大压力
        if max_consecutive >= 4:
            return 0.95
        elif max_consecutive == 3:
            return 0.8
        elif max_consecutive == 2:
            return 0.6
        else:
            return 0.3
    
    def _calculate_pattern_pressure(self, tail: int, historical_data: List[Dict]) -> float:
        """计算模式压力"""
        # 检测是否存在明显的规律性模式
        appearances = [1 if tail in period.get('tails', []) else 0 for period in historical_data[:20]]
        
        # 简单的模式检测
        pattern_detected = False
        
        # 检测交替模式 (1010...)
        alternating_score = 0
        for i in range(1, len(appearances)):
            if appearances[i] != appearances[i-1]:
                alternating_score += 1
        
        if alternating_score / len(appearances) > 0.8:
            pattern_detected = True
        
        # 检测周期模式
        for cycle in [2, 3, 4, 5]:
            if len(appearances) >= cycle * 2:
                matches = sum(1 for i in range(cycle, len(appearances)) 
                            if appearances[i] == appearances[i-cycle])
                if matches / (len(appearances) - cycle) > 0.7:
                    pattern_detected = True
                    break
        
        return 0.8 if pattern_detected else 0.3
    
    def _classify_pressure_level(self, pressure: float) -> str:
        """分类压力等级"""
        if pressure > self.pressure_threshold:
            return 'high'
        elif pressure > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _predict_kill_probability(self, pressure: float, popularity_info: Dict) -> float:
        """预测杀号概率"""
        base_kill_prob = self.risk_model['kill_probability_base']
        
        # 压力调整
        pressure_adjustment = pressure * self.risk_model['pressure_amplifier']
        
        # 热门度调整
        popularity_adjustment = popularity_info['comprehensive_popularity'] * self.risk_model['hot_penalty']
        
        # 综合杀号概率
        kill_probability = base_kill_prob + pressure_adjustment + popularity_adjustment
        
        return min(max(kill_probability, 0.0), 1.0)
    
    def _identify_pressure_risk_factors(self, base_risk: float, popularity_pressure: float,
                                      frequency_pressure: float, continuity_pressure: float,
                                      pattern_pressure: float) -> List[str]:
        """识别压力风险因子"""
        risk_factors = []
        
        if base_risk > 0.6:
            risk_factors.append('historical_deviation')
        if popularity_pressure > 0.7:
            risk_factors.append('high_popularity')
        if frequency_pressure > 0.7:
            risk_factors.append('high_frequency')
        if continuity_pressure > 0.7:
            risk_factors.append('consecutive_appearance')
        if pattern_pressure > 0.6:
            risk_factors.append('pattern_detected')
        
        return risk_factors
    
    def _calculate_pressure_rankings(self, pressure_data: Dict) -> List[Tuple[int, float]]:
        """计算压力排名"""
        rankings = [(tail, data['comprehensive_pressure']) for tail, data in pressure_data.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _analyze_overall_pressure(self, pressure_data: Dict) -> Dict:
        """分析整体压力"""
        pressures = [data['comprehensive_pressure'] for data in pressure_data.values()]
        avg_pressure = sum(pressures) / len(pressures)
        max_pressure = max(pressures)
        high_pressure_count = len([p for p in pressures if p > 0.7])
        
        return {
            'average_pressure': avg_pressure,
            'maximum_pressure': max_pressure,
            'high_pressure_count': high_pressure_count,
            'pressure_level': 'high' if avg_pressure > 0.6 else 'moderate' if avg_pressure > 0.4 else 'low'
        }
    
    def _summarize_pressure_analysis(self, pressure_data: Dict, pressure_rankings: List, 
                                   overall_pressure: Dict) -> str:
        """总结压力分析"""
        high_pressure_count = overall_pressure['high_pressure_count']
        total_count = len(pressure_data)
        pressure_level = overall_pressure['pressure_level']
        
        return f"发现{high_pressure_count}/{total_count}个高压力尾数，整体压力等级为{pressure_level}"
    
    # === 风险识别方法 ===
    
    def _calculate_compound_risk(self, pressure_info: Dict, flow_info: Dict) -> float:
        """计算复合风险"""
        pressure_risk = pressure_info['comprehensive_pressure']
        flow_risk = flow_info['comprehensive_score']
        
        # 压力和流向的乘积效应
        compound = pressure_risk * flow_risk * 1.5
        
        return min(compound, 1.0)
    
    def _assess_timing_risk(self, tail: int, pressure_info: Dict, flow_info: Dict) -> float:
        """评估时机风险"""
        # 基于压力累积时间和流向变化
        pressure_level = pressure_info['pressure_level']
        flow_trend = flow_info.get('trend_direction', 'stable')
        
        timing_risk = 0.5  # 基础时机风险
        
        if pressure_level == 'high' and flow_trend == 'up':
            timing_risk = 0.9  # 高压力且上升趋势
        elif pressure_level == 'high':
            timing_risk = 0.7  # 仅高压力
        elif flow_trend == 'up':
            timing_risk = 0.6  # 仅上升趋势
        
        return timing_risk
    
    def _analyze_strategy_risk(self, tail: int, pressure_info: Dict, flow_info: Dict) -> float:
        """分析策略风险"""
        # 基于庄家可能的策略选择
        kill_probability = pressure_info['kill_probability']
        flow_level = flow_info['flow_level']
        
        if kill_probability > 0.8 and flow_level == 'high':
            return 0.95  # 极高策略风险
        elif kill_probability > 0.6:
            return 0.8   # 高策略风险
        elif flow_level == 'high':
            return 0.6   # 中等策略风险
        else:
            return 0.3   # 低策略风险
    
    def _assess_market_risk(self, tail: int, pressure_info: Dict, flow_info: Dict) -> float:
        """评估市场风险"""
        # 基于市场整体环境
        risk_factors = pressure_info.get('risk_factors', [])
        flow_stability = flow_info.get('flow_stability', 0.5)
        
        market_risk = 0.4  # 基础市场风险
        
        # 风险因子越多，市场风险越高
        market_risk += len(risk_factors) * 0.1
        
        # 流向不稳定增加市场风险
        market_risk += (1.0 - flow_stability) * 0.3
        
        return min(market_risk, 1.0)
    
    def _classify_risk_level(self, risk: float) -> str:
        """分类风险等级"""
        if risk > self.risk_threshold:
            return 'high'
        elif risk > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _predict_action_probability(self, risk: float) -> float:
        """预测行动概率"""
        # 风险越高，庄家采取行动的概率越高
        action_prob = risk * 0.8 + 0.1  # 基础概率10%
        return min(action_prob, 0.95)
    
    def _compile_risk_indicators(self, compound_risk: float, timing_risk: float,
                                strategy_risk: float, market_risk: float) -> List[str]:
        """编译风险指标"""
        indicators = []
        
        if compound_risk > 0.7:
            indicators.append('high_compound_risk')
        if timing_risk > 0.7:
            indicators.append('critical_timing')
        if strategy_risk > 0.8:
            indicators.append('strategy_target')
        if market_risk > 0.6:
            indicators.append('market_volatility')
        
        return indicators
    
    def _detect_warning_signals(self, tail: int, pressure_info: Dict, flow_info: Dict) -> List[str]:
        """检测警告信号"""
        warnings = []
        
        if pressure_info['kill_probability'] > 0.8:
            warnings.append('high_kill_probability')
        if flow_info['flow_level'] == 'high' and flow_info['trend_direction'] == 'up':
            warnings.append('accelerating_flow')
        if len(pressure_info.get('risk_factors', [])) >= 3:
            warnings.append('multiple_risk_factors')
        
        return warnings
    
    def _calculate_risk_rankings(self, risk_data: Dict) -> List[Tuple[int, float]]:
        """计算风险排名"""
        rankings = [(tail, data['comprehensive_risk']) for tail, data in risk_data.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _assess_systemic_risk(self, risk_data: Dict) -> Dict:
        """评估系统性风险"""
        risks = [data['comprehensive_risk'] for data in risk_data.values()]
        avg_risk = sum(risks) / len(risks)
        max_risk = max(risks)
        high_risk_count = len([r for r in risks if r > 0.7])
        
        return {
            'average_risk': avg_risk,
            'maximum_risk': max_risk,
            'high_risk_count': high_risk_count,
            'systemic_level': 'high' if avg_risk > 0.6 else 'moderate' if avg_risk > 0.4 else 'low'
        }
    
    def _summarize_risk_analysis(self, risk_data: Dict, risk_rankings: List, 
                                systemic_risk: Dict) -> str:
        """总结风险分析"""
        high_risk_count = systemic_risk['high_risk_count']
        total_count = len(risk_data)
        systemic_level = systemic_risk['systemic_level']
        
        return f"发现{high_risk_count}/{total_count}个高风险尾数，系统风险等级为{systemic_level}"
    
    # === 策略建议方法 ===
    
    def _calculate_historical_opportunity(self, tail: int) -> float:
        """计算历史机会性"""
        # 这里可以基于更长期的历史数据计算
        # 简化实现：基于尾数特性
        if tail in [0, 5]:  # 0和5结尾的数字通常较少
            return 0.7
        elif tail in [1, 2, 3, 4, 6, 7, 8, 9]:
            return 0.5
        else:
            return 0.3
    
    def _calculate_overall_confidence(self, candidate_tails: List[int], analysis_data: Dict,
                                    recommended: List[int], avoid_choices: List[int]) -> float:
        """计算整体置信度"""
        if not recommended:
            return 0.2
        
        # 基于推荐尾数的综合评分
        total_confidence = 0.0
        for tail in recommended:
            data = analysis_data[tail]
            
            # 综合各项指标
            safety = 1.0 - data['risk']['comprehensive_risk']
            opportunity = 1.0 - data['pressure']['comprehensive_pressure']
            stability = data['flow']['flow_stability']
            
            tail_confidence = (safety * 0.4 + opportunity * 0.3 + stability * 0.3)
            total_confidence += tail_confidence
        
        avg_confidence = total_confidence / len(recommended)
        
        # 根据避开选择的数量调整置信度
        avoidance_factor = 1.0 - (len(avoid_choices) / len(candidate_tails)) * 0.2
        
        return min(avg_confidence * avoidance_factor, 0.95)
    
    def _generate_reasoning(self, candidate_tails: List[int], analysis_data: Dict,
                          recommended: List[int], avoid_choices: List[int],
                          safe_choices: List[int], opportunity_choices: List[int],
                          safety_scores: Dict, opportunity_scores: Dict) -> str:
        """生成推理说明"""
        reasoning_parts = []
        
        # 推荐理由
        if recommended:
            if set(recommended) & set(safe_choices) & set(opportunity_choices):
                reasoning_parts.append(f"推荐尾数{recommended}同时具备安全性和机会性")
            elif set(recommended) & set(safe_choices):
                reasoning_parts.append(f"推荐尾数{recommended}具有较高的安全性评分")
            elif set(recommended) & set(opportunity_choices):
                reasoning_parts.append(f"推荐尾数{recommended}具有良好的机会性评分")
            else:
                reasoning_parts.append(f"推荐尾数{recommended}在综合分析中风险相对较低")
        
        # 避开理由
        if avoid_choices:
            high_risk_reasons = []
            for tail in avoid_choices:
                data = analysis_data[tail]
                if data['risk']['risk_level'] == 'high':
                    high_risk_reasons.append(f"尾数{tail}(风险{data['risk']['comprehensive_risk']:.2f})")
                elif data['pressure']['pressure_level'] == 'high':
                    high_risk_reasons.append(f"尾数{tail}(压力{data['pressure']['comprehensive_pressure']:.2f})")
            
            if high_risk_reasons:
                reasoning_parts.append(f"避开{', '.join(high_risk_reasons[:3])}等高风险尾数")
        
        # 分析总结
        total_analyzed = len(candidate_tails)
        safe_count = len(safe_choices)
        opportunity_count = len(opportunity_choices)
        
        reasoning_parts.append(f"共分析{total_analyzed}个候选尾数，识别出{safe_count}个安全选择和{opportunity_count}个机会选择")
        
        return "；".join(reasoning_parts)
    
    def _determine_strategy_type(self, recommended: List[int], avoid_choices: List[int],
                               safe_choices: List[int], opportunity_choices: List[int]) -> str:
        """确定策略类型"""
        if set(recommended) & set(safe_choices) & set(opportunity_choices):
            return 'balanced'  # 平衡策略
        elif set(recommended) & set(safe_choices):
            return 'conservative'  # 保守策略
        elif set(recommended) & set(opportunity_choices):
            return 'aggressive'  # 激进策略
        else:
            return 'defensive'  # 防守策略
    
    def _breakdown_confidence(self, overall_confidence: float, analysis_data: Dict) -> Dict:
        """分解置信度"""
        return {
            'overall': overall_confidence,
            'data_quality': min(len(analysis_data) / 10.0, 1.0),  # 数据质量评分
            'analysis_depth': 0.9,  # 分析深度评分
            'model_reliability': 0.8  # 模型可靠性评分
        }
    
    # === 详细指标编译方法 ===
    
    def _compile_detailed_metrics(self, candidate_tails: List[int], flow_analysis: Dict,
                                popularity_analysis: Dict, pressure_analysis: Dict, 
                                risk_analysis: Dict) -> Dict:
        """编译详细指标"""
        detailed_metrics = {}
        
        for tail in candidate_tails:
            detailed_metrics[tail] = {
                'flow_metrics': {
                    'strength': flow_analysis['individual_flows'][tail]['flow_strength'],
                    'trend': flow_analysis['individual_flows'][tail]['flow_trend'],
                    'stability': flow_analysis['individual_flows'][tail]['flow_stability'],
                    'momentum': flow_analysis['individual_flows'][tail]['flow_momentum'],
                    'level': flow_analysis['individual_flows'][tail]['flow_level']
                },
                'popularity_metrics': {
                    'frequency_heat': popularity_analysis['individual_popularity'][tail]['frequency_heat'],
                    'continuity_heat': popularity_analysis['individual_popularity'][tail]['continuity_heat'],
                    'pattern_heat': popularity_analysis['individual_popularity'][tail]['pattern_heat'],
                    'overall_heat': popularity_analysis['individual_popularity'][tail]['comprehensive_popularity'],
                    'heat_level': popularity_analysis['individual_popularity'][tail]['heat_level']
                },
                'pressure_metrics': {
                    'payout_pressure': pressure_analysis['individual_pressure'][tail]['comprehensive_pressure'],
                    'kill_probability': pressure_analysis['individual_pressure'][tail]['kill_probability'],
                    'pressure_level': pressure_analysis['individual_pressure'][tail]['pressure_level'],
                    'risk_factors': pressure_analysis['individual_pressure'][tail]['risk_factors']
                },
                'risk_metrics': {
                    'comprehensive_risk': risk_analysis['individual_risk'][tail]['comprehensive_risk'],
                    'action_probability': risk_analysis['individual_risk'][tail]['action_probability'],
                    'risk_level': risk_analysis['individual_risk'][tail]['risk_level'],
                    'warning_signals': risk_analysis['individual_risk'][tail]['warning_signals']
                }
            }
        
        return detailed_metrics
    
    # === 统计和缓存方法 ===
    
    def _update_analysis_stats(self) -> Dict:
        """更新分析统计"""
        self.analysis_stats['total_analyses'] += 1
        return self.analysis_stats.copy()
    
    def _cache_analysis_result(self, result: Dict):
        """缓存分析结果"""
        cache_entry = {
            'timestamp': result['timestamp'],
            'candidate_tails': result['candidate_tails'],
            'recommended_tails': result['recommended_tails'],
            'confidence': result['confidence']
        }
        
        self.flow_cache.append(cache_entry)
    
    def _create_empty_result(self, message: str) -> Dict:
        """创建空结果"""
        return {
            'success': False,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'recommended_tails': [],
            'avoid_tails': [],
            'confidence': 0.0,
            'reasoning': f"分析失败：{message}"
        }
    
    def learn_from_outcome(self, prediction_result: Dict, actual_tails: List[int]) -> Dict:
        """从结果中学习"""
        try:
            if not prediction_result or not actual_tails:
                return {'learning_success': False, 'message': '输入数据不足'}
            
            recommended_tails = prediction_result.get('recommended_tails', [])
            avoid_tails = prediction_result.get('avoid_tails', [])
            
            # 计算推荐准确性
            recommend_correct = any(tail in actual_tails for tail in recommended_tails) if recommended_tails else False
            
            # 计算避开准确性  
            avoid_correct = not any(tail in actual_tails for tail in avoid_tails) if avoid_tails else True
            
            # 更新统计
            if recommend_correct:
                self.analysis_stats['successful_predictions'] += 1
            else:
                self.analysis_stats['false_positives'] += 1
            
            # 计算准确率
            total_predictions = self.analysis_stats['successful_predictions'] + self.analysis_stats['false_positives']
            current_accuracy = self.analysis_stats['successful_predictions'] / total_predictions if total_predictions > 0 else 0.0
            
            self.analysis_stats['accuracy_history'].append(current_accuracy)
            
            # 保持历史记录在合理长度
            if len(self.analysis_stats['accuracy_history']) > 100:
                self.analysis_stats['accuracy_history'].pop(0)
            
            print(f"💰 资金流向分析器学习完成:")
            print(f"   推荐准确: {recommend_correct}")
            print(f"   避开准确: {avoid_correct}")
            print(f"   当前准确率: {current_accuracy:.3f}")
            
            return {
                'learning_success': True,
                'recommend_accuracy': recommend_correct,
                'avoid_accuracy': avoid_correct,
                'overall_accuracy': current_accuracy,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            print(f"❌ 资金流向分析器学习失败: {e}")
            return {'learning_success': False, 'message': f'学习过程出错: {str(e)}'}
    
    def get_analysis_statistics(self) -> Dict:
        """获取分析统计信息"""
        return {
            'total_analyses': self.analysis_stats['total_analyses'],
            'successful_predictions': self.analysis_stats['successful_predictions'],
            'false_positives': self.analysis_stats['false_positives'],
            'current_accuracy': self.analysis_stats['accuracy_history'][-1] if self.analysis_stats['accuracy_history'] else 0.0,
            'accuracy_trend': self.analysis_stats['accuracy_history'][-10:] if len(self.analysis_stats['accuracy_history']) >= 10 else self.analysis_stats['accuracy_history'],
            'model_status': 'active',
            'last_analysis': self.flow_cache[-1] if self.flow_cache else None
        }