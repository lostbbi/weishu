# ai_engine/prediction/game_theory_strategist.py - 博弈论策略器

import numpy as np
import math
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import itertools

class GameTheoryStrategist:
    """
    基于博弈论的策略模型
    - 多方博弈分析（玩家-庄家-系统三方博弈）
    - 纳什均衡计算
    - 最优策略选择
    - 基于"杀多赔少"原理的反向策略
    """
    
    def __init__(self):
        self.name = "GameTheoryStrategist"
        self.version = "1.0.0"
        
        # 博弈论参数
        self.game_history = []  # 博弈历史记录
        self.strategy_effectiveness = {}  # 策略有效性记录
        self.nash_equilibrium_cache = {}  # 纳什均衡缓存
        
        # 三方博弈参数
        self.player_strategies = {}  # 玩家策略分布
        self.banker_strategies = {}  # 庄家策略分布
        self.system_randomness = 0.5  # 系统随机性权重
        
        # 学习参数
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.memory_depth = 100  # 记忆深度
        
        # 策略类型定义
        self.strategy_types = {
            'aggressive': {'risk_tolerance': 0.8, 'bet_multiplier': 1.5},
            'conservative': {'risk_tolerance': 0.3, 'bet_multiplier': 0.8},
            'balanced': {'risk_tolerance': 0.5, 'bet_multiplier': 1.0},
            'contrarian': {'risk_tolerance': 0.6, 'bet_multiplier': 1.2},  # 反向策略
            'adaptive': {'risk_tolerance': 0.4, 'bet_multiplier': 1.1}
        }
        
        # 纳什均衡求解参数
        self.nash_convergence_threshold = 1e-6
        self.nash_max_iterations = 1000
        
        print(f"✅ {self.name} 初始化完成")
    
    def analyze_game_state(self, candidate_tails: List[int], data_list: List[Dict]) -> Dict:
        """
        分析当前博弈状态
        """
        try:
            if not data_list or not candidate_tails:
                return {'success': False, 'reason': '数据不足'}
            
            # 1. 分析历史博弈模式
            historical_patterns = self._analyze_historical_patterns(data_list)
            
            # 2. 识别当前博弈参与方的策略倾向
            game_participants = self._identify_game_participants(data_list, candidate_tails)
            
            # 3. 计算三方博弈的收益矩阵
            payoff_matrix = self._calculate_payoff_matrix(candidate_tails, data_list, historical_patterns)
            
            # 4. 分析庄家的"杀多赔少"策略
            banker_kill_strategy = self._analyze_banker_kill_strategy(data_list, candidate_tails)
            
            # 5. 预测玩家群体行为
            crowd_behavior = self._predict_crowd_behavior(data_list, candidate_tails)
            
            return {
                'success': True,
                'historical_patterns': historical_patterns,
                'game_participants': game_participants,
                'payoff_matrix': payoff_matrix,
                'banker_strategy': banker_kill_strategy,
                'crowd_behavior': crowd_behavior,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'reason': f'博弈状态分析失败: {e}'
            }
    
    def _analyze_historical_patterns(self, data_list: List[Dict]) -> Dict:
        """分析历史博弈模式"""
        patterns = {
            'kill_patterns': [],  # 杀号模式
            'compensation_patterns': [],  # 补偿模式
            'manipulation_signals': [],  # 操控信号
            'randomness_periods': []  # 随机期间
        }
        
        if len(data_list) < 10:
            return patterns
        
        # 分析最近20期的杀号模式
        recent_data = data_list[:20]
        
        for i, period in enumerate(recent_data[:-1]):
            current_tails = set(period.get('tails', []))
            next_period_tails = set(recent_data[i + 1].get('tails', []))
            
            # 识别杀号模式：当期热门尾数在下期被"杀掉"
            killed_tails = current_tails - next_period_tails
            if len(killed_tails) >= 2:  # 至少杀掉2个尾数才算有效模式
                patterns['kill_patterns'].append({
                    'period_index': i,
                    'killed_tails': list(killed_tails),
                    'survival_tails': list(current_tails & next_period_tails),
                    'new_tails': list(next_period_tails - current_tails)
                })
            
            # 识别补偿模式：冷门尾数突然出现
            if i >= 3:
                # 检查过去3期都没出现的尾数
                past_3_tails = set()
                for j in range(1, 4):
                    if i + j < len(recent_data):
                        past_3_tails.update(recent_data[i + j].get('tails', []))
                
                compensated_tails = next_period_tails - past_3_tails
                if compensated_tails:
                    patterns['compensation_patterns'].append({
                        'period_index': i,
                        'compensated_tails': list(compensated_tails),
                        'compensation_strength': len(compensated_tails)
                    })
        
        # 分析操控信号强度
        patterns['manipulation_strength'] = self._calculate_manipulation_strength(recent_data)
        
        return patterns
    
    def _calculate_manipulation_strength(self, data_list: List[Dict]) -> float:
        """计算操控信号强度"""
        if len(data_list) < 5:
            return 0.5
        
        # 基于多个指标计算操控强度
        manipulation_indicators = []
        
        # 指标1：尾数分布的均匀性偏差
        all_tails = []
        for period in data_list[:10]:
            all_tails.extend(period.get('tails', []))
        
        if all_tails:
            tail_counts = Counter(all_tails)
            expected_count = len(all_tails) / 10
            variance = np.var([tail_counts.get(i, 0) for i in range(10)])
            expected_variance = expected_count * (1 - 1/10)  # 二项分布方差
            
            if expected_variance > 0:
                uniformity_deviation = abs(variance - expected_variance) / expected_variance
                manipulation_indicators.append(min(1.0, uniformity_deviation))
        
        # 指标2：连续性模式的规律性
        consecutive_patterns = 0
        total_checks = 0
        
        for i in range(len(data_list) - 1):
            current_tails = set(data_list[i].get('tails', []))
            next_tails = set(data_list[i + 1].get('tails', []))
            
            # 检查是否有明显的延续或反转模式
            overlap = len(current_tails & next_tails)
            overlap_ratio = overlap / len(current_tails) if current_tails else 0
            
            total_checks += 1
            if overlap_ratio < 0.3 or overlap_ratio > 0.8:  # 过低或过高的重叠率都可能是操控
                consecutive_patterns += 1
        
        if total_checks > 0:
            pattern_regularity = consecutive_patterns / total_checks
            manipulation_indicators.append(pattern_regularity)
        
        # 指标3：极值频率异常
        extreme_events = 0
        for period in data_list[:10]:
            tail_count = len(period.get('tails', []))
            if tail_count <= 3 or tail_count >= 8:  # 极少或极多的尾数都是异常
                extreme_events += 1
        
        extreme_frequency = extreme_events / min(10, len(data_list))
        manipulation_indicators.append(extreme_frequency)
        
        # 综合计算操控强度
        if manipulation_indicators:
            manipulation_strength = np.mean(manipulation_indicators)
            return max(0.1, min(0.9, manipulation_strength))
        
        return 0.5
    
    def _identify_game_participants(self, data_list: List[Dict], candidate_tails: List[int]) -> Dict:
        """识别博弈参与方及其策略倾向"""
        participants = {
            'players': {
                'popular_choices': [],  # 玩家热门选择
                'risk_preference': 'moderate',  # 风险偏好
                'strategy_consistency': 0.5  # 策略一致性
            },
            'banker': {
                'kill_targets': [],  # 庄家杀号目标
                'profit_maximization': 0.5,  # 利润最大化倾向
                'manipulation_frequency': 0.5  # 操控频率
            },
            'system': {
                'randomness_level': 0.5,  # 随机性水平
                'bias_indicators': {}  # 系统偏差指标
            }
        }
        
        if len(data_list) < 5:
            return participants
        
        # 分析玩家行为模式
        recent_tails = []
        for period in data_list[:15]:
            recent_tails.extend(period.get('tails', []))
        
        if recent_tails:
            tail_frequency = Counter(recent_tails)
            # 玩家通常偏好出现频率高的尾数
            popular_tails = [tail for tail, count in tail_frequency.most_common(5)]
            participants['players']['popular_choices'] = popular_tails
            
            # 计算玩家风险偏好
            # 如果玩家集中选择少数几个尾数，说明风险偏好低（保守）
            top_3_frequency = sum(count for _, count in tail_frequency.most_common(3))
            total_frequency = sum(tail_frequency.values())
            concentration = top_3_frequency / total_frequency if total_frequency > 0 else 0
            
            if concentration > 0.7:
                participants['players']['risk_preference'] = 'conservative'
            elif concentration < 0.4:
                participants['players']['risk_preference'] = 'aggressive'
            else:
                participants['players']['risk_preference'] = 'moderate'
        
        # 分析庄家策略
        # 庄家的杀号目标通常是热门尾数
        if len(data_list) >= 3:
            # 统计哪些尾数经常在热门后被"杀掉"
            kill_statistics = defaultdict(int)
            total_kill_opportunities = 0
            
            for i in range(len(data_list) - 1):
                current_tails = set(data_list[i].get('tails', []))
                next_tails = set(data_list[i + 1].get('tails', []))
                
                killed_tails = current_tails - next_tails
                for tail in killed_tails:
                    kill_statistics[tail] += 1
                    total_kill_opportunities += 1
            
            if total_kill_opportunities > 0:
                # 计算各尾数被杀概率
                kill_probabilities = {tail: count / total_kill_opportunities 
                                    for tail, count in kill_statistics.items()}
                
                # 选择被杀概率最高的尾数作为庄家目标
                potential_kill_targets = sorted(kill_probabilities.items(), 
                                              key=lambda x: x[1], reverse=True)[:3]
                participants['banker']['kill_targets'] = [tail for tail, _ in potential_kill_targets]
        
        return participants
    
    def _calculate_payoff_matrix(self, candidate_tails: List[int], data_list: List[Dict], 
                                historical_patterns: Dict) -> Dict:
        """计算三方博弈的收益矩阵"""
        matrix = {}
        
        for tail in candidate_tails:
            matrix[tail] = {
                'player_payoff': self._calculate_player_payoff(tail, data_list, historical_patterns),
                'banker_payoff': self._calculate_banker_payoff(tail, data_list, historical_patterns),
                'system_payoff': self._calculate_system_payoff(tail, data_list),
                'nash_potential': 0.0  # 将在纳什均衡计算中填充
            }
        
        return matrix
    
    def _calculate_player_payoff(self, tail: int, data_list: List[Dict], patterns: Dict) -> float:
        """计算玩家选择某尾数的期望收益"""
        if not data_list:
            return 0.5
        
        # 基础收益：历史出现频率
        recent_data = data_list[:20]
        appearance_count = sum(1 for period in recent_data if tail in period.get('tails', []))
        base_payoff = appearance_count / len(recent_data)
        
        # 调整因子1：补偿模式加成
        compensation_bonus = 0.0
        for pattern in patterns.get('compensation_patterns', []):
            if tail in pattern.get('compensated_tails', []):
                compensation_bonus += 0.1 * pattern.get('compensation_strength', 1)
        
        # 调整因子2：被杀风险惩罚
        kill_penalty = 0.0
        for pattern in patterns.get('kill_patterns', []):
            if tail in pattern.get('killed_tails', []):
                kill_penalty += 0.05
        
        # 调整因子3：连续性奖励
        continuity_bonus = 0.0
        if len(data_list) >= 2:
            if tail in data_list[0].get('tails', []) and tail in data_list[1].get('tails', []):
                continuity_bonus = 0.08
        
        total_payoff = base_payoff + compensation_bonus - kill_penalty + continuity_bonus
        return max(0.0, min(1.0, total_payoff))
    
    def _calculate_banker_payoff(self, tail: int, data_list: List[Dict], patterns: Dict) -> float:
        """计算庄家选择操控某尾数的期望收益（基于"杀多赔少"原理）"""
        if not data_list:
            return 0.5
        
        # 庄家收益与玩家收益相反（零和博弈特性）
        player_payoff = self._calculate_player_payoff(tail, data_list, patterns)
        
        # 基础收益：杀掉热门尾数的收益
        base_banker_payoff = 1.0 - player_payoff
        
        # 调整因子1：杀号历史成功率
        kill_success_bonus = 0.0
        kill_attempts = 0
        successful_kills = 0
        
        for pattern in patterns.get('kill_patterns', []):
            if tail in pattern.get('killed_tails', []):
                successful_kills += 1
            kill_attempts += 1
        
        if kill_attempts > 0:
            kill_success_rate = successful_kills / kill_attempts
            kill_success_bonus = (kill_success_rate - 0.5) * 0.2  # 超过50%成功率有奖励
        
        # 调整因子2：操控成本
        manipulation_cost = patterns.get('manipulation_strength', 0.5) * 0.1
        
        # 调整因子3：热门程度奖励（杀热门尾数收益更高）
        recent_data = data_list[:10]
        all_recent_tails = []
        for period in recent_data:
            all_recent_tails.extend(period.get('tails', []))
        
        if all_recent_tails:
            tail_frequency = Counter(all_recent_tails)
            tail_popularity = tail_frequency.get(tail, 0) / len(all_recent_tails)
            popularity_bonus = tail_popularity * 0.15  # 越热门，杀掉的收益越高
        else:
            popularity_bonus = 0.0
        
        total_banker_payoff = base_banker_payoff + kill_success_bonus - manipulation_cost + popularity_bonus
        return max(0.0, min(1.0, total_banker_payoff))
    
    def _calculate_system_payoff(self, tail: int, data_list: List[Dict]) -> float:
        """计算系统（随机性）的收益"""
        # 系统收益代表真正的随机性，倾向于平衡
        if not data_list:
            return 0.5
        
        # 基于长期均衡的理论概率
        theoretical_probability = 0.1  # 每个尾数理论上10%的出现概率
        
        # 计算实际偏差
        recent_data = data_list[:50] if len(data_list) >= 50 else data_list
        if recent_data:
            all_tails = []
            for period in recent_data:
                all_tails.extend(period.get('tails', []))
            
            if all_tails:
                actual_frequency = all_tails.count(tail) / len(all_tails)
                deviation = abs(actual_frequency - theoretical_probability)
                
                # 系统倾向于纠正偏差，偏差越大，系统越倾向于平衡
                system_payoff = 0.5 + (deviation * 2)  # 放大偏差影响
                return max(0.0, min(1.0, system_payoff))
        
        return 0.5
    
    def _analyze_banker_kill_strategy(self, data_list: List[Dict], candidate_tails: List[int]) -> Dict:
        """分析庄家的杀号策略"""
        strategy_analysis = {
            'kill_probability': {},  # 各尾数被杀概率
            'kill_timing_patterns': [],  # 杀号时机模式
            'optimal_kill_targets': [],  # 最优杀号目标
            'counter_strategy': {}  # 反制策略
        }
        
        if len(data_list) < 5:
            return strategy_analysis
        
        # 分析各尾数的被杀概率
        for tail in candidate_tails:
            kill_events = 0
            exposure_opportunities = 0
            
            for i in range(len(data_list) - 1):
                current_tails = set(data_list[i].get('tails', []))
                next_tails = set(data_list[i + 1].get('tails', []))
                
                if tail in current_tails:
                    exposure_opportunities += 1
                    if tail not in next_tails:
                        kill_events += 1
            
            if exposure_opportunities > 0:
                kill_probability = kill_events / exposure_opportunities
                strategy_analysis['kill_probability'][tail] = kill_probability
            else:
                strategy_analysis['kill_probability'][tail] = 0.5
        
        # 识别杀号时机模式
        for i in range(len(data_list) - 2):
            current_tails = set(data_list[i].get('tails', []))
            next_tails = set(data_list[i + 1].get('tails', []))
            killed_tails = current_tails - next_tails
            
            if killed_tails:
                # 分析杀号前的市场状态
                previous_period_tails = set(data_list[i + 1].get('tails', [])) if i + 1 < len(data_list) else set()
                
                pattern = {
                    'period_index': i,
                    'pre_kill_state': {
                        'hot_tails': list(current_tails),
                        'killed_tails': list(killed_tails),
                        'previous_tails': list(previous_period_tails)
                    },
                    'kill_effectiveness': len(killed_tails) / len(current_tails) if current_tails else 0
                }
                strategy_analysis['kill_timing_patterns'].append(pattern)
        
        # 确定最优杀号目标（对于庄家来说）
        if strategy_analysis['kill_probability']:
            sorted_targets = sorted(strategy_analysis['kill_probability'].items(), 
                                  key=lambda x: x[1], reverse=True)
            strategy_analysis['optimal_kill_targets'] = [tail for tail, prob in sorted_targets[:3]]
        
        # 制定反制策略（对玩家有利）
        for tail in candidate_tails:
            kill_prob = strategy_analysis['kill_probability'].get(tail, 0.5)
            
            # 反制策略：如果被杀概率高，降低选择权重；如果被杀概率低，提高权重
            if kill_prob > 0.6:
                counter_weight = 0.3  # 高风险，低权重
                strategy_type = 'avoid_hot_target'
            elif kill_prob < 0.3:
                counter_weight = 0.8  # 低风险，高权重
                strategy_type = 'exploit_safe_target'
            else:
                counter_weight = 0.5  # 中等风险，中等权重
                strategy_type = 'neutral'
            
            strategy_analysis['counter_strategy'][tail] = {
                'weight': counter_weight,
                'strategy_type': strategy_type,
                'risk_level': 'high' if kill_prob > 0.6 else 'low' if kill_prob < 0.3 else 'medium'
            }
        
        return strategy_analysis
    
    def _predict_crowd_behavior(self, data_list: List[Dict], candidate_tails: List[int]) -> Dict:
        """预测群体行为"""
        crowd_analysis = {
            'popularity_ranking': [],  # 受欢迎程度排名
            'herd_tendency': 0.5,  # 从众倾向
            'contrarian_opportunity': {},  # 反向机会
            'crowd_consensus': None  # 群体共识
        }
        
        if not data_list or not candidate_tails:
            return crowd_analysis
        
        # 分析各尾数的群体受欢迎程度
        recent_data = data_list[:15]
        tail_appearances = Counter()
        
        for period in recent_data:
            for tail in period.get('tails', []):
                tail_appearances[tail] += 1
        
        # 只考虑候选尾数的受欢迎程度
        candidate_popularity = {tail: tail_appearances.get(tail, 0) for tail in candidate_tails}
        popularity_ranking = sorted(candidate_popularity.items(), key=lambda x: x[1], reverse=True)
        crowd_analysis['popularity_ranking'] = popularity_ranking
        
        # 计算从众倾向
        if popularity_ranking:
            total_appearances = sum(candidate_popularity.values())
            if total_appearances > 0:
                top_3_share = sum(count for _, count in popularity_ranking[:3]) / total_appearances
                crowd_analysis['herd_tendency'] = top_3_share
        
        # 识别反向操作机会
        for tail in candidate_tails:
            popularity = candidate_popularity.get(tail, 0)
            popularity_percentile = self._calculate_popularity_percentile(tail, data_list)
            
            # 反向机会：不受欢迎但有潜力的尾数
            if popularity_percentile < 0.3:  # 受欢迎程度低于30%分位
                # 检查是否有复出潜力
                recent_absence = self._calculate_recent_absence(tail, data_list)
                if recent_absence >= 3:  # 连续3期未出现
                    opportunity_score = (recent_absence / 10) * (1 - popularity_percentile)
                    crowd_analysis['contrarian_opportunity'][tail] = {
                        'score': min(1.0, opportunity_score),
                        'reason': f'连续{recent_absence}期未出现，受欢迎程度低',
                        'risk_level': 'medium'
                    }
        
        # 确定群体共识
        if popularity_ranking:
            most_popular = popularity_ranking[0][0]
            most_popular_count = popularity_ranking[0][1]
            total_periods = len(recent_data)
            
            if most_popular_count >= total_periods * 0.6:  # 60%以上期数出现
                crowd_analysis['crowd_consensus'] = {
                    'tail': most_popular,
                    'confidence': most_popular_count / total_periods,
                    'consensus_type': 'strong_favorite'
                }
            elif len(popularity_ranking) >= 2 and popularity_ranking[1][1] >= most_popular_count * 0.8:
                crowd_analysis['crowd_consensus'] = {
                    'tail': None,
                    'confidence': 0.3,
                    'consensus_type': 'divided_opinion'
                }
            else:
                crowd_analysis['crowd_consensus'] = {
                    'tail': most_popular,
                    'confidence': most_popular_count / total_periods,
                    'consensus_type': 'moderate_favorite'
                }
        
        return crowd_analysis
    
    def _calculate_popularity_percentile(self, tail: int, data_list: List[Dict]) -> float:
        """计算尾数受欢迎程度的百分位数"""
        if len(data_list) < 10:
            return 0.5
        
        recent_data = data_list[:20]
        all_tail_counts = Counter()
        
        for period in recent_data:
            for t in period.get('tails', []):
                all_tail_counts[t] += 1
        
        if not all_tail_counts:
            return 0.5
        
        tail_count = all_tail_counts.get(tail, 0)
        sorted_counts = sorted(all_tail_counts.values(), reverse=True)
        
        if tail_count == 0:
            return 0.0
        
        rank = sorted_counts.index(tail_count) + 1
        percentile = 1.0 - (rank - 1) / len(sorted_counts)
        return percentile
    
    def _calculate_recent_absence(self, tail: int, data_list: List[Dict]) -> int:
        """计算尾数最近连续未出现的期数"""
        absence_count = 0
        
        for period in data_list:
            if tail in period.get('tails', []):
                break
            absence_count += 1
        
        return absence_count
    
    def calculate_nash_equilibrium(self, candidate_tails: List[int], payoff_matrix: Dict) -> Dict:
        """计算纳什均衡"""
        nash_result = {
            'equilibrium_found': False,
            'equilibrium_strategies': {},
            'equilibrium_payoffs': {},
            'stability_score': 0.0,
            'computation_details': {}
        }
        
        try:
            if len(candidate_tails) < 2:
                return nash_result
            
            # 简化的纳什均衡求解（混合策略）
            # 为三方博弈构建策略概率分布
            
            # 初始化策略概率
            num_tails = len(candidate_tails)
            player_probabilities = np.ones(num_tails) / num_tails  # 玩家选择各尾数的概率
            banker_probabilities = np.ones(num_tails) / num_tails  # 庄家操控各尾数的概率
            system_probabilities = np.ones(num_tails) / num_tails  # 系统随机选择概率
            
            # 迭代求解纳什均衡
            for iteration in range(self.nash_max_iterations):
                old_player_probs = player_probabilities.copy()
                old_banker_probs = banker_probabilities.copy()
                
                # 更新玩家策略（基于期望收益）
                player_expected_payoffs = np.zeros(num_tails)
                for i, tail in enumerate(candidate_tails):
                    # 计算选择该尾数的期望收益
                    player_payoff = payoff_matrix[tail]['player_payoff']
                    banker_counter_prob = banker_probabilities[i]  # 庄家反制概率
                    
                    # 期望收益 = 基础收益 × (1 - 庄家反制概率)
                    player_expected_payoffs[i] = player_payoff * (1 - banker_counter_prob * 0.5)
                
                # 使用softmax更新玩家策略
                player_probabilities = self._softmax(player_expected_payoffs, temperature=0.1)
                
                # 更新庄家策略（基于"杀多赔少"原理）
                banker_expected_payoffs = np.zeros(num_tails)
                for i, tail in enumerate(candidate_tails):
                    banker_payoff = payoff_matrix[tail]['banker_payoff']
                    player_choice_prob = player_probabilities[i]  # 玩家选择概率
                    
                    # 庄家收益 = 基础收益 × 玩家选择概率（杀热门更有利）
                    banker_expected_payoffs[i] = banker_payoff * player_choice_prob
                
                banker_probabilities = self._softmax(banker_expected_payoffs, temperature=0.1)
                
                # 检查收敛性
                player_change = np.sum(np.abs(player_probabilities - old_player_probs))
                banker_change = np.sum(np.abs(banker_probabilities - old_banker_probs))
                
                if player_change < self.nash_convergence_threshold and banker_change < self.nash_convergence_threshold:
                    nash_result['equilibrium_found'] = True
                    nash_result['computation_details']['iterations'] = iteration + 1
                    nash_result['computation_details']['convergence'] = True
                    break
            
            if nash_result['equilibrium_found']:
                # 记录均衡策略
                for i, tail in enumerate(candidate_tails):
                    nash_result['equilibrium_strategies'][tail] = {
                        'player_probability': float(player_probabilities[i]),
                        'banker_probability': float(banker_probabilities[i]),
                        'system_probability': float(system_probabilities[i])
                    }
                    
                    # 计算均衡收益
                    nash_result['equilibrium_payoffs'][tail] = {
                        'player': float(payoff_matrix[tail]['player_payoff'] * player_probabilities[i]),
                        'banker': float(payoff_matrix[tail]['banker_payoff'] * banker_probabilities[i]),
                        'system': float(payoff_matrix[tail]['system_payoff'] * system_probabilities[i])
                    }
                
                # 计算稳定性得分
                nash_result['stability_score'] = self._calculate_stability_score(
                    player_probabilities, banker_probabilities, payoff_matrix, candidate_tails
                )
            
            return nash_result
            
        except Exception as e:
            nash_result['computation_details']['error'] = str(e)
            return nash_result
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / np.sum(exp_x)
    
    def _calculate_stability_score(self, player_probs: np.ndarray, banker_probs: np.ndarray, 
                                 payoff_matrix: Dict, candidate_tails: List[int]) -> float:
        """计算纳什均衡的稳定性得分"""
        stability_factors = []
        
        # 因子1：策略分散度（越分散越稳定）
        player_entropy = -np.sum(player_probs * np.log(player_probs + 1e-10))
        banker_entropy = -np.sum(banker_probs * np.log(banker_probs + 1e-10))
        max_entropy = np.log(len(candidate_tails))
        
        if max_entropy > 0:
            avg_entropy = (player_entropy + banker_entropy) / (2 * max_entropy)
            stability_factors.append(avg_entropy)
        
        # 因子2：收益差异的均衡性
        player_payoffs = [payoff_matrix[tail]['player_payoff'] for tail in candidate_tails]
        banker_payoffs = [payoff_matrix[tail]['banker_payoff'] for tail in candidate_tails]
        
        player_payoff_var = np.var(player_payoffs)
        banker_payoff_var = np.var(banker_payoffs)
        
        # 方差越小，均衡越稳定
        avg_variance = (player_payoff_var + banker_payoff_var) / 2
        variance_stability = 1.0 / (1.0 + avg_variance * 10)  # 归一化
        stability_factors.append(variance_stability)
        
        # 因子3：博弈论经典稳定性指标
        # 计算每个参与者偏离均衡策略的损失
        deviation_costs = []
        for i, tail in enumerate(candidate_tails):
            # 玩家偏离成本
            current_payoff = payoff_matrix[tail]['player_payoff'] * player_probs[i]
            uniform_payoff = payoff_matrix[tail]['player_payoff'] / len(candidate_tails)
            player_deviation_cost = abs(current_payoff - uniform_payoff)
            
            # 庄家偏离成本
            current_banker_payoff = payoff_matrix[tail]['banker_payoff'] * banker_probs[i]
            uniform_banker_payoff = payoff_matrix[tail]['banker_payoff'] / len(candidate_tails)
            banker_deviation_cost = abs(current_banker_payoff - uniform_banker_payoff)
            
            deviation_costs.append((player_deviation_cost + banker_deviation_cost) / 2)
        
        if deviation_costs:
            avg_deviation_cost = np.mean(deviation_costs)
            deviation_stability = min(1.0, avg_deviation_cost)  # 偏离成本越高，越稳定
            stability_factors.append(deviation_stability)
        
        # 综合稳定性得分
        return np.mean(stability_factors) if stability_factors else 0.5
    
    def predict(self, candidate_tails: List[int], data_list: List[Dict]) -> Dict:
        """
        基于博弈论的预测主方法
        """
        try:
            prediction_start_time = datetime.now()
            
            # 1. 博弈状态分析
            game_analysis = self.analyze_game_state(candidate_tails, data_list)
            if not game_analysis['success']:
                return {
                    'success': False,
                    'recommended_tails': [],
                    'confidence': 0.0,
                    'reason': game_analysis['reason']
                }
            
            # 2. 计算纳什均衡
            nash_equilibrium = self.calculate_nash_equilibrium(
                candidate_tails, game_analysis['payoff_matrix']
            )
            
            # 3. 制定最优策略
            optimal_strategy = self._formulate_optimal_strategy(
                candidate_tails, game_analysis, nash_equilibrium
            )
            
            # 4. 选择推荐尾数
            recommendations = self._select_recommendations(
                candidate_tails, optimal_strategy, game_analysis
            )
            
            # 5. 计算整体置信度
            confidence = self._calculate_prediction_confidence(
                recommendations, optimal_strategy, nash_equilibrium
            )
            
            # 6. 更新博弈历史
            self._update_game_history(game_analysis, nash_equilibrium, recommendations)
            
            prediction_end_time = datetime.now()
            processing_time = (prediction_end_time - prediction_start_time).total_seconds()
            
            return {
                'success': True,
                'recommended_tails': recommendations.get('recommended_tails', []),
                'confidence': confidence,
                'strategy_type': optimal_strategy.get('strategy_type', 'balanced'),
                'nash_equilibrium': nash_equilibrium,
                'game_analysis': game_analysis,
                'optimal_strategy': optimal_strategy,
                'processing_time': processing_time,
                'prediction_reasoning': recommendations.get('reasoning', ''),
                'risk_assessment': optimal_strategy.get('risk_assessment', {}),
                'meta_info': {
                    'predictor': self.name,
                    'version': self.version,
                    'prediction_timestamp': prediction_start_time.isoformat()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'recommended_tails': [],
                'confidence': 0.0,
                'error': str(e),
                'reason': f'博弈论预测失败: {e}'
            }
    
    def _formulate_optimal_strategy(self, candidate_tails: List[int], game_analysis: Dict, 
                                  nash_equilibrium: Dict) -> Dict:
        """制定最优策略"""
        strategy = {
            'strategy_type': 'balanced',
            'risk_level': 'medium',
            'primary_targets': [],
            'secondary_targets': [],
            'avoid_targets': [],
            'strategy_weights': {},
            'risk_assessment': {}
        }
        
        # 基于纳什均衡制定策略
        if nash_equilibrium.get('equilibrium_found', False):
            equilibrium_strategies = nash_equilibrium['equilibrium_strategies']
            
            # 分析玩家最优策略
            player_probabilities = {tail: strategies['player_probability'] 
                                  for tail, strategies in equilibrium_strategies.items()}
            
            # 按玩家概率排序
            sorted_by_player_prob = sorted(player_probabilities.items(), 
                                         key=lambda x: x[1], reverse=True)
            
            # 分配目标等级
            total_tails = len(candidate_tails)
            primary_count = max(1, total_tails // 3)
            secondary_count = max(1, total_tails // 2)
            
            strategy['primary_targets'] = [tail for tail, _ in sorted_by_player_prob[:primary_count]]
            strategy['secondary_targets'] = [tail for tail, _ in sorted_by_player_prob[primary_count:secondary_count]]
            strategy['avoid_targets'] = [tail for tail, _ in sorted_by_player_prob[secondary_count:]]
            
            # 设置策略权重
            for tail in candidate_tails:
                player_prob = player_probabilities.get(tail, 0.0)
                banker_prob = equilibrium_strategies[tail]['banker_probability']
                
                # 综合权重 = 玩家概率 - 庄家反制概率
                combined_weight = player_prob - (banker_prob * 0.3)
                strategy['strategy_weights'][tail] = max(0.0, min(1.0, combined_weight))
        else:
            # 回退策略：基于收益矩阵
            payoff_matrix = game_analysis['payoff_matrix']
            
            tail_scores = {}
            for tail in candidate_tails:
                player_payoff = payoff_matrix[tail]['player_payoff']
                banker_payoff = payoff_matrix[tail]['banker_payoff']
                system_payoff = payoff_matrix[tail]['system_payoff']
                
                # 综合得分
                combined_score = (player_payoff * 0.5 + system_payoff * 0.3 - banker_payoff * 0.2)
                tail_scores[tail] = combined_score
            
            sorted_by_score = sorted(tail_scores.items(), key=lambda x: x[1], reverse=True)
            
            total_tails = len(candidate_tails)
            primary_count = max(1, total_tails // 3)
            
            strategy['primary_targets'] = [tail for tail, _ in sorted_by_score[:primary_count]]
            strategy['secondary_targets'] = [tail for tail, _ in sorted_by_score[primary_count:]]
            strategy['strategy_weights'] = {tail: score for tail, score in tail_scores.items()}
        
        # 确定策略类型和风险水平
        banker_strategy = game_analysis.get('banker_strategy', {})
        crowd_behavior = game_analysis.get('crowd_behavior', {})
        
        # 基于市场状态调整策略类型
        manipulation_strength = game_analysis.get('historical_patterns', {}).get('manipulation_strength', 0.5)
        herd_tendency = crowd_behavior.get('herd_tendency', 0.5)
        
        if manipulation_strength > 0.7:
            strategy['strategy_type'] = 'contrarian'  # 反向策略
            strategy['risk_level'] = 'high'
        elif herd_tendency > 0.8:
            strategy['strategy_type'] = 'anti_crowd'  # 反群体策略
            strategy['risk_level'] = 'medium'
        elif nash_equilibrium.get('stability_score', 0) > 0.7:
            strategy['strategy_type'] = 'equilibrium_based'  # 均衡策略
            strategy['risk_level'] = 'low'
        else:
            strategy['strategy_type'] = 'adaptive'  # 自适应策略
            strategy['risk_level'] = 'medium'
        
        # 风险评估
        strategy['risk_assessment'] = self._assess_strategy_risk(
            strategy, game_analysis, nash_equilibrium
        )
        
        return strategy
    
    def _assess_strategy_risk(self, strategy: Dict, game_analysis: Dict, 
                            nash_equilibrium: Dict) -> Dict:
        """评估策略风险"""
        risk_assessment = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'mitigation_suggestions': [],
            'confidence_intervals': {}
        }
        
        # 风险因子分析
        manipulation_strength = game_analysis.get('historical_patterns', {}).get('manipulation_strength', 0.5)
        if manipulation_strength > 0.7:
            risk_assessment['risk_factors'].append({
                'factor': '高操控风险',
                'severity': 'high',
                'description': f'检测到{manipulation_strength:.1%}的操控信号强度'
            })
            risk_assessment['mitigation_suggestions'].append('采用反操控策略，避开热门尾数')
        
        # 群体风险
        crowd_behavior = game_analysis.get('crowd_behavior', {})
        herd_tendency = crowd_behavior.get('herd_tendency', 0.5)
        if herd_tendency > 0.8:
            risk_assessment['risk_factors'].append({
                'factor': '强从众效应',
                'severity': 'medium',
                'description': f'群体表现出{herd_tendency:.1%}的从众倾向'
            })
            risk_assessment['mitigation_suggestions'].append('考虑反向操作，寻找冷门机会')
        
        # 纳什均衡稳定性风险
        stability_score = nash_equilibrium.get('stability_score', 0.5)
        if stability_score < 0.3:
            risk_assessment['risk_factors'].append({
                'factor': '均衡不稳定',
                'severity': 'medium',
                'description': f'纳什均衡稳定性仅为{stability_score:.1%}'
            })
            risk_assessment['mitigation_suggestions'].append('增加策略灵活性，准备快速调整')
        
        # 庄家反制风险
        banker_strategy = game_analysis.get('banker_strategy', {})
        kill_probabilities = banker_strategy.get('kill_probability', {})
        
        high_risk_targets = []
        for tail, kill_prob in kill_probabilities.items():
            if kill_prob > 0.6 and tail in strategy.get('primary_targets', []):
                high_risk_targets.append((tail, kill_prob))
        
        if high_risk_targets:
            risk_assessment['risk_factors'].append({
                'factor': '主要目标被杀风险',
                'severity': 'high',
                'description': f'{len(high_risk_targets)}个主要目标面临高被杀风险'
            })
            risk_assessment['mitigation_suggestions'].append('分散投资，增加备选目标')
        
        # 综合风险等级
        high_risk_count = sum(1 for factor in risk_assessment['risk_factors'] 
                             if factor['severity'] == 'high')
        medium_risk_count = sum(1 for factor in risk_assessment['risk_factors'] 
                               if factor['severity'] == 'medium')
        
        if high_risk_count >= 2:
            risk_assessment['overall_risk'] = 'high'
        elif high_risk_count >= 1 or medium_risk_count >= 3:
            risk_assessment['overall_risk'] = 'medium'
        else:
            risk_assessment['overall_risk'] = 'low'
        
        return risk_assessment
    
    def _select_recommendations(self, candidate_tails: List[int], optimal_strategy: Dict, 
                              game_analysis: Dict) -> Dict:
        """选择最终推荐"""
        recommendations = {
            'recommended_tails': [],
            'reasoning': '',
            'alternative_choices': [],
            'confidence_ranking': {}
        }
        
        strategy_weights = optimal_strategy.get('strategy_weights', {})
        strategy_type = optimal_strategy.get('strategy_type', 'balanced')
        
        if not strategy_weights:
            return recommendations
        
        # 按权重排序
        sorted_recommendations = sorted(strategy_weights.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        # 选择最佳推荐（通常选择权重最高的）
        if sorted_recommendations:
            best_tail = sorted_recommendations[0][0]
            best_weight = sorted_recommendations[0][1]
            
            recommendations['recommended_tails'] = [best_tail]
            recommendations['confidence_ranking'] = {
                tail: weight for tail, weight in sorted_recommendations
            }
            
            # 生成推理说明
            reasoning_parts = [
                f"基于{strategy_type}策略分析",
                f"推荐尾数{best_tail}（权重:{best_weight:.3f}）"
            ]
            
            # 添加具体推理依据
            nash_equilibrium = game_analysis.get('nash_equilibrium', {})
            if nash_equilibrium.get('equilibrium_found', False):
                equilibrium_strategies = nash_equilibrium['equilibrium_strategies']
                if best_tail in equilibrium_strategies:
                    player_prob = equilibrium_strategies[best_tail]['player_probability']
                    reasoning_parts.append(f"纳什均衡下玩家最优选择概率:{player_prob:.3f}")
            
            # 添加风险评估
            risk_assessment = optimal_strategy.get('risk_assessment', {})
            overall_risk = risk_assessment.get('overall_risk', 'medium')
            reasoning_parts.append(f"整体风险水平:{overall_risk}")
            
            # 添加反制策略说明
            banker_strategy = game_analysis.get('banker_strategy', {})
            counter_strategy = banker_strategy.get('counter_strategy', {})
            if best_tail in counter_strategy:
                counter_info = counter_strategy[best_tail]
                reasoning_parts.append(
                    f"反制策略:{counter_info['strategy_type']}"
                    f"(风险:{counter_info['risk_level']})"
                )
            
            recommendations['reasoning'] = " | ".join(reasoning_parts)
            
            # 提供备选方案
            if len(sorted_recommendations) > 1:
                alternatives = sorted_recommendations[1:3]  # 最多提供2个备选
                recommendations['alternative_choices'] = [
                    {
                        'tail': tail,
                        'weight': weight,
                        'rank': rank + 2  # 排名从2开始（第1是主推荐）
                    }
                    for rank, (tail, weight) in enumerate(alternatives)
                ]
        
        return recommendations
    
    def _calculate_prediction_confidence(self, recommendations: Dict, optimal_strategy: Dict, 
                                       nash_equilibrium: Dict) -> float:
        """计算预测置信度"""
        confidence_factors = []
        
        # 因子1：纳什均衡的稳定性
        stability_score = nash_equilibrium.get('stability_score', 0.5)
        confidence_factors.append(stability_score)
        
        # 因子2：策略权重的集中度
        strategy_weights = optimal_strategy.get('strategy_weights', {})
        if strategy_weights:
            weights = list(strategy_weights.values())
            max_weight = max(weights)
            weight_concentration = max_weight / sum(weights) if sum(weights) > 0 else 0
            confidence_factors.append(weight_concentration)
        
        # 因子3：推荐的一致性
        recommended_tails = recommendations.get('recommended_tails', [])
        if recommended_tails and strategy_weights:
            recommended_tail = recommended_tails[0]
            recommended_weight = strategy_weights.get(recommended_tail, 0)
            
            # 如果推荐的尾数确实是权重最高的，增加置信度
            is_top_choice = all(recommended_weight >= weight for weight in strategy_weights.values())
            if is_top_choice:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
        
        # 因子4：风险水平调整
        risk_level = optimal_strategy.get('risk_level', 'medium')
        risk_adjustment = {
            'low': 0.8,
            'medium': 0.6,
            'high': 0.4
        }
        confidence_factors.append(risk_adjustment.get(risk_level, 0.6))
        
        # 因子5：纳什均衡收敛性
        if nash_equilibrium.get('equilibrium_found', False):
            computation_details = nash_equilibrium.get('computation_details', {})
            if computation_details.get('convergence', False):
                iterations = computation_details.get('iterations', 1000)
                convergence_quality = max(0.3, 1.0 - iterations / 1000)  # 收敛越快质量越高
                confidence_factors.append(convergence_quality)
            else:
                confidence_factors.append(0.3)  # 未收敛的惩罚
        else:
            confidence_factors.append(0.4)  # 均衡求解失败的惩罚
        
        # 综合置信度
        if confidence_factors:
            base_confidence = np.mean(confidence_factors)
            
            # 应用策略类型调整
            strategy_type = optimal_strategy.get('strategy_type', 'balanced')
            strategy_adjustments = {
                'equilibrium_based': 1.1,
                'contrarian': 0.9,
                'anti_crowd': 0.95,
                'adaptive': 1.0,
                'balanced': 1.0
            }
            
            adjustment_factor = strategy_adjustments.get(strategy_type, 1.0)
            final_confidence = base_confidence * adjustment_factor
            
            return max(0.1, min(0.9, final_confidence))
        
        return 0.5
    
    def _update_game_history(self, game_analysis: Dict, nash_equilibrium: Dict, 
                           recommendations: Dict):
        """更新博弈历史记录"""
        history_record = {
            'timestamp': datetime.now().isoformat(),
            'game_state': {
                'manipulation_strength': game_analysis.get('historical_patterns', {}).get('manipulation_strength', 0.5),
                'crowd_herd_tendency': game_analysis.get('crowd_behavior', {}).get('herd_tendency', 0.5),
                'banker_kill_targets': game_analysis.get('banker_strategy', {}).get('optimal_kill_targets', [])
            },
            'nash_equilibrium': {
                'found': nash_equilibrium.get('equilibrium_found', False),
                'stability': nash_equilibrium.get('stability_score', 0.0)
            },
            'recommendation': {
                'tails': recommendations.get('recommended_tails', []),
                'reasoning': recommendations.get('reasoning', '')
            }
        }
        
        self.game_history.append(history_record)
        
        # 保持历史记录在合理范围内
        if len(self.game_history) > self.memory_depth:
            self.game_history = self.game_history[-self.memory_depth:]
    
    def learn_from_outcome(self, prediction_result: Dict, actual_tails: List[int]) -> Dict:
        """从实际结果中学习，更新策略有效性"""
        learning_result = {
            'learning_success': False,
            'strategy_updates': {},
            'performance_metrics': {}
        }
        
        try:
            if not prediction_result.get('success', False):
                return learning_result
            
            recommended_tails = prediction_result.get('recommended_tails', [])
            strategy_type = prediction_result.get('strategy_type', 'balanced')
            confidence = prediction_result.get('confidence', 0.0)
            
            # 判断预测是否成功
            prediction_success = any(tail in actual_tails for tail in recommended_tails)
            
            # 更新策略有效性统计
            if strategy_type not in self.strategy_effectiveness:
                self.strategy_effectiveness[strategy_type] = {
                    'total_predictions': 0,
                    'successful_predictions': 0,
                    'total_confidence': 0.0,
                    'recent_performance': []
                }
            
            strategy_stats = self.strategy_effectiveness[strategy_type]
            strategy_stats['total_predictions'] += 1
            strategy_stats['total_confidence'] += confidence
            
            if prediction_success:
                strategy_stats['successful_predictions'] += 1
            
            # 记录最近表现
            performance_record = {
                'success': prediction_success,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            strategy_stats['recent_performance'].append(performance_record)
            
            # 保持最近表现记录在合理范围
            if len(strategy_stats['recent_performance']) > 50:
                strategy_stats['recent_performance'] = strategy_stats['recent_performance'][-50:]
            
            # 计算性能指标
            total_preds = strategy_stats['total_predictions']
            successful_preds = strategy_stats['successful_predictions']
            
            learning_result['performance_metrics'] = {
                'strategy_type': strategy_type,
                'accuracy': successful_preds / total_preds if total_preds > 0 else 0.0,
                'average_confidence': strategy_stats['total_confidence'] / total_preds if total_preds > 0 else 0.0,
                'total_predictions': total_preds,
                'recent_accuracy': self._calculate_recent_accuracy(strategy_stats['recent_performance'])
            }
            
            # 自适应学习率调整
            if total_preds >= 10:  # 有足够数据后开始调整
                current_accuracy = successful_preds / total_preds
                recent_accuracy = self._calculate_recent_accuracy(strategy_stats['recent_performance'])
                
                # 如果最近表现下降，增加探索
                if recent_accuracy < current_accuracy - 0.1:
                    self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
                elif recent_accuracy > current_accuracy + 0.1:
                    self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
            
            # 更新博弈论参数
            self._update_game_theory_parameters(prediction_result, actual_tails, prediction_success)
            
            learning_result['learning_success'] = True
            learning_result['strategy_updates'] = {
                'exploration_rate': self.exploration_rate,
                'strategy_effectiveness': dict(self.strategy_effectiveness)
            }
            
            return learning_result
            
        except Exception as e:
            learning_result['error'] = str(e)
            return learning_result
    
    def _calculate_recent_accuracy(self, recent_performance: List[Dict]) -> float:
        """计算最近的准确率"""
        if not recent_performance:
            return 0.0
        
        # 只考虑最近20次预测
        recent_records = recent_performance[-20:]
        successful_count = sum(1 for record in recent_records if record.get('success', False))
        
        return successful_count / len(recent_records)
    
    def _update_game_theory_parameters(self, prediction_result: Dict, actual_tails: List[int], 
                                     prediction_success: bool):
        """更新博弈论参数"""
        # 更新系统随机性估计
        if prediction_success:
            # 预测成功，说明博弈论模型有效，随机性可能较低
            self.system_randomness = max(0.2, self.system_randomness - self.learning_rate)
        else:
            # 预测失败，可能是随机性较高或模型需要调整
            self.system_randomness = min(0.8, self.system_randomness + self.learning_rate)
        
        # 更新学习率（自适应）
        nash_equilibrium = prediction_result.get('nash_equilibrium', {})
        if nash_equilibrium.get('equilibrium_found', False):
            stability_score = nash_equilibrium.get('stability_score', 0.5)
            
            # 如果均衡稳定但预测失败，可能需要更大的学习率来探索
            if stability_score > 0.7 and not prediction_success:
                self.learning_rate = min(0.05, self.learning_rate * 1.2)
            elif stability_score < 0.3 and prediction_success:
                self.learning_rate = max(0.005, self.learning_rate * 0.8)
    
    def get_strategy_performance_summary(self) -> Dict:
        """获取策略表现总结"""
        summary = {
            'overall_performance': {},
            'strategy_breakdown': {},
            'learning_progress': {},
            'current_parameters': {}
        }
        
        if not self.strategy_effectiveness:
            return summary
        
        # 整体表现
        total_predictions = sum(stats['total_predictions'] for stats in self.strategy_effectiveness.values())
        total_successful = sum(stats['successful_predictions'] for stats in self.strategy_effectiveness.values())
        
        summary['overall_performance'] = {
            'total_predictions': total_predictions,
            'total_successful': total_successful,
            'overall_accuracy': total_successful / total_predictions if total_predictions > 0 else 0.0,
            'strategies_used': len(self.strategy_effectiveness)
        }
        
        # 各策略表现
        for strategy_type, stats in self.strategy_effectiveness.items():
            accuracy = stats['successful_predictions'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0.0
            avg_confidence = stats['total_confidence'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0.0
            recent_accuracy = self._calculate_recent_accuracy(stats['recent_performance'])
            
            summary['strategy_breakdown'][strategy_type] = {
                'accuracy': accuracy,
                'average_confidence': avg_confidence,
                'recent_accuracy': recent_accuracy,
                'total_use_count': stats['total_predictions'],
                'confidence_accuracy_ratio': accuracy / avg_confidence if avg_confidence > 0 else 0.0
            }
        
        # 学习进展
        summary['learning_progress'] = {
            'game_history_length': len(self.game_history),
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'system_randomness_estimate': self.system_randomness
        }
        
        # 当前参数
        summary['current_parameters'] = {
            'memory_depth': self.memory_depth,
            'nash_convergence_threshold': self.nash_convergence_threshold,
            'nash_max_iterations': self.nash_max_iterations,
            'available_strategies': list(self.strategy_types.keys())
        }
        
        return summary

# 测试代码
if __name__ == "__main__":
    print("🎮 测试博弈论策略器...")
    
    # 创建策略器实例
    strategist = GameTheoryStrategist()
    
    # 模拟测试数据
    test_data = [
        {'tails': [1, 3, 5, 7, 9], 'numbers': ['01', '13', '25', '37', '49', '02', '14']},
        {'tails': [0, 2, 4, 6, 8], 'numbers': ['10', '22', '34', '46', '08', '20', '32']},
        {'tails': [1, 2, 3, 4, 5], 'numbers': ['11', '12', '13', '14', '15', '26', '37']},
        {'tails': [6, 7, 8, 9, 0], 'numbers': ['16', '27', '38', '49', '10', '21', '32']},
        {'tails': [1, 4, 7, 2, 5], 'numbers': ['01', '14', '27', '32', '45', '18', '29']},
    ]
    
    test_candidates = [1, 2, 3, 4, 5]
    
    # 测试预测
    print("\n🔮 开始博弈论预测...")
    result = strategist.predict(test_candidates, test_data)
    
    if result['success']:
        print(f"✅ 预测成功!")
        print(f"📍 推荐尾数: {result['recommended_tails']}")
        print(f"🎯 置信度: {result['confidence']:.3f}")
        print(f"📊 策略类型: {result['strategy_type']}")
        print(f"⚖️ 纳什均衡: {'找到' if result['nash_equilibrium']['equilibrium_found'] else '未找到'}")
        print(f"🔍 推理过程: {result['prediction_reasoning']}")
        
        # 模拟学习过程
        print(f"\n📚 模拟学习过程...")
        actual_result = [1, 6, 8]  # 模拟实际开奖
        learning_result = strategist.learn_from_outcome(result, actual_result)
        
        if learning_result['learning_success']:
            print(f"✅ 学习成功!")
            metrics = learning_result['performance_metrics']
            print(f"📈 策略准确率: {metrics['accuracy']:.3f}")
            print(f"📊 平均置信度: {metrics['average_confidence']:.3f}")
            print(f"🔄 总预测次数: {metrics['total_predictions']}")
        
        # 显示策略表现总结
        print(f"\n📋 策略表现总结:")
        summary = strategist.get_strategy_performance_summary()
        overall = summary['overall_performance']
        print(f"   总预测: {overall.get('total_predictions', 0)}")
        print(f"   总成功: {overall.get('total_successful', 0)}")
        print(f"   整体准确率: {overall.get('overall_accuracy', 0):.3f}")
        
    else:
        print(f"❌ 预测失败: {result.get('reason', '未知错误')}")
    
    print(f"\n🎮 博弈论策略器测试完成!")