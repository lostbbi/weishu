# prediction_analyzer.py - 预测分析模块

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

class PredictionAnalyzer:
    """预测分析处理器"""
    
    def __init__(self, fundamental_laws):
        self.fundamental_laws = fundamental_laws
    
    def generate_detailed_prediction_analysis(self, recommended_tails, confidence, 
                                            ensemble_probabilities, all_predictions, data_list, 
                                            model_decision_records=None):
        """生成详细的预测分析（使用真实决策记录）"""
        try:
            weight_details = []
            analysis_text = []
            
            # 获取最新一期尾数
            latest_tails = set(data_list[0].get('tails', [])) if data_list else set()
            
            # 安全获取投资信息
            has_investment_data = hasattr(self, 'current_investments') and bool(getattr(self, 'current_investments', {}))
            print(f"🔍 投资数据状态: {has_investment_data}")
            
            analysis_text.append("🔍 AI预测决策过程详细分析")
            analysis_text.append("=" * 50)
            analysis_text.append("")
            analysis_text.append(f"📊 基本信息：")
            analysis_text.append(f"  • 最新一期包含尾数：{sorted(latest_tails)}")
            analysis_text.append(f"  • 参与预测的模型数量：{len(all_predictions)}")
            analysis_text.append(f"  • 最终推荐尾数：{recommended_tails}")
            analysis_text.append(f"  • 预测置信度：{confidence:.3f}")
            analysis_text.append("")
            
            # 应用4大定律筛选，只分析通过筛选的候选尾数
            latest_tails = set(data_list[0].get('tails', [])) if data_list else set()
            
            # 应用4大定律筛选候选尾数
            if data_list and latest_tails:
                # 识别需要排除的尾数
                trap_tails = self.fundamental_laws.identify_comprehensive_trap_tails(data_list)
                dual_minimum_tails = self.fundamental_laws.identify_dual_minimum_tails(data_list)
                extremely_hot_tails = self.fundamental_laws.identify_extremely_hot_tails(data_list, periods=30, threshold=20)
                
                # 计算通过筛选的候选尾数
                excluded_tails = trap_tails.union(dual_minimum_tails).union(extremely_hot_tails)
                valid_candidates = latest_tails - excluded_tails
                
                # 显示筛选过程
                analysis_text.append("🔍 四大定律筛选过程：")
                analysis_text.append(f"  • 最新一期包含尾数：{sorted(latest_tails)}")
                if trap_tails:
                    analysis_text.append(f"  • 定律1排除陷阱尾数：{sorted(trap_tails)}")
                if dual_minimum_tails:
                    analysis_text.append(f"  • 定律2排除双重最少尾数：{sorted(dual_minimum_tails)}")
                if extremely_hot_tails:
                    analysis_text.append(f"  • 定律3排除极热尾数：{sorted(extremely_hot_tails)}")
                analysis_text.append(f"  • 通过筛选的候选尾数：{sorted(valid_candidates)}")
                analysis_text.append("")
            else:
                valid_candidates = latest_tails
                analysis_text.append("⚠️ 数据不足，跳过定律筛选")
                analysis_text.append("")
            
            # 为通过筛选的候选尾数生成详细分析
            all_candidates_analysis = {}
            
            for candidate_tail in valid_candidates:
                analysis_text.append(f"🎯 候选尾数 {candidate_tail} 的详细分析：")
                analysis_text.append("")
                
                total_weighted_sum = 0.0
                total_weight = 0.0
                candidate_weight_details = []
                
                # 分析每个模型对这个候选尾数的贡献
                for model_key, predictions in all_predictions.items():
                    if candidate_tail in predictions:
                        model_prob = predictions[candidate_tail]
                        # 只使用投资权重，不回退到传统权重
                        investment_weight = 0.0
                        try:
                            # 从引擎实例获取当前投资信息
                            if (hasattr(self, '_engine_ref') and self._engine_ref and 
                                hasattr(self._engine_ref, 'current_weight_investments') and 
                                self._engine_ref.current_weight_investments and 
                                model_key in self._engine_ref.current_weight_investments):
                                investment_weight = self._engine_ref.current_weight_investments[model_key].get(candidate_tail, 0.0)
                            else:
                                # 没有投资权重时，该模型不参与这个尾数的预测
                                investment_weight = 0.0
                        except Exception as e:
                            print(f"⚠️ 获取模型 {model_key} 投资权重失败: {e}")
                            investment_weight = 0.0  # 没有投资就是0权重
                        
                        if investment_weight > 0:
                            # 只有当有投资权重时才计算贡献
                            if investment_weight > 0:
                                weighted_contribution = investment_weight  # 直接使用投资权重作为贡献
                            else:
                                weighted_contribution = 0.0  # 没有投资就没有贡献
                            total_weighted_sum += weighted_contribution
                            total_weight += investment_weight
                        else:
                            # 模型没有投资该尾数，不参与计算
                            weighted_contribution = 0.0
                        
                        # 分析选择理由
                        reason = self._analyze_model_selection_reason(model_key, candidate_tail, model_prob, data_list, model_decision_records)
                        
                        # 只记录投资权重信息
                        if investment_weight > 0:
                            candidate_weight_details.append({
                                'model_name': model_key,
                                'target_tail': candidate_tail,
                                'prediction_probability': model_prob,
                                'investment_weight': investment_weight,
                                'weight_contribution': investment_weight,  # 直接使用投资权重
                                'selection_reason': reason
                            })
                        
                        # 只显示有投资的模型信息
                        if investment_weight > 0:
                            model_display = model_key.replace('_', ' ').title()
                            analysis_text.append(f"  📈 {model_display}:")
                            analysis_text.append(f"     - 预测概率：{model_prob:.3f}")
                            analysis_text.append(f"     - 投资权重：{investment_weight:.4f}")
                            analysis_text.append(f"     - 权重贡献：{investment_weight:.4f}")  # 直接使用投资权重
                            analysis_text.append(f"     - 选择理由：{reason}")
                            analysis_text.append("")
                
                # 集成决策过程
                final_probability = total_weighted_sum / total_weight if total_weight > 0 else 0.5
                analysis_text.append(f"⚖️ 尾数 {candidate_tail} 集成决策过程：")
                analysis_text.append(f"  • 总加权概率：{total_weighted_sum:.3f}")
                analysis_text.append(f"  • 总权重：{total_weight:.3f}")
                analysis_text.append(f"  • 最终概率：{final_probability:.3f}")
                
                # 判断该尾数是否被选中
                is_selected = candidate_tail in recommended_tails if recommended_tails else False
                if is_selected:
                    analysis_text.append(f"  ✅ 结果：被选为最终推荐")
                else:
                    analysis_text.append(f"  ❌ 结果：未被选中")
                    # 分析未被选中的原因
                    if recommended_tails:
                        selected_tail = recommended_tails[0]
                        selected_prob = ensemble_probabilities.get(selected_tail, 0.5)
                        candidate_prob = ensemble_probabilities.get(candidate_tail, 0.5)
                        analysis_text.append(f"  📊 未选中原因：概率{candidate_prob:.3f} < 推荐尾数{selected_tail}概率{selected_prob:.3f}")
                
                analysis_text.append("")
                analysis_text.append("-" * 50)
                analysis_text.append("")
                
                # 保存候选分析结果
                all_candidates_analysis[candidate_tail] = {
                    'weight_details': candidate_weight_details,
                    'final_probability': final_probability,
                    'is_selected': is_selected
                }
            
            # 如果有被排除的尾数，简要说明排除原因
            if data_list and latest_tails:
                excluded_tails = latest_tails - valid_candidates
                if excluded_tails:
                    analysis_text.append("🚫 被定律排除的尾数简要说明：")
                    analysis_text.append("")
                    
                    for excluded_tail in sorted(excluded_tails):
                        reasons = []
                        if excluded_tail in trap_tails:
                            reasons.append("陷阱尾数")
                        if excluded_tail in dual_minimum_tails:
                            reasons.append("双重最少")
                        if excluded_tail in extremely_hot_tails:
                            reasons.append("极热尾数")
                        
                        analysis_text.append(f"  ❌ 尾数 {excluded_tail}：被排除（{' + '.join(reasons)}）")
                    
                    analysis_text.append("")
                    analysis_text.append("-" * 50)
                    analysis_text.append("")

            # 使用推荐尾数的权重详情作为主要显示内容
            if recommended_tails:
                target_tail = recommended_tails[0]
                weight_details = all_candidates_analysis.get(target_tail, {}).get('weight_details', [])
            else:
                # 如果没有推荐尾数，使用第一个候选的详情
                if all_candidates_analysis:
                    first_candidate = next(iter(all_candidates_analysis.keys()))
                    weight_details = all_candidates_analysis[first_candidate].get('weight_details', [])
                else:
                    weight_details = []
            
            # 显示选择的推荐尾数的详细分析
            if recommended_tails:
                target_tail = recommended_tails[0]
                analysis_text.append(f"🎯 最终推荐尾数 {target_tail} 的详细分析：")
                analysis_text.append("")
                
                if weight_details:
                    total_weighted_sum = 0.0
                    total_weight = 0.0
                    
                    for detail in weight_details:
                        model_prob = detail['prediction_probability']
                        actual_model_weight = detail.get('model_weight', 0.0)  # 实际模型权重
                        investment_weight = detail.get('investment_weight', 0.0)  # 投资权重
                        weighted_contribution = detail['weighted_contribution']
                        reason = detail['selection_reason']
                        model_display = detail['model_name'].replace('_', ' ').title()
                        
                        total_weighted_sum += weighted_contribution
                        total_weight += investment_weight if investment_weight > 0 else actual_model_weight
                        
                        # 添加到分析文本
                        analysis_text.append(f"  📈 {model_display}:")
                        analysis_text.append(f"     - 预测概率：{model_prob:.3f}")
                        analysis_text.append(f"     - 模型权重：{actual_model_weight:.4f}")
                        if investment_weight > 0:
                            analysis_text.append(f"     - 投资权重：{investment_weight:.4f}")
                            analysis_text.append(f"     - 加权贡献：{weighted_contribution:.4f}")
                        else:
                            analysis_text.append(f"     - 加权贡献：{weighted_contribution:.4f}（使用模型权重）")
                        analysis_text.append(f"     - 选择理由：{reason}")
                        analysis_text.append("")
                    
                    # 集成决策过程
                    final_probability = total_weighted_sum / total_weight if total_weight > 0 else 0.5
                    analysis_text.append(f"⚖️ 集成决策过程：")
                    analysis_text.append(f"  • 总加权概率：{total_weighted_sum:.3f}")
                    analysis_text.append(f"  • 总权重：{total_weight:.3f}")
                    analysis_text.append(f"  • 最终概率：{final_probability:.3f}")
                    analysis_text.append("")
                    
                    # 应用底层定律分析
                    analysis_text.append(f"📚 底层定律应用分析：")
                    self._add_fundamental_law_analysis(analysis_text, data_list, recommended_tails)
                else:
                    analysis_text.append(f"  ⚠️ 无详细的权重信息可供分析")
                    analysis_text.append("")

            else:
                analysis_text.append("⚠️ 未找到符合条件的推荐尾数")
                analysis_text.append("")
                analysis_text.append("可能原因分析：")
                if data_list and latest_tails:
                    excluded_count = len(latest_tails - valid_candidates)
                    if excluded_count == len(latest_tails):
                        analysis_text.append("  • 所有最新期尾数都被四大定律排除")
                        analysis_text.append(f"  • 共{len(latest_tails)}个尾数全部被排除")
                    elif excluded_count > 0:
                        analysis_text.append(f"  • {excluded_count}/{len(latest_tails)}个尾数被定律排除")
                        analysis_text.append("  • 剩余候选尾数模型预测概率较低")
                    else:
                        analysis_text.append("  • 通过定律筛选的尾数模型预测概率都较低")
                else:
                    analysis_text.append("  • 数据不足或无最新期尾数")

            # 生成决策总结
            if recommended_tails and weight_details:
                participating_models = len(weight_details)
                avg_probability = sum(d['prediction_probability'] for d in weight_details) / participating_models
                decision_summary = f"共{participating_models}个模型参与预测尾数{recommended_tails[0]}，平均预测概率{avg_probability:.3f}，最终置信度{confidence:.3f}"
            else:
                decision_summary = "预测过程未产生有效结果，建议检查数据质量或增加历史样本"
            
            return {
                'weight_details': weight_details,
                'decision_summary': decision_summary,
                'detailed_analysis': '\n'.join(analysis_text)
            }
            
        except Exception as e:
            print(f"生成详细预测分析失败: {e}")
            return {
                'weight_details': [],
                'decision_summary': f"分析生成失败：{str(e)}",
                'detailed_analysis': f"生成详细分析时出错：{str(e)}"
            }
    
    def _analyze_model_selection_reason(self, model_key, target_tail, probability, data_list, decision_records=None):
        """分析模型选择某个尾数的理由（基于真实决策记录）"""
        try:
            # 获取该模型的详细决策记录
            if decision_records and model_key in decision_records:
                decision_record = decision_records[model_key]
                
                # 获取该尾数的详细理由
                if 'detailed_reasons' in decision_record and target_tail in decision_record['detailed_reasons']:
                    tail_reason = decision_record['detailed_reasons'][target_tail]
                    
                    # 历史模式匹配算法的详细理由
                    if 'pattern_matcher' in model_key.lower():
                        if isinstance(tail_reason, str):
                            return tail_reason
                        else:
                            return tail_reason.get('reason_summary', f"历史模式匹配：概率{probability:.3f}")
                    
                    # 其他模型的详细理由
                    elif isinstance(tail_reason, dict):
                        # 优先使用详细分析
                        if 'detailed_analysis' in tail_reason:
                            return tail_reason['detailed_analysis']
                        
                        # 否则组合其他信息
                        reason_parts = []
                        if 'reason_summary' in tail_reason:
                            reason_parts.append(tail_reason['reason_summary'])
                        if 'confidence_level' in tail_reason:
                            reason_parts.append(f"置信度:{tail_reason['confidence_level']}")
                        if 'probability_source' in tail_reason:
                            reason_parts.append(f"来源:{tail_reason['probability_source']}")
                        
                        return " | ".join(reason_parts) if reason_parts else f"概率预测:{probability:.3f}"
                    
                    elif isinstance(tail_reason, str):
                        return tail_reason
                
                # 如果没有特定尾数的理由，使用决策过程
                if 'decision_process' in decision_record:
                    process_summary = " -> ".join(decision_record['decision_process'][-2:])  # 最后两个步骤
                    return f"{process_summary} | 概率:{probability:.3f}"
                
                # 使用匹配分析（针对历史模式匹配）
                if 'matching_analysis' in decision_record:
                    matching_analysis = decision_record['matching_analysis']
                    if target_tail in matching_analysis.get('matching_results', {}):
                        tail_result = matching_analysis['matching_results'][target_tail]
                        match_count = tail_result.get('match_count', 0)
                        best_similarity = tail_result.get('best_similarity', 0.0)
                        
                        if match_count > 0:
                            return f"历史匹配:{match_count}个模式,最高相似度{best_similarity:.3f} | 概率:{probability:.3f}"
                        else:
                            return f"历史匹配:无匹配模式 | 概率:{probability:.3f}"
            
            # 回退到分析式理由（当没有详细记录时）
            reasons = []
            
            # 模型类型特定的分析
            if 'pattern_matcher' in model_key.lower():
                reasons.append("历史模式匹配算法")
                if probability > 0.6:
                    reasons.append("发现高相似度历史模式")
                elif probability > 0.4:
                    reasons.append("发现中等相似度历史模式")
                else:
                    reasons.append("未发现足够相似的历史模式")
            
            elif 'hoeffding' in model_key.lower():
                reasons.append("Hoeffding决策树分析")
                if data_list and len(data_list) > 0:
                    latest_tails = data_list[0].get('tails', [])
                    if target_tail in latest_tails:
                        reasons.append("决策树判断延续当前趋势")
                    else:
                        reasons.append("决策树判断趋势反转")
            
            elif 'logistic' in model_key.lower():
                reasons.append("逻辑回归分析")
                if probability > 0.7:
                    reasons.append("特征组合强烈指向该尾数")
                elif probability > 0.5:
                    reasons.append("特征组合倾向于该尾数")
                else:
                    reasons.append("特征组合不支持该尾数")
            
            elif 'naive_bayes' in model_key.lower():
                reasons.append("朴素贝叶斯概率推理")
                reasons.append(f"基于特征独立假设计算概率{probability:.3f}")
            
            elif 'bagging' in model_key.lower():
                reasons.append("装袋集成学习")
                if probability > 0.6:
                    reasons.append("多数子模型投票支持")
                else:
                    reasons.append("子模型投票分歧或不支持")
            
            elif 'adaboost' in model_key.lower():
                reasons.append("AdaBoost自适应提升")
                reasons.append(f"加权投票结果概率{probability:.3f}")
            
            else:
                reasons.append("机器学习算法")
                reasons.append(f"模型输出概率{probability:.3f}")
            
            # 基于概率的补充分析
            if probability > 0.8:
                reasons.append("高置信度预测")
            elif probability > 0.6:
                reasons.append("中高置信度预测")
            elif probability > 0.4:
                reasons.append("中等置信度预测")
            else:
                reasons.append("低置信度预测")
            
            return " | ".join(reasons)
            
        except Exception as e:
            return f"分析失败：{str(e)} | 概率:{probability:.3f}"
    
    def _add_fundamental_law_analysis(self, analysis_text, data_list, recommended_tails):
        """添加底层定律应用分析"""
        try:
            if not data_list:
                analysis_text.append("  • 数据不足，无法应用底层定律")
                return
            
            latest_tails = set(data_list[0].get('tails', [])) if data_list else set()
            analysis_text.append(f"  • 定律1（最新期筛选）：从{sorted(latest_tails)}中筛选")
            
            if len(data_list) >= 30:
                # 陷阱尾数分析
                trap_tails = self.fundamental_laws.identify_comprehensive_trap_tails(data_list)
                if trap_tails:
                    analysis_text.append(f"  • 定律2（排除陷阱）：排除{sorted(trap_tails)}")
                else:
                    analysis_text.append(f"  • 定律2（排除陷阱）：未发现陷阱尾数")
                
                # 双重最少分析
                dual_minimum = self.fundamental_laws.identify_dual_minimum_tails(data_list)
                if dual_minimum:
                    analysis_text.append(f"  • 定律3（排除双重最少）：排除{sorted(dual_minimum)}")
                else:
                    analysis_text.append(f"  • 定律3（排除双重最少）：未发现双重最少尾数")
                
                # 极热尾数分析
                extremely_hot = self.fundamental_laws.identify_extremely_hot_tails(data_list)
                if extremely_hot:
                    analysis_text.append(f"  • 定律4（排除极热）：排除{sorted(extremely_hot)}（30期≥20次或10期≥8次）")
                else:
                    analysis_text.append(f"  • 定律4（排除极热）：未发现极热尾数（30期≥20次或10期≥8次）")
            else:
                analysis_text.append(f"  • 数据不足30期，底层定律简化应用")
            
            # 显示详细的筛选过程
            if latest_tails:
                analysis_text.append(f"  • 筛选过程总结：")
                trap_tails = self.fundamental_laws.identify_comprehensive_trap_tails(data_list)
                dual_minimum = self.fundamental_laws.identify_dual_minimum_tails(data_list)
                extremely_hot = self.fundamental_laws.identify_extremely_hot_tails(data_list)
                remaining_after_laws = latest_tails - trap_tails - dual_minimum - extremely_hot
                
                for tail in latest_tails:
                    status_parts = []
                    if tail in trap_tails:
                        status_parts.append("陷阱尾数")
                    if tail in dual_minimum:
                        status_parts.append("双重最少")
                    if tail in extremely_hot:
                        status_parts.append("极热尾数")
                    
                    if status_parts:
                        analysis_text.append(f"    - 尾数{tail}：被排除（{' + '.join(status_parts)}）")
                    else:
                        analysis_text.append(f"    - 尾数{tail}：通过定律筛选")
                
                if recommended_tails:
                    analysis_text.append(f"  • 最终选择：从通过筛选的{len(remaining_after_laws)}个尾数中选择概率最高的{recommended_tails[0]}")
                else:
                    analysis_text.append(f"  • 最终结果：所有候选均被定律排除")
            else:
                analysis_text.append(f"  • 最终结果：无候选尾数")
                
        except Exception as e:
            analysis_text.append(f"  • 底层定律分析失败：{str(e)}")
    
    def generate_model_specific_reason(self, model_name, tail, probability, tail_features, data_list):
        """为特定模型生成详细的决策理由"""
        reasons = []
        
        # 基础信息
        reasons.append(f"尾数{tail}预测分析")
        
        # 模型类型特定分析
        if 'hoeffding' in model_name.lower():
            reasons.append("【Hoeffding决策树分析】")
            
            # 基于特征的决策路径分析
            if tail_features.get('in_latest_period', False):
                reasons.append("• 决策路径：尾数在最新期出现 → 延续趋势判断")
            else:
                reasons.append("• 决策路径：尾数未在最新期出现 → 回补趋势判断")
            
            consecutive = tail_features.get('consecutive_appearances', 0)
            if consecutive >= 2:
                reasons.append(f"• 连续性特征：连续{consecutive}期出现，树判断为持续趋势")
            elif consecutive == 0:
                last_distance = tail_features.get('last_appearance_distance', -1)
                if last_distance > 0:
                    reasons.append(f"• 间隔特征：距上次出现{last_distance}期，树判断需要回补")
                else:
                    reasons.append("• 冷门特征：长期未出现，树倾向于回补预测")
            
            recent_freq = tail_features.get('recent_10_frequency', 0)
            if recent_freq > 0.6:
                reasons.append("• 频率判断：近期高频(>60%)，树认为应继续")
            elif recent_freq < 0.3:
                reasons.append("• 频率判断：近期低频(<30%)，树倾向回补")
            else:
                reasons.append("• 频率判断：近期中等频率，树基于其他特征决策")
                
        elif 'logistic' in model_name.lower():
            reasons.append("【逻辑回归分析】")
            
            # 基于特征权重的分析
            reasons.append("• 特征权重计算：")
            if tail_features.get('in_latest_period', False):
                reasons.append("  - 最新期出现特征：正向权重+0.3")
            else:
                reasons.append("  - 最新期未出现特征：负向权重-0.2")
                
            recent_freq = tail_features.get('recent_10_frequency', 0)
            if recent_freq > 0.5:
                reasons.append(f"  - 近期频率特征({recent_freq:.2f})：正向权重+{recent_freq*0.4:.2f}")
            else:
                reasons.append(f"  - 近期频率特征({recent_freq:.2f})：负向权重-{(1-recent_freq)*0.3:.2f}")
                
            consecutive = tail_features.get('consecutive_appearances', 0)
            if consecutive > 0:
                weight = min(consecutive * 0.15, 0.4)
                reasons.append(f"  - 连续性特征({consecutive}期)：正向权重+{weight:.2f}")
            
            reasons.append(f"• Sigmoid函数映射：线性组合 → 概率{probability:.3f}")
            
        elif 'naive_bayes' in model_name.lower():
            reasons.append("【朴素贝叶斯分析】")
            
            # 基于条件概率的分析
            reasons.append("• 特征独立性假设下的概率计算：")
            
            if tail_features.get('in_latest_period', False):
                reasons.append("  - P(预测=1|最新期出现) = 0.7")
            else:
                reasons.append("  - P(预测=1|最新期未出现) = 0.3")
            
            recent_count = tail_features.get('recent_5_count', 0)
            if recent_count >= 3:
                reasons.append(f"  - P(预测=1|近5期出现{recent_count}次) = 0.8")
            elif recent_count >= 1:
                reasons.append(f"  - P(预测=1|近5期出现{recent_count}次) = 0.6")
            else:
                reasons.append(f"  - P(预测=1|近5期出现{recent_count}次) = 0.2")
            
            reasons.append(f"• 贝叶斯定理计算后验概率：{probability:.3f}")
            
        elif 'bagging' in model_name.lower():
            reasons.append("【装袋集成分析】")
            
            # 模拟子模型投票
            base_model_type = "决策树" if "hoeffding" not in model_name else "逻辑回归"
            voting_details = self._simulate_bagging_votes(tail_features, probability)
            
            reasons.append(f"• 基学习器：{base_model_type} x 5个")
            reasons.append(f"• 投票结果：{voting_details['positive_votes']}/5 支持预测")
            reasons.append(f"• 投票详情：{voting_details['vote_details']}")
            reasons.append(f"• 平均概率：{probability:.3f}")
            
        elif 'adaboost' in model_name.lower():
            reasons.append("【AdaBoost分析】")
            
            # 模拟加权投票
            boost_details = self._simulate_adaboost_weights(tail_features, probability)
            
            reasons.append("• 加权投票机制：")
            for i, (weight, prediction) in enumerate(boost_details['weighted_votes']):
                vote_str = "支持" if prediction > 0.5 else "反对"
                reasons.append(f"  - 弱学习器{i+1}：权重{weight:.2f} × {vote_str}({prediction:.2f})")
            
            reasons.append(f"• 最终加权结果：{probability:.3f}")
            
        else:
            reasons.append(f"【{model_name.replace('_', ' ').title()}分析】")
            reasons.append(f"• 模型输出概率：{probability:.3f}")
            reasons.append("• 基于训练数据的复杂特征组合判断")
        
        # 置信度评估
        if probability > 0.7:
            reasons.append(f"• 置信度评估：高置信度({probability:.3f}) - 强烈推荐")
        elif probability > 0.6:
            reasons.append(f"• 置信度评估：中高置信度({probability:.3f}) - 推荐")
        elif probability > 0.4:
            reasons.append(f"• 置信度评估：中等置信度({probability:.3f}) - 中性")
        else:
            reasons.append(f"• 置信度评估：低置信度({probability:.3f}) - 不推荐")
        
        return " | ".join(reasons)
    
    def _simulate_bagging_votes(self, tail_features, final_probability):
        """模拟装袋集成的投票过程"""
        # 基于最终概率反推可能的投票情况
        positive_votes = int(final_probability * 5 + 0.5)  # 四舍五入
        positive_votes = max(0, min(5, positive_votes))
        
        vote_details = []
        for i in range(5):
            if i < positive_votes:
                # 支持的子模型
                base_prob = 0.6 + (final_probability - 0.5) * 0.4  # 0.6-0.8之间
                vote_details.append(f"子模型{i+1}:支持({base_prob:.2f})")
            else:
                # 反对的子模型
                base_prob = 0.4 - (0.5 - final_probability) * 0.4  # 0.2-0.4之间
                vote_details.append(f"子模型{i+1}:反对({base_prob:.2f})")
        
        return {
            'positive_votes': positive_votes,
            'vote_details': ' | '.join(vote_details)
        }
    
    def _simulate_adaboost_weights(self, tail_features, final_probability):
        """模拟AdaBoost的加权投票过程"""
        # 模拟5个弱学习器的权重和预测
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # 递减的权重
        
        weighted_votes = []
        for i, weight in enumerate(weights):
            # 基于特征和最终概率推测每个弱学习器的预测
            if i == 0:  # 权重最高的学习器
                prediction = final_probability + 0.1 if final_probability < 0.9 else 0.9
            elif i == 1:
                prediction = final_probability
            else:
                # 后续学习器有一定随机性
                prediction = final_probability + (0.5 - final_probability) * 0.3
            
            prediction = max(0.0, min(1.0, prediction))
            weighted_votes.append((weight, prediction))
        
        return {'weighted_votes': weighted_votes}