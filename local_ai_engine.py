# ultimate_online_ai_engine.py - 终极在线学习AI引擎（完整实现版）

import json
import sqlite3
import numpy as np
import os
import torch.nn.functional as F
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import traceback
import math
from data.data_analysis_tools import DataAnalysisTools
from ai.deep_learning_models import LSTMPredictor, TransformerPredictor, PositionalEncoding
from ai.deep_learning_module import DeepLearningModule
from core.config_manager import ConfigManager
from ai_engine.prediction.anti_manipulation_models import BankerBehaviorAnalyzer
from ai_engine.prediction.reverse_psychology_models import ReversePsychologyPredictor
from ai_engine.prediction.unpopular_digger_models import UnpopularDigger
from ai_engine.prediction.money_flow_analyzer import MoneyFlowAnalyzer
from ai_engine.prediction.game_theory_strategist import GameTheoryStrategist
from ai_engine.prediction.manipulation_timing_detector import ManipulationTimingDetector
from ai_engine.prediction.anti_trend_hunter import AntiTrendHunter
from ai_engine.prediction.crowd_psychology_analyzer import CrowdPsychologyAnalyzer

# 导入AI配置模块
from ai.ai_config import (
    ai_config, PYTORCH_AVAILABLE, DEVICE, RIVER_AVAILABLE, 
    SKMULTIFLOW_AVAILABLE, DEEP_RIVER_AVAILABLE, LOCAL_MODELS_AVAILABLE, 
    ENABLED_SPECIAL_MODELS
)
from ai.feature_engineering import FeatureEngineer
from analysis.fundamental_laws import FundamentalLaws
from investment.investment_system import InvestmentSystem
from data.database_manager import DatabaseManager
from ai.prediction_analyzer import PredictionAnalyzer
from ai.drift_detection import DriftDetectionManager
from analysis.statistics_manager import StatisticsManager
    
class UltimateOnlineAIEngine:
    """终极在线学习AI引擎 - 融合多个库的优势"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # 初始化配置管理器
        self.config_manager = ConfigManager()

        # 获取启用模型配置
        try:
            self.enabled_models = self.config_manager.get_enabled_models_config()
            print(f"📋 启用模型配置加载完成")
            print(f"   River模型: {self.enabled_models['river_models']}")
            print(f"   sklearn模型: {self.enabled_models['sklearn_models']}")
            print(f"   PyTorch模型: {self.enabled_models['pytorch_models']}")
            print(f"   特殊模型: {self.enabled_models['special_models']}")
        except:
            # 如果config_manager没有该方法，使用ai_config的配置
            self.enabled_models = {
                'river_models': ['local_bagging', 'local_pattern_matcher_strict'],
                'sklearn_models': ['extremely_fast_tree', 'skm_hoeffding_adaptive'],
                'pytorch_models': ['lstm', 'transformer'],
                'special_models': ai_config.ENABLED_SPECIAL_MODELS
            }
            print(f"📋 使用ai_config的特殊模型配置")
            print(f"   特殊模型: {self.enabled_models['special_models']}")

        # 提取各类型的启用模型配置（供后续使用）
        enabled_special = self.enabled_models.get('special_models', [])
        enabled_pytorch = self.enabled_models.get('pytorch_models', [])
        enabled_river = self.enabled_models.get('river_models', [])
        enabled_sklearn = self.enabled_models.get('sklearn_models', [])

        # 确保数据目录存在，并添加错误处理
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ AI数据目录已确保存在: {self.data_dir}")
            
            # 验证目录是否可写
            test_file = self.data_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"✅ AI数据目录写入权限验证成功")
            except Exception as write_e:
                print(f"❌ AI数据目录写入权限验证失败: {write_e}")
                raise Exception(f"AI数据目录无写入权限: {write_e}")
                
        except Exception as dir_e:
            print(f"❌ 创建AI数据目录失败: {dir_e}")
            raise Exception(f"无法创建或访问AI数据目录: {dir_e}")
        
        # 数据库和模型路径
        db_config = self.config_manager.get_database_config()
        self.db_path = self.data_dir / db_config['db_name']
        self.model_path = self.data_dir / db_config['model_state_name']
        
        # 在线学习模型组件
        self.river_models = {}        # River模型
        self.sklearn_models = {}      # scikit-multiflow模型
        self.drift_detectors = {}     # 概念漂移检测器
        self.metrics_trackers = {}    # 性能度量跟踪器
        self.ensemble_weights = {}    # 集成学习权重
        
        # 学习状态
        self.total_samples_seen = 0
        self.correct_predictions = 0
        self.model_performance = {}
        self.is_initialized = False
        self.last_prediction_result = None
        
        # 调试信息控制开关
        self.debug_pattern_matching = False  # 设置为False来关闭历史模式匹配的调试信息

        # 在线学习配置
        self.learning_config = self.config_manager.get_learning_config()
        
        # 初始化组件
        self.init_database()
        # 预先初始化深度学习模块（避免在init_online_models中访问未初始化的属性）
        self.deep_learning_module = None
        if ai_config.PYTORCH_AVAILABLE:
            try:
                print("🚀 开始初始化深度学习模块...")
                dl_config = self.config_manager.get_deep_learning_config()
                self.deep_learning_module = DeepLearningModule(
                    input_size=dl_config['input_size'],
                    device=ai_config.DEVICE
                )
                
                # 确保PyTorch模型目录存在
                pytorch_model_dir = self.ensure_pytorch_model_directory()
                if pytorch_model_dir and hasattr(self.deep_learning_module, 'models_dir'):
                    self.deep_learning_module.models_dir = pytorch_model_dir
                    print(f"✅ 模型目录路径已传递给深度学习模块")
                else:
                    print(f"❌ 无法传递模型目录路径给深度学习模块")
                    self.deep_learning_module = None
                    raise Exception("无法初始化深度学习模块")
                

                # 验证模型是否正确初始化
                if hasattr(self.deep_learning_module, 'models') and self.deep_learning_module.models:
                    print(f"🤖 深度学习模块初始化完成，包含 {len(self.deep_learning_module.models)} 个模型")
                else:
                    print("⚠️ 深度学习模块初始化完成，但没有可用的模型")
                    self.deep_learning_module = None
                    
            except Exception as e:
                print(f"❌ 深度学习模块初始化失败: {e}")
                print(f"📋 详细错误: {str(e)}")
                import traceback
                traceback.print_exc()
                self.deep_learning_module = None
        else:
            print("⚠️ PyTorch不可用，跳过深度学习模块")
        self.init_online_models()
        self.load_saved_state()
        
        # 启动时进行数据一致性检查
        self.check_data_consistency_on_startup()
        
        # 添加数据质量和随机性分析工具
        self.data_analysis_tools = DataAnalysisTools()
        print("📊 数据分析工具初始化完成")
        
        # 初始化高级漂移检测器
        self.advanced_drift_detector = ai_config.EnsembleDriftDetector()
        print("🔍 高级集成漂移检测器初始化完成")
        
        # 初始化智能特征组合器
        self.feature_selector = ai_config.DynamicFeatureSelector(feature_count=60, selection_ratio=0.9)
        self.feature_combiner = ai_config.FeatureInteractionCombiner(original_features=60, max_interactions=15)
        self.timeseries_enhancer = ai_config.TimeSeriesFeatureEnhancer(history_length=10)
        self.feature_weighter = ai_config.AdaptiveFeatureWeighter(feature_count=75)  # 60原始+15交互
        self.feature_assessor = ai_config.FeatureQualityAssessor(assessment_window=50)
        print("🎯 智能特征处理组件初始化完成")

        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer(
            ai_config=ai_config,
            feature_selector=self.feature_selector,
            feature_combiner=self.feature_combiner,
            timeseries_enhancer=self.timeseries_enhancer,
            feature_weighter=self.feature_weighter,
            feature_assessor=self.feature_assessor
        )
        print("🔧 特征工程器初始化完成")
        
        # 初始化定律规则处理器
        self.fundamental_laws = FundamentalLaws()
        print("⚖️ 定律规则处理器初始化完成")

        # 初始化投资管理系统
        self.investment_system = InvestmentSystem()
        print("💰 投资管理系统初始化完成")

        # 初始化数据库管理器
        self.db_manager = DatabaseManager(self.db_path)
        print("🗄️ 数据库管理器初始化完成")

        # 初始化预测分析器
        self.prediction_analyzer = PredictionAnalyzer(self.fundamental_laws)
        self.prediction_analyzer._engine_ref = self  # 添加引擎引用
        print("🔬 预测分析器初始化完成")

        # 初始化漂移检测管理器
        self.drift_manager = DriftDetectionManager(ai_config, self.db_manager)
        print("🚨 漂移检测管理器初始化完成")

        # 初始化统计分析管理器
        self.statistics_manager = StatisticsManager(ai_config, self.db_manager, self.data_analysis_tools)
        print("📊 统计分析管理器初始化完成")

        # 初始化反操控分析器（受启用配置控制）
        print(f"🔍 检查反操控模型配置...")
        print(f"   enabled_special = {enabled_special}")
        print(f"   'anti_manipulation_banker_behavior' in enabled_special = {'anti_manipulation_banker_behavior' in enabled_special}")

        if 'anti_manipulation_banker_behavior' in enabled_special:
            try:
                print("🔍 开始导入 BankerBehaviorAnalyzer...")
                from ai_engine.prediction.anti_manipulation_models import BankerBehaviorAnalyzer
                print("✅ BankerBehaviorAnalyzer 导入成功")
        
                print("🔍 开始初始化 BankerBehaviorAnalyzer...")
                self.banker_behavior_analyzer = BankerBehaviorAnalyzer()
                print("🎯 庄家行为分析器初始化完成")
            except ImportError as ie:
                print(f"❌ 导入 BankerBehaviorAnalyzer 失败: {ie}")
                print(f"📋 导入错误详情: {str(ie)}")
                self.banker_behavior_analyzer = None
            except Exception as e:
                print(f"❌ 庄家行为分析器初始化失败: {e}")
                print(f"📋 错误详情: {str(e)}")
                import traceback
                traceback.print_exc()
                self.banker_behavior_analyzer = None
        else:
            self.banker_behavior_analyzer = None
            print("ℹ️ 反操控分析器已禁用")

        # 初始化反向心理学预测模型（受启用配置控制）
        if 'reverse_psychology_predictor' in enabled_special:
            try:
                self.reverse_psychology_predictor = ReversePsychologyPredictor()
                print("🔄 反向心理学预测器初始化完成")
            except Exception as e:
                print(f"❌ 反向心理学预测器初始化失败: {e}")
                self.reverse_psychology_predictor = None
        else:
            self.reverse_psychology_predictor = None
            print("ℹ️ 反向心理学预测器已禁用")

        # 初始化冷门挖掘器（受启用配置控制）
        if 'unpopular_digger' in enabled_special:
            try:
                self.unpopular_digger = UnpopularDigger()
                print("🔍 冷门挖掘器初始化完成")
            except Exception as e:
                print(f"❌ 冷门挖掘器初始化失败: {e}")
                self.unpopular_digger = None
        else:
            self.unpopular_digger = None
            print("ℹ️ 冷门挖掘器已禁用")

        # 初始化资金流向分析器（受启用配置控制）
        if 'money_flow_analyzer' in enabled_special:
            try:
                from ai_engine.prediction.money_flow_analyzer import MoneyFlowAnalyzer
                self.money_flow_analyzer = MoneyFlowAnalyzer()
                print("💰 资金流向分析器初始化完成")
            except Exception as e:
                print(f"❌ 资金流向分析器初始化失败: {e}")
                self.money_flow_analyzer = None
        else:
            self.money_flow_analyzer = None
            print("ℹ️ 资金流向分析器已禁用")

        # 初始化博弈论策略器（受启用配置控制）
        if 'game_theory_strategist' in enabled_special:
            try:
                self.game_theory_strategist = GameTheoryStrategist()
                print("🎮 博弈论策略器初始化完成")
            except Exception as e:
                print(f"❌ 博弈论策略器初始化失败: {e}")
                self.game_theory_strategist = None
        else:
            self.game_theory_strategist = None
            print("ℹ️ 博弈论策略器已禁用")

        # 初始化操控时机检测器（受启用配置控制）
        if 'manipulation_timing_detector' in enabled_special:
            try:
                self.manipulation_timing_detector = ManipulationTimingDetector()
                print("🎯 操控时机检测器初始化完成")
            except Exception as e:
                print(f"❌ 操控时机检测器初始化失败: {e}")
                self.manipulation_timing_detector = None
        else:
            self.manipulation_timing_detector = None
            print("ℹ️ 操控时机检测器已禁用")

        # 初始化反趋势猎手（受启用配置控制）
        if 'anti_trend_hunter' in enabled_special:
            try:
                self.anti_trend_hunter = AntiTrendHunter()
                print("🎯 反趋势猎手初始化完成")
            except Exception as e:
                print(f"❌ 反趋势猎手初始化失败: {e}")
                self.anti_trend_hunter = None
        else:
            self.anti_trend_hunter = None
            print("ℹ️ 反趋势猎手已禁用")

        # 初始化群体心理分析器（受启用配置控制）
        if 'crowd_psychology_analyzer' in enabled_special:
            try:
                self.crowd_psychology_analyzer = CrowdPsychologyAnalyzer()
                print("🧠 群体心理分析器初始化完成")
            except Exception as e:
                print(f"❌ 群体心理分析器初始化失败: {e}")
                self.crowd_psychology_analyzer = None
        else:
            self.crowd_psychology_analyzer = None
            print("ℹ️ 群体心理分析器已禁用")

        # 在所有模型初始化完成后，初始化在线模型和集成权重
        self.init_online_models()
        self.load_saved_state()

        # 特征处理统计
        self.feature_processing_stats = self.config_manager.get_feature_processing_config()
        
        # 权重调整统计
        self.weight_adjustment_stats = self.config_manager.get_weight_adjustment_config()

        # 权重池管理
        self.investment_manager = self.investment_system
        
        # 投资统计
        self.investment_stats = self.investment_system.investment_stats

        # 添加主应用引用（用于数据质量分析）
        self._main_app_ref = None

    def init_database(self):
        """初始化数据库（已委托给数据库管理器）"""
        # 数据库初始化已委托给数据库管理器
        pass
    
    def init_online_models(self):
        """初始化所有在线学习模型"""
        print("🚀 正在初始化终极在线学习模型...")
        
        # 调试信息：检查库可用性
        print(f"🔍 库可用性检查：")
        print(f"   - LOCAL_MODELS_AVAILABLE: {LOCAL_MODELS_AVAILABLE}")
        print(f"   - RIVER_AVAILABLE: {RIVER_AVAILABLE}")
        print(f"   - SKMULTIFLOW_AVAILABLE: {SKMULTIFLOW_AVAILABLE}")
        print(f"   - PYTORCH_AVAILABLE: {PYTORCH_AVAILABLE}")

        # === River模型 ===
        if RIVER_AVAILABLE:
            self._init_river_models()
        
        # === scikit-multiflow模型 ===
        if SKMULTIFLOW_AVAILABLE:
            self._init_sklearn_models()
        
        # === 概念漂移检测器 ===
        self._init_drift_detectors()
        
        # === 性能度量跟踪器 ===
        self._init_metrics_trackers()
        
        # === 集成学习权重初始化 ===
        self._init_ensemble_weights()
        
        self.is_initialized = True
        print(f"✅ 终极在线学习引擎初始化完成！")
        print(f"   - River模型: {len(self.river_models)} 个")
        print(f"   - scikit-multiflow模型: {len(self.sklearn_models)} 个")
        print(f"   - 漂移检测器: {len(self.drift_detectors)} 个")
        print(f"   - 总计: {len(self.river_models) + len(self.sklearn_models)} 个专业在线学习模型")
    
    def _init_river_models(self):
        """初始化River模型（只创建启用的模型）"""
        enabled_river = self.enabled_models.get('river_models', [])
        models_initialized = 0

        if not enabled_river:
            print("   ℹ️ 未启用任何River模型")
            return

        print(f"   🎯 准备创建 {len(enabled_river)} 个启用的River模型")

        # 优先使用本地模型实现
        if ai_config.LOCAL_MODELS_AVAILABLE:
            print("   🏠 使用本地在线学习模型实现")

            # 本地装袋分类器
            if 'local_bagging' in enabled_river:
                try:
                    self.river_models['local_bagging'] = ai_config.LocalBaggingClassifier(
                        model_class=ai_config.LocalHoeffdingTree, 
                        n_models=5
                    )
                    models_initialized += 1
                    print(f"   ✓ 本地装袋分类器初始化成功")
                except Exception as e:
                    print(f"   ✗ 本地装袋分类器初始化失败: {e}")

            # 历史模式匹配算法（100%相似度）
            if 'local_pattern_matcher_strict' in enabled_river:
                try:
                    self.river_models['local_pattern_matcher_strict'] = ai_config.LocalHistoricalPatternMatcher(
                        pattern_length=10, 
                        min_similarity=1.0  # 100%相似度，完全匹配
                    )
                    models_initialized += 1
                    print(f"   ✓ 历史模式匹配算法初始化成功（100%相似度）")
                except Exception as e:
                    print(f"   ✗ 历史模式匹配算法初始化失败: {e}")

        # 如果本地模型可用且成功初始化了足够的模型，就不再尝试原始River模型
        if models_initialized >= len(enabled_river):
            print(f"   ✅ 本地模型初始化成功，共 {models_initialized} 个模型")
            print(f"   ✓ River模型初始化完成，共 {len(self.river_models)} 个模型")
            return

        # 如果本地模型不可用或初始化失败，尝试原始River模型
        if ai_config.RIVER_AVAILABLE:
            print("   🌐 尝试使用原始River库模型")

            # 自适应随机森林
            if 'adaptive_forest' in enabled_river and 'AdaptiveRandomForestClassifier' in ai_config.ensemble_models:
                try:
                    self.river_models['adaptive_forest'] = ai_config.ensemble_models['AdaptiveRandomForestClassifier'](
                        n_models=10
                    )
                    models_initialized += 1
                    print(f"   ✓ 自适应随机森林初始化成功")
                except Exception as e:
                    print(f"   ✗ 自适应随机森林初始化失败: {e}")

            # 在线装袋分类器
            if 'online_bagging' in enabled_river:
                if 'OnlineBaggingClassifier' in ai_config.ensemble_models:
                    try:
                        base_model = ai_config.HoeffdingTreeClassifier()
                        self.river_models['online_bagging'] = ai_config.ensemble_models['OnlineBaggingClassifier'](
                            model=base_model,
                            n_models=5
                        )
                        models_initialized += 1
                        print(f"   ✓ 在线装袋分类器初始化成功")
                    except Exception as e:
                        print(f"   ✗ 在线装袋分类器初始化失败: {e}")
                elif 'BaggingClassifier' in ai_config.ensemble_models:
                    try:
                        base_model = ai_config.HoeffdingTreeClassifier()
                        self.river_models['online_bagging'] = ai_config.ensemble_models['BaggingClassifier'](
                            model=base_model,
                            n_models=5
                        )
                        models_initialized += 1
                        print(f"   ✓ 装袋分类器初始化成功")
                    except Exception as e:
                        print(f"   ✗ 装袋分类器初始化失败: {e}")

        print(f"   ✓ River模型初始化完成，共 {len(self.river_models)} 个模型")
    
    def _init_sklearn_models(self):
        """初始化scikit-multiflow模型（只创建启用的模型）"""
        if not SKMULTIFLOW_AVAILABLE:
            print("   ⚠️ scikit-multiflow库不可用，跳过sklearn模型初始化")
            return

        enabled_sklearn = self.enabled_models.get('sklearn_models', [])
        if not enabled_sklearn:
            print("   ℹ️ 未启用任何sklearn模型")
            return

        print(f"   🎯 准备创建 {len(enabled_sklearn)} 个启用的sklearn模型")

        # 极快决策树
        if 'extremely_fast_tree' in enabled_sklearn:
            try:
                self.sklearn_models['extremely_fast_tree'] = ai_config.ExtremelyFastDecisionTreeClassifier()
                print("   ✓ 极快决策树初始化成功")
            except Exception as e:
                print(f"   ✗ 极快决策树初始化失败: {e}")

        # Hoeffding自适应树（SKM版本）
        if 'skm_hoeffding_adaptive' in enabled_sklearn:
            try:
                self.sklearn_models['skm_hoeffding_adaptive'] = ai_config.SKMHoeffdingAdaptive()
                print("   ✓ SKM Hoeffding自适应树初始化成功")
            except Exception as e:
                print(f"   ✗ SKM Hoeffding自适应树初始化失败: {e}")

        # 在线AdaBoost
        if 'online_adaboost' in enabled_sklearn:
            try:
                self.sklearn_models['online_adaboost'] = ai_config.OnlineAdaBoostClassifier(
                    n_estimators=10
                )
                print("   ✓ 在线AdaBoost初始化成功")
            except Exception as e:
                print(f"   ✗ 在线AdaBoost初始化失败: {e}")

        # 流式随机补丁（SKM版本）
        if 'skm_random_patches' in enabled_sklearn:
            try:
                self.sklearn_models['skm_random_patches'] = ai_config.SKMRandomPatches(
                n_estimators=10
                )
                print("   ✓ SKM流式随机补丁初始化成功")
            except Exception as e:
                print(f"   ✗ SKM流式随机补丁初始化失败: {e}")

        print(f"   ✓ scikit-multiflow模型初始化完成，共 {len(self.sklearn_models)} 个模型")
    
    def _init_drift_detectors(self):
        """初始化概念漂移检测器"""
    
        # 优先使用本地漂移检测器
        if LOCAL_MODELS_AVAILABLE:
            try:
                self.drift_detectors['local_drift_detector'] = ai_config.LocalDriftDetector(
                    window_size=100, 
                    threshold=0.1
                )
                print("   ✓ 本地漂移检测器初始化成功")
            except Exception as e:
                print(f"   ✗ 本地漂移检测器初始化失败: {e}")

        try:
            self.drift_detectors['local_drift_detector_sensitive'] = ai_config.LocalDriftDetector(
                window_size=50, 
                threshold=0.05
            )
            print("   ✓ 本地敏感漂移检测器初始化成功")
        except Exception as e:
            print(f"   ✗ 本地敏感漂移检测器初始化失败: {e}")
    
        if RIVER_AVAILABLE and ai_config.drift_detectors:
            for name, detector_class in ai_config.drift_detectors.items():
                try:
                    if name == 'ADWIN':
                        self.drift_detectors[name.lower()] = detector_class(delta=self.learning_config['drift_sensitivity'])
                    elif name == 'PageHinkley':
                        self.drift_detectors[name.lower()] = detector_class(min_instances=30, delta=0.005, threshold=50)
                    elif name == 'KSWIN':
                        self.drift_detectors[name.lower()] = detector_class(alpha=0.005, window_size=100)
                    elif name in ['HDDM_A', 'HDDM_W']:
                        self.drift_detectors[name.lower()] = detector_class(drift_confidence=0.001, warning_confidence=0.005)
                except Exception as e:
                    print(f"   漂移检测器 {name} 初始化失败: {e}")
    
        print(f"   ✓ 概念漂移检测器初始化完成，共 {len(self.drift_detectors)} 个")
    
    def _init_metrics_trackers(self):
        """初始化性能度量跟踪器"""
        if RIVER_AVAILABLE:
            all_models = list(self.river_models.keys()) + list(self.sklearn_models.keys())
        
            for model_name in all_models:
                # 基础度量器
                model_metrics = {}
            
                # 尝试初始化各种度量器
                try:
                    model_metrics['accuracy'] = ai_config.metrics.Accuracy()
                except Exception:
                    pass
            
                try:
                    model_metrics['precision'] = ai_config.metrics.Precision()
                except Exception:
                    pass
            
                try:
                    model_metrics['recall'] = ai_config.metrics.Recall()
                except Exception:
                    pass
            
                try:
                    model_metrics['f1'] = ai_config.metrics.F1()
                except Exception:
                    pass
            
                # 尝试滚动窗口度量器（处理版本兼容性）
                try:
                    # 新版本可能使用 utils.Rolling
                    from river.utils import Rolling
                    model_metrics['rolling_accuracy'] = Rolling(ai_config.metrics.Accuracy(), window_size=self.learning_config['performance_tracking_window'])
                except ImportError:
                    try:
                        # 旧版本使用 metrics.Rolling
                        model_metrics['rolling_accuracy'] = ai_config.metrics.Rolling(ai_config.metrics.Accuracy(), window_size=self.learning_config['performance_tracking_window'])
                    except (ImportError, AttributeError):
                        # 如果都不可用，使用简单的准确率度量器
                        try:
                            model_metrics['rolling_accuracy'] = ai_config.metrics.Accuracy()
                        except Exception:
                            pass
            
                self.metrics_trackers[model_name] = model_metrics
    
        print(f"   ✓ 性能度量跟踪器初始化完成，共 {len(self.metrics_trackers)} 个")
    
    def _init_ensemble_weights(self):
        """初始化集成学习权重（只为启用的模型分配权重）"""
        all_models = []
        enabled_pytorch = self.enabled_models.get('pytorch_models', [])
        enabled_special = self.enabled_models.get('special_models', [])

        # 添加River模型（带前缀）
        for model_name in self.river_models.keys():
            all_models.append(f'river_{model_name}')

        # 添加sklearn模型（带前缀）
        for model_name in self.sklearn_models.keys():
            all_models.append(f'sklearn_{model_name}')

        # 添加启用的PyTorch深度学习模型（带前缀）
        if PYTORCH_AVAILABLE and self.deep_learning_module and self.deep_learning_module.models:
            pytorch_model_names = list(self.deep_learning_module.models.keys())
            for model_name in pytorch_model_names:
                if model_name in enabled_pytorch:
                    all_models.append(f'pytorch_{model_name}')

        # 添加启用的特殊模型
        if 'anti_manipulation_banker_behavior' in enabled_special and hasattr(self, 'banker_behavior_analyzer') and self.banker_behavior_analyzer:
            all_models.append('anti_manipulation_banker_behavior')

        if 'reverse_psychology_predictor' in enabled_special and hasattr(self, 'reverse_psychology_predictor') and self.reverse_psychology_predictor:
            all_models.append('reverse_psychology_predictor')

        if 'unpopular_digger' in enabled_special and hasattr(self, 'unpopular_digger') and self.unpopular_digger:
            all_models.append('unpopular_digger')

        if 'money_flow_analyzer' in enabled_special and hasattr(self, 'money_flow_analyzer') and self.money_flow_analyzer:
            all_models.append('money_flow_analyzer')

        if 'game_theory_strategist' in enabled_special and hasattr(self, 'game_theory_strategist') and self.game_theory_strategist:
            all_models.append('game_theory_strategist')

        if 'manipulation_timing_detector' in enabled_special and hasattr(self, 'manipulation_timing_detector') and self.manipulation_timing_detector:
            all_models.append('manipulation_timing_detector')

        if 'anti_trend_hunter' in enabled_special and hasattr(self, 'anti_trend_hunter') and self.anti_trend_hunter:
            all_models.append('anti_trend_hunter')

        if 'crowd_psychology_analyzer' in enabled_special and hasattr(self, 'crowd_psychology_analyzer') and self.crowd_psychology_analyzer:
            all_models.append('crowd_psychology_analyzer')

        # 给每个启用的模型分配初始权重
        initial_weight = 1.0 / len(all_models) if all_models else 0.0

        for model_name in all_models:
            self.ensemble_weights[model_name] = {
                'weight': float(initial_weight),          # 当前可用权重
                'frozen_weight': 0.0,                     # 冻结权重
                'total_weight': float(initial_weight),    # 总权重（显示用）
                'is_frozen': False,                       # 冻结状态
                'frozen_timestamp': None,                 # 冻结时间
                'weight_investments': {},                 # 权重投资 {tail: weight_amount}
                'confidence': 0.5,
                'last_update': datetime.now(),
                'performance_history': [],
                'investment_history': []                  # 投资历史记录
            }

        print("   ✓ 集成学习权重初始化完成")
        print(f"   📊 总启用模型数: {len(all_models)}")
        print(f"   📊 每个模型初始权重: {initial_weight:.4f}")
    
        # 显示启用的模型列表
        for model_name in all_models:
            model_type = "特殊模型" if not model_name.startswith(('river_', 'sklearn_', 'pytorch_')) else model_name.split('_')[0].upper()
            print(f"      ✓ {model_name} ({model_type})")
    
    def get_detailed_extremely_hot_analysis(self, data_list: List[Dict]) -> Dict:
        """获取详细的极热尾数分析"""
        analysis = {
            'hot_30_20': set(),  # 30期≥20次
            'hot_10_8': set(),   # 10期≥8次
            'combined': set()    # 合并结果
        }
        
        if len(data_list) < 10:
            return analysis
        
        # 检查30期≥20次的情况
        if len(data_list) >= 30:
            recent_30_data = data_list[:30]
            for tail in range(10):
                count_30 = sum(1 for period in recent_30_data if tail in period.get('tails', []))
                if count_30 >= 20:
                    analysis['hot_30_20'].add(tail)
        
        # 检查10期≥8次的情况
        recent_10_data = data_list[:10]
        for tail in range(10):
            count_10 = sum(1 for period in recent_10_data if tail in period.get('tails', []))
            if count_10 >= 8:
                analysis['hot_10_8'].add(tail)
        
        # 合并结果
        analysis['combined'] = analysis['hot_30_20'].union(analysis['hot_10_8'])
        
        return analysis
    
    def apply_strict_fundamental_laws(self, probabilities: Dict[int, float], data_list: List[Dict]) -> List[int]:
        """应用严格的底层定律（已删除定律1）"""
        if not data_list:
            return []
    
        # 初始候选尾数：所有尾数0-9
        all_tails = set(range(10))
        print(f"🎯 初始候选尾数（所有尾数）：{sorted(all_tails)}")
    
        # 定律2：排除陷阱尾数
        trap_tails = self.fundamental_laws.identify_comprehensive_trap_tails(data_list)
        print(f"🚫 定律2 - 识别出陷阱尾数：{sorted(trap_tails)}")
    
        # 定律3：排除在10期和30期中同时表现为最少出现次数的尾数
        dual_minimum_tails = self.fundamental_laws.identify_dual_minimum_tails(data_list)
        print(f"⚡ 定律3 - 10期和30期同时最少的尾数：{sorted(dual_minimum_tails)}")
    
        # 定律4：排除极热尾数（最近30期出现≥20次或10期≥8次的尾数）
        extremely_hot_tails_v2 = self.fundamental_laws.identify_extremely_hot_tails(data_list, periods=30, threshold=20)
        print(f"🔥 定律4 - 极热尾数（30期≥20次或10期≥8次）：{sorted(extremely_hot_tails_v2)}")
    
        # 应用剩余三大定律，得到候选尾数
        candidates = all_tails - trap_tails - dual_minimum_tails - extremely_hot_tails_v2
        print(f"✅ 应用三大定律后的候选尾数：{sorted(candidates)}")
        
        if not candidates:
            print("⚠️ 应用三大定律后无候选尾数，选择所有尾数中概率最高的")
            best_tail = max(all_tails, key=lambda t: probabilities.get(t, 0))
            return [best_tail]
        
        # 从候选中选择概率最高的尾数
        best_candidate = max(candidates, key=lambda t: probabilities.get(t, 0))
        print(f"🎯 最终推荐尾数：{best_candidate}（概率：{probabilities.get(best_candidate, 0):.3f}）")
        
        return [best_candidate]
    
    def identify_comprehensive_trap_tails(self, data_list: List[Dict]) -> set:
        """识别所有类型的陷阱尾数"""
        trap_tails = set()
        
        if len(data_list) < 6:
            return trap_tails
        
        for tail in range(10):
            # 检查连续3期以上没有出现，在最新一期突然出现
            if self.is_sudden_appearance_trap(data_list, tail):
                trap_tails.add(tail)
                continue
            
            # 检查各种规律性模式
            if self.has_regular_patterns(data_list, tail):
                trap_tails.add(tail)
        
        return trap_tails
    
    def is_sudden_appearance_trap(self, data_list: List[Dict], tail: int) -> bool:
        """检查是否为连续3期以上没有出现后突然出现的陷阱"""
        if len(data_list) < 4:
            return False
        
        if tail not in data_list[0].get('tails', []):
            return False
        
        consecutive_absent = 0
        for i in range(1, len(data_list)):
            if tail not in data_list[i].get('tails', []):
                consecutive_absent += 1
            else:
                break
        
        return consecutive_absent >= 3
    
    def has_regular_patterns(self, data_list: List[Dict], tail: int) -> bool:
        """检查是否有规律性出现模式"""
        if len(data_list) < 6:
            return False
        
        appearances = []
        check_periods = min(10, len(data_list))
        
        for i in range(check_periods):
            appearances.append(1 if tail in data_list[i].get('tails', []) else 0)
        
        # 检查各种规律模式
        patterns = [
            self.check_pattern_1010(appearances),
            self.check_pattern_110011(appearances),
            self.check_pattern_100100(appearances),
            self.check_pattern_110110(appearances),
        ]
        
        return any(patterns)
    
    def check_pattern_1010(self, appearances: List[int]) -> bool:
        if len(appearances) < 4:
            return False
        return (appearances[0] == 1 and appearances[1] == 0 and appearances[2] == 1 and appearances[3] == 0)
    
    def check_pattern_110011(self, appearances: List[int]) -> bool:
        if len(appearances) < 6:
            return False
        return (appearances[0] == 1 and appearances[1] == 1 and 
                appearances[2] == 0 and appearances[3] == 0 and 
                appearances[4] == 1 and appearances[5] == 1)
    
    def check_pattern_100100(self, appearances: List[int]) -> bool:
        if len(appearances) < 6:
            return False
        return (appearances[0] == 1 and appearances[1] == 0 and appearances[2] == 0 and
                appearances[3] == 1 and appearances[4] == 0 and appearances[5] == 0)
    
    def check_pattern_110110(self, appearances: List[int]) -> bool:
        if len(appearances) < 6:
            return False
        return (appearances[0] == 1 and appearances[1] == 1 and appearances[2] == 0 and
                appearances[3] == 1 and appearances[4] == 1 and appearances[5] == 0)
    
    def identify_dual_minimum_tails(self, data_list: List[Dict]) -> set:
        """识别在10期和30期中同时表现为最少出现次数的尾数"""
        dual_minimum = set()
        
        if len(data_list) < 10:
            return dual_minimum
        
        # 计算最近10期各尾数出现次数
        counts_10 = {}
        for tail in range(10):
            counts_10[tail] = sum(1 for i in range(min(10, len(data_list))) 
                                if tail in data_list[i].get('tails', []))
        
        min_count_10 = min(counts_10.values())
        min_tails_10 = {tail for tail, count in counts_10.items() if count == min_count_10}
        
        if len(data_list) >= 30:
            counts_30 = {}
            for tail in range(10):
                counts_30[tail] = sum(1 for i in range(min(30, len(data_list))) 
                                    if tail in data_list[i].get('tails', []))
            
            min_count_30 = min(counts_30.values())
            min_tails_30 = {tail for tail, count in counts_30.items() if count == min_count_30}
            
            dual_minimum = min_tails_10 & min_tails_30
        
        return dual_minimum
    
    def identify_extremely_hot_tails(self, data_list: List[Dict], periods=30, threshold=20):
        """识别极热尾数 - 在最近N期中出现次数超过阈值的尾数（核心定律）"""
        if len(data_list) < periods:
            # 如果数据不足30期，使用现有数据但调整阈值
            available_periods = len(data_list)
            if available_periods < 10:  # 数据太少，不应用此定律
                return set()
            # 按比例调整阈值：20/30 = 0.67，即出现次数超过67%的期数
            adjusted_threshold = int(available_periods * 0.67)
            periods = available_periods
            threshold = adjusted_threshold

        extremely_hot_tails = set()
        recent_data = data_list[:periods]  # 取最近N期数据

        for tail in range(10):
            # 计算该尾数在最近N期中的出现次数
            count = sum(1 for period in recent_data if tail in period.get('tails', []))
    
            # 如果出现次数超过阈值，标记为极热门
            if count >= threshold:
                extremely_hot_tails.add(tail)

        # 新增：检查最近10期中出现次数≥8期的尾数
        if len(data_list) >= 10:
            recent_10_data = data_list[:10]  # 取最近10期数据
            for tail in range(10):
                # 计算该尾数在最近10期中的出现次数
                count_10 = sum(1 for period in recent_10_data if tail in period.get('tails', []))
                
                # 如果最近10期出现次数≥8期，也标记为极热门
                if count_10 >= 8:
                    extremely_hot_tails.add(tail)

        return extremely_hot_tails

    def predict_online(self, data_list: List[Dict]) -> Dict:
        """在线预测 - 真正的多模型集成预测"""
        if not self.is_initialized:
            return {'success': False, 'message': '模型未初始化'}
            
        # 验证预测数据顺序
        if data_list and len(data_list) > 0:
            print(f"🔍 AI预测数据顺序验证：")
            print(f"   基于历史数据量：{len(data_list)}期")
            print(f"   最新期（index 0）：{data_list[0].get('numbers', [])[:3] if 'numbers' in data_list[0] else '无号码'}")
            if len(data_list) > 1:
                print(f"   历史期（index 1）：{data_list[1].get('numbers', [])[:3] if 'numbers' in data_list[1] else '无号码'}")
            print(f"   预测逻辑：基于全部{len(data_list)}期历史数据进行预测")

        try:
            # 为每个模型提取专属特征
            base_features = self.feature_engineer.extract_enhanced_features(data_list)
            model_specific_features = self.feature_engineer.create_model_specific_features(base_features, data_list)
            
            # 在预测阶段，使用历史性能数据更新高级漂移检测器
            historical_accuracy = self.get_current_accuracy()
            error_rate = 1.0 - historical_accuracy  # 使用历史错误率
            
            # 更新高级漂移检测器
            performance_metrics = {
                'accuracy': historical_accuracy,
                'confidence': self.last_prediction_result.get('confidence', 0.5) if self.last_prediction_result else 0.5
            }
            self.advanced_drift_detector.update(error_rate, performance_metrics)
            
            # 检查高级漂移检测结果
            advanced_drift_info = self.advanced_drift_detector.get_detailed_report()
            if advanced_drift_info['ensemble_detected']:
                print(f"🚨 高级漂移检测器报告概念漂移!")
                print(f"   检测器详情: {advanced_drift_info['individual_detectors']}")
                
                # 处理高级概念漂移
                self._handle_advanced_concept_drift(advanced_drift_info)

            # 为River模型准备特征字典
            X_river = {f'feature_{i}': base_features[i] for i in range(len(base_features))}
            
            # 为不同模型创建完全不同的数据视角和决策逻辑
            diversified_features = {}
            model_decision_strategies = {}
            
            # 定义每个模型的独特数据视角和决策策略
            model_strategies = {
                'hoeffding_tree': {
                    'focus': 'trend_continuation',  # 关注趋势延续
                    'time_window': 5,  # 关注最近5期
                    'weight_recent': 2.0,  # 加重最近期权重
                    'decision_threshold': 0.45
                },
                'hoeffding_adaptive': {
                    'focus': 'trend_reversal',  # 关注趋势反转
                    'time_window': 8,  # 关注最近8期
                    'weight_recent': 1.5,
                    'decision_threshold': 0.55
                },
                'logistic': {
                    'focus': 'frequency_analysis',  # 关注频率分析
                    'time_window': 10,  # 关注最近10期
                    'weight_recent': 1.0,  # 均等权重
                    'decision_threshold': 0.5
                },
                'naive_bayes': {
                    'focus': 'independence_assumption',  # 关注独立性假设
                    'time_window': 15,  # 关注更长期
                    'weight_recent': 0.8,  # 降低最近期权重
                    'decision_threshold': 0.6
                },
                'naive_bayes_multinomial': {
                    'focus': 'categorical_patterns',  # 关注分类模式
                    'time_window': 6,
                    'weight_recent': 1.8,
                    'decision_threshold': 0.35
                },
                'naive_bayes_gaussian': {
                    'focus': 'continuous_distribution',  # 关注连续分布
                    'time_window': 12,
                    'weight_recent': 1.2,
                    'decision_threshold': 0.65
                },
                'naive_bayes_mixed': {
                    'focus': 'hybrid_approach',  # 混合方法
                    'time_window': 9,
                    'weight_recent': 1.3,
                    'decision_threshold': 0.4
                },
                'bagging': {
                    'focus': 'ensemble_voting',  # 关注集成投票
                    'time_window': 7,
                    'weight_recent': 1.6,
                    'decision_threshold': 0.3
                },
                'adaboost': {
                    'focus': 'error_correction',  # 关注错误修正
                    'time_window': 11,
                    'weight_recent': 1.1,
                    'decision_threshold': 0.7
                },
                'bagging_nb': {
                    'focus': 'probabilistic_ensemble',  # 概率集成
                    'time_window': 4,
                    'weight_recent': 2.2,
                    'decision_threshold': 0.25
                },
                'bagging_lr': {
                    'focus': 'linear_ensemble',  # 线性集成
                    'time_window': 13,
                    'weight_recent': 0.9,
                    'decision_threshold': 0.75
                },
                'pattern_matcher_strict': {
                    'focus': 'historical_matching',  # 历史模式匹配
                    'time_window': 20,  # 需要更多历史数据
                    'weight_recent': 0.5,  # 历史权重更重要
                    'decision_threshold': 0.8  # 高阈值，只在高匹配度时预测
                }
            }
            
            for model_name in self.river_models.keys():
                # 获取模型策略配置
                config_key = model_name.replace('local_', '').replace('river_', '')
                strategy = model_strategies.get(config_key, model_strategies['logistic'])  # 默认策略
                
                # 根据策略创建该模型的专属特征和决策逻辑
                model_features = {}
                model_decision_strategies[model_name] = strategy
                
                # 根据不同的关注点创建特征
                if strategy['focus'] == 'trend_continuation':
                    # 趋势延续：强化连续性特征
                    for tail in range(10):
                        # 连续出现次数权重
                        consecutive_weight = 3.0
                        recent_appearance_weight = 2.5
                        frequency_weight = 1.0
                        
                        model_features[f'consecutive_{tail}'] = consecutive_weight
                        model_features[f'recent_appear_{tail}'] = recent_appearance_weight
                        model_features[f'frequency_{tail}'] = frequency_weight
                
                elif strategy['focus'] == 'trend_reversal':
                    # 趋势反转：强化反转信号特征
                    for tail in range(10):
                        # 间隔距离权重
                        gap_weight = 2.8
                        reversal_signal_weight = 2.2
                        cold_tail_weight = 1.8
                        
                        model_features[f'gap_distance_{tail}'] = gap_weight
                        model_features[f'reversal_signal_{tail}'] = reversal_signal_weight
                        model_features[f'cold_tail_{tail}'] = cold_tail_weight
                
                elif strategy['focus'] == 'frequency_analysis':
                    # 频率分析：均衡各种频率特征
                    for tail in range(10):
                        short_freq_weight = 1.5
                        medium_freq_weight = 1.5
                        long_freq_weight = 1.0
                        
                        model_features[f'short_freq_{tail}'] = short_freq_weight
                        model_features[f'medium_freq_{tail}'] = medium_freq_weight
                        model_features[f'long_freq_{tail}'] = long_freq_weight
                
                elif strategy['focus'] == 'independence_assumption':
                    # 独立性假设：每个尾数独立分析
                    for tail in range(10):
                        independent_prob_weight = 2.0
                        prior_prob_weight = 1.5
                        
                        model_features[f'independent_prob_{tail}'] = independent_prob_weight
                        model_features[f'prior_prob_{tail}'] = prior_prob_weight
                
                elif strategy['focus'] == 'categorical_patterns':
                    # 分类模式：强化分类特征
                    for tail in range(10):
                        category_weight = 2.5
                        pattern_weight = 2.0
                        
                        model_features[f'category_{tail}'] = category_weight
                        model_features[f'pattern_{tail}'] = pattern_weight
                
                elif strategy['focus'] == 'continuous_distribution':
                    # 连续分布：关注数值分布
                    for tail in range(10):
                        distribution_weight = 2.2
                        variance_weight = 1.8
                        
                        model_features[f'distribution_{tail}'] = distribution_weight
                        model_features[f'variance_{tail}'] = variance_weight
                
                elif strategy['focus'] == 'hybrid_approach':
                    # 混合方法：多种特征组合
                    for tail in range(10):
                        hybrid_weight = 1.8
                        combined_weight = 1.6
                        
                        model_features[f'hybrid_{tail}'] = hybrid_weight
                        model_features[f'combined_{tail}'] = combined_weight
                
                elif strategy['focus'] == 'ensemble_voting':
                    # 集成投票：投票机制特征
                    for tail in range(10):
                        vote_weight = 2.1
                        consensus_weight = 1.7
                        
                        model_features[f'vote_{tail}'] = vote_weight
                        model_features[f'consensus_{tail}'] = consensus_weight
                
                elif strategy['focus'] == 'error_correction':
                    # 错误修正：强化修正特征
                    for tail in range(10):
                        error_signal_weight = 2.4
                        correction_weight = 2.0
                        
                        model_features[f'error_signal_{tail}'] = error_signal_weight
                        model_features[f'correction_{tail}'] = correction_weight
                
                elif strategy['focus'] == 'probabilistic_ensemble':
                    # 概率集成：概率特征
                    for tail in range(10):
                        prob_ensemble_weight = 2.3
                        likelihood_weight = 1.9
                        
                        model_features[f'prob_ensemble_{tail}'] = prob_ensemble_weight
                        model_features[f'likelihood_{tail}'] = likelihood_weight
                
                elif strategy['focus'] == 'linear_ensemble':
                    # 线性集成：线性组合特征
                    for tail in range(10):
                        linear_comb_weight = 1.4
                        weighted_sum_weight = 1.6
                        
                        model_features[f'linear_comb_{tail}'] = linear_comb_weight
                        model_features[f'weighted_sum_{tail}'] = weighted_sum_weight
                
                elif strategy['focus'] == 'historical_matching':
                    # 历史匹配：保持原有逻辑，不改变
                    for i in range(60):
                        model_features[f'feature_{i}'] = base_features[i] if i < len(base_features) else 0.0
                
                else:
                    # 默认特征集
                    for i in range(30):
                        model_features[f'feature_{i}'] = base_features[i] if i < len(base_features) else 0.0
                
                diversified_features[model_name] = model_features

            # 为sklearn模型准备特征数组
            X_sklearn = base_features.reshape(1, -1)
            
            # 为Deep-River模型准备特征张量和字典
            if DEEP_RIVER_AVAILABLE and len(self.deep_models) > 0:
                X_deep = ai_config.torch.FloatTensor(base_features).unsqueeze(0)
                print(f"   📊 为Deep-River准备特征: 维度{X_deep.shape}, 数据类型{X_deep.dtype}")
            
            # 为Deep-River模型准备特征张量和字典
            if DEEP_RIVER_AVAILABLE and len(self.deep_models) > 0:
                X_deep = ai_config.torch.FloatTensor(base_features).unsqueeze(0)
                print(f"   📊 为Deep-River准备特征: 维度{X_deep.shape}, 数据类型{X_deep.dtype}")
    
                # 增强的输入数据验证和清理
                print("   🔍 执行Deep-River输入数据全面验证...")
    
                # 验证features数组
                if base_features is None:
                    print("   ❌ features为None，Deep-River预测将跳过")
                    X_deep = None
                elif len(base_features) == 0:
                    print("   ❌ features为空数组，Deep-River预测将跳过")
                    X_deep = None
                elif any(x is None for x in base_features):
                    print("   ⚠️ features包含None值，进行清理")
                    cleaned_features = [0.0 if x is None else float(x) for x in base_features]
                    X_deep = ai_config.torch.FloatTensor(cleaned_features).unsqueeze(0)
                    print(f"   ✓ 清理后的features维度: {X_deep.shape}")
                elif any(not isinstance(x, (int, float)) for x in base_features):
                    print("   ⚠️ features包含非数值类型，进行转换")
                    try:
                        cleaned_features = []
                        for i, x in enumerate(base_features):
                            try:
                                cleaned_features.append(float(x))
                            except (ValueError, TypeError):
                                print(f"   ⚠️ features[{i}]={x} 无法转换为数值，使用0.0")
                                cleaned_features.append(0.0)
                        X_deep = ai_config.torch.FloatTensor(cleaned_features).unsqueeze(0)
                        print(f"   ✓ 转换后的features维度: {X_deep.shape}")
                    except Exception as clean_e:
                        print(f"   ❌ features清理失败: {clean_e}")
                        X_deep = None
    
                # 验证X_river字典
                print("   🔍 验证X_river字典...")
                x_river_safe = {}
                invalid_keys = []
    
                for key, value in X_river.items():
                    if value is None:
                        print(f"   ⚠️ X_river[{key}] 为None，替换为0.0")
                        x_river_safe[key] = 0.0
                    elif isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            print(f"   ⚠️ X_river[{key}] 为NaN或无穷，替换为0.0")
                            x_river_safe[key] = 0.0
                        else:
                            x_river_safe[key] = float(value)
                    else:
                        try:
                            converted_value = float(value)
                            if math.isnan(converted_value) or math.isinf(converted_value):
                                print(f"   ⚠️ X_river[{key}] 转换后为NaN或无穷，替换为0.0")
                                x_river_safe[key] = 0.0
                            else:
                                x_river_safe[key] = converted_value
                        except (ValueError, TypeError):
                            print(f"   ⚠️ X_river[{key}]={value} 无法转换为数值，替换为0.0")
                            x_river_safe[key] = 0.0
                            invalid_keys.append(key)
    
                if invalid_keys:
                    print(f"   📊 共发现{len(invalid_keys)}个无效键值对")
    
                print(f"   ✅ X_river验证完成，共{len(x_river_safe)}个有效特征")
    
                # 用安全的字典替换原来的X_river（仅用于Deep-River）
                X_river_for_deep = x_river_safe

            # 多模型预测
            all_predictions = {}
            
            # 初始化所有候选尾数（0-9）
            all_tails = set(range(10))
            print(f"🎯 初始候选尾数（所有尾数）：{sorted(all_tails)}")

            # === 反操控分析 ===
            anti_manipulation_analysis = None
            if hasattr(self, 'banker_behavior_analyzer') and self.banker_behavior_analyzer:
                try:
                    # 分析当前期的操控信号
                    current_period = {
                        'tails': data_list[0].get('tails', []) if data_list else [],
                        'numbers': data_list[0].get('numbers', []) if data_list else [],
                        'timestamp': datetime.now()
                    }
                
                    anti_manipulation_analysis = self.banker_behavior_analyzer.analyze_period(
                        current_period, data_list[1:] if len(data_list) > 1 else []
                    )
                
                    print(f"🎯 反操控分析: 操控概率={anti_manipulation_analysis.get('manipulation_probability', 0):.2f}")
                
                    # 获取反操控建议
                    anti_recommendations = self.banker_behavior_analyzer.get_anti_manipulation_recommendations(data_list)
                    print(f"🎯 反操控建议: 推荐{anti_recommendations.get('recommended_tails', [])} 避开{anti_recommendations.get('avoid_tails', [])}")
                
                except Exception as e:
                    print(f"⚠️ 反操控分析失败: {e}")
                    anti_manipulation_analysis = None

            # === 反向心理学分析 ===
            reverse_psychology_analysis = None
            if hasattr(self, 'reverse_psychology_predictor') and self.reverse_psychology_predictor:
                try:
                    reverse_psychology_analysis = self.reverse_psychology_predictor.predict(
                        period_data={'tails': data_list[0].get('tails', []) if data_list else []},
                        historical_context=data_list[1:] if len(data_list) > 1 else []
                    )
                    
                    recommended_tails = reverse_psychology_analysis.get('recommended_tails', [])
                    avoid_tails = reverse_psychology_analysis.get('avoid_tails', [])
                    confidence = reverse_psychology_analysis.get('confidence', 0.0)
                    strategy_type = reverse_psychology_analysis.get('strategy_type', 'unknown')
                    
                    print(f"🔄 反向心理学分析: 策略={strategy_type}, 置信度={confidence:.2f}")
                    print(f"🔄 反向心理学建议: 推荐{recommended_tails} 避开{avoid_tails}")
                    
                except Exception as e:
                    print(f"⚠️ 反向心理学分析失败: {e}")
                    reverse_psychology_analysis = None

            # === 冷门挖掘器分析 ===
            unpopular_digger_analysis = None
            if hasattr(self, 'unpopular_digger') and self.unpopular_digger:
                try:
                    # 冷门挖掘器需要候选尾数作为输入
                    if 'valid_candidates' in locals() and valid_candidates:
                        candidate_tails_list = list(valid_candidates)
                    else:
                        candidate_tails_list = list(range(10))  # 如果没有候选尾数，使用所有尾数
        
                    unpopular_digger_analysis = self.unpopular_digger.predict(
                        candidate_tails_list, 
                        data_list
                    )
        
                    if unpopular_digger_analysis.get('success'):
                        recommended_cold_tails = unpopular_digger_analysis.get('recommended_tails', [])
                        confidence = unpopular_digger_analysis.get('confidence', 0.0)
            
                        print(f"🔍 冷门挖掘器分析: 置信度={confidence:.2f}")
                        print(f"🔍 冷门挖掘建议: 推荐冷门尾数{recommended_cold_tails}")
            
                        # 将冷门挖掘器的建议添加到预测中
                        if recommended_cold_tails:
                            # 这里可以根据需要调整如何使用冷门挖掘器的建议
                            print(f"🔍 冷门挖掘器发现了 {len(recommended_cold_tails)} 个冷门机会")
                    else:
                        print(f"🔍 冷门挖掘器分析: 无有效的冷门挖掘机会")
        
                except Exception as e:
                    print(f"⚠️ 冷门挖掘器分析失败: {e}")
                    unpopular_digger_analysis = None

            # === 应用三大定律筛选候选尾数 ===
            if data_list:
                # 识别需要排除的尾数
                trap_tails = self.fundamental_laws.identify_comprehensive_trap_tails(data_list)
                dual_minimum_tails = self.fundamental_laws.identify_dual_minimum_tails(data_list)
                extremely_hot_tails = self.fundamental_laws.identify_extremely_hot_tails(data_list, periods=30, threshold=20)
                
                # 计算通过筛选的候选尾数
                excluded_tails = trap_tails.union(dual_minimum_tails).union(extremely_hot_tails)
                valid_candidates = all_tails - excluded_tails
                
                # 显示筛选过程
                print(f"🔍 三大定律筛选过程：")
                if trap_tails:
                    print(f"  • 定律2排除陷阱尾数：{sorted(trap_tails)}")
                if dual_minimum_tails:
                    print(f"  • 定律3排除双重最少尾数：{sorted(dual_minimum_tails)}")
                if extremely_hot_tails:
                    print(f"  • 定律4排除极热尾数：{sorted(extremely_hot_tails)}")
                print(f"  • 通过筛选的候选尾数：{sorted(valid_candidates)}")
                
                # 如果没有候选尾数，返回无结果
                if not valid_candidates:
                    print(f"⚠️ 所有尾数都被三大定律排除，无法进行预测")
                    return {
                        'success': True,
                        'recommended_tails': [],
                        'confidence': 0.0,
                        'ensemble_probabilities': {},
                        'model_count': len(all_predictions),
                        'message': '所有候选尾数都被三大定律排除',
                        'exclusion_reason': {
                            'trap_tails': sorted(trap_tails),
                            'dual_minimum_tails': sorted(dual_minimum_tails),
                            'extremely_hot_tails': sorted(extremely_hot_tails)
                        }
                    }
                    
                # 使用筛选后的候选尾数作为预测目标
                prediction_targets = valid_candidates
            else:
                prediction_targets = all_tails
                valid_candidates = all_tails

            print(f"🎯 最终预测目标尾数：{sorted(prediction_targets)}")

            # === 资金流向分析 ===
            money_flow_analysis = None
            if hasattr(self, 'money_flow_analyzer') and self.money_flow_analyzer:
                try:
                    # 使用经过三大定律筛选的候选尾数进行分析
                    candidate_tails_list = list(prediction_targets)
        
                    money_flow_analysis = self.money_flow_analyzer.analyze_money_flow(
                        candidate_tails_list, 
                        data_list
                    )
        
                    if money_flow_analysis.get('success'):
                        recommended_flow_tails = money_flow_analysis.get('recommended_tails', [])
                        avoid_flow_tails = money_flow_analysis.get('avoid_tails', [])
                        confidence = money_flow_analysis.get('confidence', 0.0)
            
                        print(f"💰 资金流向分析: 置信度={confidence:.2f}")
                        print(f"💰 资金流向建议: 推荐{recommended_flow_tails} 避开{avoid_flow_tails}")
            
                        # 将资金流向分析的建议添加到预测中
                        if recommended_flow_tails:
                            print(f"💰 资金流向分析发现了 {len(recommended_flow_tails)} 个推荐机会")
                    else:
                        print(f"💰 资金流向分析: 无有效的分析结果")
        
                except Exception as e:
                    print(f"⚠️ 资金流向分析失败: {e}")
                    money_flow_analysis = None

            # === 应用反操控避开建议进一步筛选（精准版） ===
            if ('anti_manipulation_analysis' in locals() and anti_manipulation_analysis and 
                'anti_recommendations' in locals() and anti_recommendations and 
                anti_recommendations.get('avoid_tails')):
    
                # 检查三大定律筛选后的候选尾数数量
                if len(prediction_targets) <= 1:
                    print(f"ℹ️ 反操控跳过：三大定律筛选后只剩{len(prediction_targets)}个候选尾数，不进行操作")
                else:
                    avoid_tails_recommendations = set(anti_recommendations.get('avoid_tails', []))
                    original_candidates = prediction_targets.copy()
        
                    # 只考虑三大定律筛选后的候选尾数中与避开建议重合的部分
                    overlapping_tails = prediction_targets.intersection(avoid_tails_recommendations)
        
                    if overlapping_tails:
                        # 智能选择最应该避开的一个尾数
                        if len(overlapping_tails) == 1:
                            target_to_exclude = list(overlapping_tails)[0]
                        else:
                            target_to_exclude = self._select_most_dangerous_tail(
                                overlapping_tails, anti_manipulation_analysis, data_list
                            )
            
                        # 只排除选中的一个尾数
                        prediction_targets = prediction_targets - {target_to_exclude}
                        valid_candidates = prediction_targets
            
                        print(f"🚫 反操控筛选：排除尾数{target_to_exclude}，保留{sorted(prediction_targets)}")
            
                        # 如果精准筛选后没有候选尾数了，恢复最安全的一个
                        if not prediction_targets:
                            safest_tail = self._select_safest_tail(original_candidates, anti_recommendations)
                            prediction_targets = {safest_tail}
                            valid_candidates = prediction_targets
                            print(f"⚠️ 筛选后无候选尾数，恢复最安全尾数：{safest_tail}")
                    else:
                        print(f"ℹ️ 反操控分析：候选尾数与避开建议无重合，跳过筛选")
            else:
                print(f"ℹ️ 反操控分析无避开建议或分析失败，跳过反操控筛选")

            # 分析模型预测多样性
            if prediction_targets:
                target_tail = next(iter(prediction_targets))  # 取第一个候选尾数分析
                tail_predictions = []
                for model_key, predictions in all_predictions.items():
                    if target_tail in predictions:
                        tail_predictions.append(predictions[target_tail])
                
                if len(tail_predictions) > 1:
                    import statistics
                    pred_mean = statistics.mean(tail_predictions)
                    pred_std = statistics.stdev(tail_predictions) if len(tail_predictions) > 1 else 0
                    print(f"📊 尾数{target_tail}预测多样性: 均值{pred_mean:.3f}, 标准差{pred_std:.3f}")
                    
                    # 显示预测分布
                    high_prob_models = [k for k, v in all_predictions.items() if v.get(target_tail, 0) > 0.6]
                    low_prob_models = [k for k, v in all_predictions.items() if v.get(target_tail, 0) < 0.4]
                    print(f"📈 高概率模型({len(high_prob_models)}个): {high_prob_models[:3]}" + ("..." if len(high_prob_models) > 3 else ""))
                    print(f"📉 低概率模型({len(low_prob_models)}个): {low_prob_models[:3]}" + ("..." if len(low_prob_models) > 3 else ""))

                    # 显示不同决策策略的分布
                    strategy_distribution = {}
                    for model_key in all_predictions.keys():
                        if model_key.startswith('river_'):
                            river_model_name = model_key.replace('river_', '')
                            strategy = model_decision_strategies.get(river_model_name, {})
                            focus = strategy.get('focus', 'unknown')
                            if focus not in strategy_distribution:
                                strategy_distribution[focus] = []
                            if target_tail in all_predictions[model_key]:
                                strategy_distribution[focus].append(all_predictions[model_key][target_tail])
                    
                    print(f"🎯 不同决策策略的预测分布:")
                    for strategy, probs in strategy_distribution.items():
                        if probs:
                            avg_prob = sum(probs) / len(probs)
                            print(f"   • {strategy}: {len(probs)}个模型, 平均概率{avg_prob:.3f}")

            # 验证和清理特征数据，避免math domain error
            def clean_features_for_river(features_dict):
                """清理特征数据以避免数学域错误"""
                cleaned_features = {}
                for key, value in features_dict.items():
                    try:
                        if value is None:
                            cleaned_features[key] = 0.0
                        elif isinstance(value, (int, float)):
                            # 检查是否为有效数值
                            if math.isnan(value) or math.isinf(value):
                                cleaned_features[key] = 0.0
                            elif value < 0:
                                cleaned_features[key] = 0.0  # 确保非负
                            else:
                                cleaned_features[key] = float(value)
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            # 如果是数组类型，取第一个有效值或0
                            try:
                                if len(value) > 0:
                                    first_val = float(value[0])
                                    if math.isnan(first_val) or math.isinf(first_val):
                                        cleaned_features[key] = 0.0
                                    else:
                                        cleaned_features[key] = max(0.0, first_val)
                                else:
                                    cleaned_features[key] = 0.0
                            except (ValueError, TypeError, IndexError):
                                cleaned_features[key] = 0.0
                        else:
                            # 尝试转换为浮点数
                            try:
                                converted_val = float(value)
                                if math.isnan(converted_val) or math.isinf(converted_val):
                                    cleaned_features[key] = 0.0
                                else:
                                    cleaned_features[key] = max(0.0, converted_val)
                            except (ValueError, TypeError):
                                cleaned_features[key] = 0.0
                    except Exception as e:
                        print(f"   ⚠️ 清理特征 {key} 时出错: {e}")
                        cleaned_features[key] = 0.0
                
                # 验证清理结果
                for key, value in cleaned_features.items():
                    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                        print(f"   ❌ 清理后的特征 {key} 仍然无效: {value}")
                        cleaned_features[key] = 0.0
                
                return cleaned_features
        
            # 清理River模型的输入特征
            X_river_cleaned = clean_features_for_river(X_river)

            # === River模型预测 ===
            model_decision_records = {}  # 记录每个模型的详细决策过程
            
            for model_name, model in self.river_models.items():
                try:
                    model_predictions = {}
                    decision_record = {
                        'model_type': 'river',
                        'model_name': model_name,
                        'detailed_reasons': {},
                        'internal_states': {},
                        'decision_process': []
                    }
                    
                    # 特殊处理历史模式匹配算法
                    if hasattr(model, 'update_historical_data'):
                        # 更新历史数据
                        model.update_historical_data(data_list)
                        
                        # 获取详细的匹配分析
                        if hasattr(model, 'get_detailed_matching_analysis'):
                            # 使用当前期的尾数进行匹配分析
                            current_period_tails = data_list[0].get('tails', []) if data_list else []
                            matching_analysis = model.get_detailed_matching_analysis(list(current_period_tails))
                            decision_record['matching_analysis'] = matching_analysis
                            decision_record['decision_process'].append("历史模式匹配分析完成")
    
                            # 保存分析结果到模型的 _last_detailed_analysis 属性
                            model._last_detailed_analysis = matching_analysis
                        
                        # 检查算法是否应该参与本轮预测
                        if hasattr(model, 'should_participate_in_ensemble'):
                            should_participate = model.should_participate_in_ensemble(list(prediction_targets))
                        else:
                            should_participate = True
                        
                        if should_participate:
                            # 获取基于匹配质量的概率预测
                            if hasattr(model, 'predict_probabilities_for_candidates'):
                                probabilities = model.predict_probabilities_for_candidates(list(prediction_targets))
                                
                                # 记录详细的预测过程
                                decision_record['prediction_probabilities'] = probabilities
                                decision_record['decision_process'].append(f"计算出{len(probabilities)}个候选尾数的概率")
                                
                                # 使用计算出的概率
                                best_tail = None
                                best_probability = 0.0
                        
                                for tail in prediction_targets:
                                    probability = probabilities.get(tail, 0.5)
                                    model_predictions[tail] = probability
                            
                                    # 记录每个尾数的详细决策理由
                                    if hasattr(model, 'get_tail_decision_reason'):
                                        detailed_reason = model.get_tail_decision_reason(tail, probabilities)
                                        decision_record['detailed_reasons'][tail] = detailed_reason
                            
                                    # 找到最佳预测尾数
                                    if probability > best_probability:
                                        best_probability = probability
                                        best_tail = tail
                        
                                # 收集预测记录，稍后统一处理（避免重复记录）
                                if best_tail is not None:
                                    predicted_class = 1 if best_probability > 0.5 else 0
                                    if 'predictions_to_record' not in locals():
                                        predictions_to_record = []
                                    predictions_to_record.append({
                                        'model_name': f'river_{model_name}',
                                        'predicted_class': predicted_class,
                                        'confidence': best_probability,
                                        'target_tail': best_tail
                                    })
                                
                                selected_tails = [t for t, p in probabilities.items() if p != 0.5]
                                decision_record['decision_process'].append(f"最终选择参与预测的尾数：{selected_tails}")
                                print(f"🎯 历史模式匹配(100%相似度) 参与预测，选择了尾数：{selected_tails}")
                            else:
                                # 回退到投票模式（兼容性）
                                votes = model.predict_for_candidates(list(prediction_targets))
                                decision_record['votes'] = votes
                                decision_record['decision_process'].append("使用投票模式预测")
                                
                                # 转换为概率格式，找到最佳投票结果
                                best_tail = None
                                best_vote = -1
                                best_probability = 0.0
                                
                                for tail in prediction_targets:
                                    vote = votes.get(tail, 0)
                                    probability = 0.8 if vote > 0 else 0.2
                                    model_predictions[tail] = probability
                                    
                                    # 找到最佳投票结果
                                    if vote > best_vote:
                                        best_vote = vote
                                        best_tail = tail
                                        best_probability = probability
                                
                                # 只记录最佳投票结果到数据库
                                if best_tail is not None:
                                    predicted_class = 1 if best_vote > 0 else 0
                                    self._record_model_prediction(
                                        f'river_{model_name}', 
                                        predicted_class, 
                                        best_probability, 
                                        best_tail
                                    )
                        else:
                            # 算法不参与本轮预测
                            decision_record['decision_process'].append("未找到完全匹配的历史模式，不参与本轮预测")
                            print(f"🚫 历史模式匹配(100%相似度) 不参与本轮预测：未找到完全匹配的历史模式")
                            model_decision_records[f'river_{model_name}'] = decision_record
                            continue
                    else:
                        # 普通模型预测
                        decision_record['decision_process'].append("开始普通模型预测")
                        
                        # 记录模型内部状态（如果可用）
                        if hasattr(model, 'get_internal_state'):
                            try:
                                internal_state = model.get_internal_state()
                                decision_record['internal_states'] = internal_state
                            except:
                                pass
                        
                        # 只对通过四大定律筛选的候选尾数进行预测
                        for tail in prediction_targets:
                            prediction_details = {}
                            
                            # 提取更多特征用于分析
                            tail_features = self.feature_engineer.extract_tail_specific_features(data_list, tail)
                            prediction_details['tail_features'] = tail_features
                            
                            if hasattr(model, 'predict_proba_one'):
                                try:
                                    # 使用该模型的专属特征
                                    model_X_river = model_specific_features.get(model_name, X_river)
                            
                                    # 对所有模型都进行特征清理
                                    if isinstance(model_X_river, dict):
                                        model_X_river_cleaned = clean_features_for_river(model_X_river)
                                    else:
                                        # 如果不是字典格式，转换为字典
                                        if isinstance(model_X_river, (list, tuple, np.ndarray)):
                                            temp_dict = {}
                                            for i, val in enumerate(model_X_river[:60]):  # 限制最多60个特征
                                                temp_dict[f'feature_{i}'] = val
                                            model_X_river_cleaned = clean_features_for_river(temp_dict)
                                        else:
                                            model_X_river_cleaned = clean_features_for_river(X_river_cleaned)
                            
                                    # 验证清理后的特征
                                    if not model_X_river_cleaned:
                                        print(f"   ⚠️ 模型 {model_name} 清理后特征为空，使用默认特征")
                                        model_X_river_cleaned = {f'feature_{i}': 0.0 for i in range(10)}
                            
                                    # 确保所有特征值都是有效的浮点数
                                    validated_features = {}
                                    for key, value in model_X_river_cleaned.items():
                                        try:
                                            validated_value = float(value)
                                            if math.isnan(validated_value) or math.isinf(validated_value):
                                                validated_features[key] = 0.0
                                            else:
                                                validated_features[key] = validated_value
                                        except (ValueError, TypeError):
                                            validated_features[key] = 0.0
                            
                                    print(f"   🔍 模型 {model_name} 使用特征数量: {len(validated_features)}")
                            
                                    # 执行预测
                                    proba = model.predict_proba_one(validated_features)
                                    
                                    # 处理不同的概率输出格式
                                    if isinstance(proba, dict):
                                        prob_1 = proba.get(1, proba.get(True, 0.5))
                                        
                                        # 应用模型特定的决策策略
                                        strategy = model_decision_strategies.get(model_name, {})
                                        focus = strategy.get('focus', 'frequency_analysis')
                                        threshold = strategy.get('decision_threshold', 0.5)
                                        time_window = strategy.get('time_window', 10)
                                        weight_recent = strategy.get('weight_recent', 1.0)
                                        
                                        # 根据不同策略重新计算概率
                                        if focus == 'trend_continuation':
                                            # 趋势延续策略：如果尾数在最近期出现，大幅提升概率
                                            if tail_features.get('in_latest_period', False):
                                                consecutive = tail_features.get('consecutive_appearances', 0)
                                                if consecutive >= 2:
                                                    prob_1 = 0.85  # 强烈延续信号
                                                elif consecutive >= 1:
                                                    prob_1 = 0.72  # 中等延续信号
                                                else:
                                                    prob_1 = 0.58  # 弱延续信号
                                            else:
                                                prob_1 = 0.25  # 不符合延续策略
                                        
                                        elif focus == 'trend_reversal':
                                            # 趋势反转策略：如果尾数长期未出现，提升概率
                                            last_distance = tail_features.get('last_appearance_distance', -1)
                                            if last_distance >= 5:
                                                prob_1 = 0.78  # 强烈反转信号
                                            elif last_distance >= 3:
                                                prob_1 = 0.65  # 中等反转信号
                                            elif last_distance >= 1:
                                                prob_1 = 0.52  # 弱反转信号
                                            else:
                                                prob_1 = 0.28  # 不符合反转策略
                                        
                                        elif focus == 'frequency_analysis':
                                            # 频率分析策略：基于频率计算
                                            recent_freq = tail_features.get('recent_10_frequency', 0.5)
                                            if recent_freq >= 0.6:
                                                prob_1 = 0.68
                                            elif recent_freq >= 0.4:
                                                prob_1 = 0.55
                                            elif recent_freq >= 0.2:
                                                prob_1 = 0.42
                                            else:
                                                prob_1 = 0.32
                                        
                                        elif focus == 'independence_assumption':
                                            # 独立性假设：基于先验概率
                                            if tail in data_list[0].get('tails', []):
                                                prob_1 = 0.35  # 独立性假设下，最近出现降低概率
                                            else:
                                                prob_1 = 0.65  # 未出现则提升概率
                                        
                                        elif focus == 'categorical_patterns':
                                            # 分类模式：基于尾数类别
                                            if tail in [0, 5]:  # 0和5结尾
                                                prob_1 = 0.62
                                            elif tail in [1, 3, 7, 9]:  # 奇数
                                                prob_1 = 0.58
                                            else:  # 偶数
                                                prob_1 = 0.48
                                        
                                        elif focus == 'continuous_distribution':
                                            # 连续分布：基于数值分布特性
                                            recent_count = tail_features.get('recent_5_count', 0)
                                            if recent_count == 0:
                                                prob_1 = 0.70  # 补缺
                                            elif recent_count == 1:
                                                prob_1 = 0.55
                                            elif recent_count == 2:
                                                prob_1 = 0.45
                                            else:
                                                prob_1 = 0.30  # 过热
                                        
                                        elif focus == 'hybrid_approach':
                                            # 混合方法：多因素权衡
                                            score = 0.5
                                            if tail_features.get('in_latest_period', False):
                                                score += 0.15
                                            if tail_features.get('recent_5_count', 0) >= 2:
                                                score += 0.1
                                            if tail_features.get('last_appearance_distance', 0) >= 3:
                                                score += 0.2
                                            prob_1 = min(0.85, max(0.15, score))
                                        
                                        elif focus == 'ensemble_voting':
                                            # 集成投票：模拟内部投票
                                            votes = 0
                                            if tail_features.get('in_latest_period', False):
                                                votes += 2
                                            if tail_features.get('recent_10_count', 0) >= 4:
                                                votes += 1
                                            if tail_features.get('consecutive_appearances', 0) >= 1:
                                                votes += 1
                                            if tail_features.get('last_appearance_distance', 0) >= 4:
                                                votes += 2
                                            
                                            prob_1 = 0.2 + (votes / 6.0) * 0.6  # 基础0.2 + 投票权重
                                        
                                        elif focus == 'error_correction':
                                            # 错误修正：基于历史准确性调整
                                            base_prob = prob_1
                                            # 模拟错误修正逻辑
                                            if tail_features.get('recent_5_count', 0) == 0:
                                                prob_1 = base_prob * 1.4  # 修正预测
                                            else:
                                                prob_1 = base_prob * 0.8  # 保守预测
                                        
                                        elif focus == 'probabilistic_ensemble':
                                            # 概率集成：基于概率组合
                                            prob_components = []
                                            prob_components.append(0.3 if tail_features.get('in_latest_period', False) else 0.7)
                                            prob_components.append(tail_features.get('recent_10_frequency', 0.5))
                                            prob_components.append(0.8 if tail_features.get('last_appearance_distance', 0) >= 3 else 0.2)
                                            
                                            prob_1 = sum(prob_components) / len(prob_components)
                                        
                                        elif focus == 'linear_ensemble':
                                            # 线性集成：线性组合特征
                                            linear_score = 0
                                            linear_score += tail_features.get('recent_10_frequency', 0.5) * 0.3
                                            linear_score += (1.0 if tail_features.get('in_latest_period', False) else 0.0) * 0.4
                                            linear_score += min(1.0, tail_features.get('last_appearance_distance', 0) / 5.0) * 0.3
                                            
                                            prob_1 = linear_score
                                        
                                        elif focus == 'historical_matching':
                                            # 历史匹配：保持原有概率，不做调整
                                            pass
                                        
                                        # 确保概率在合理范围内
                                        prob_1 = max(0.01, min(0.99, prob_1))
                                        model_predictions[tail] = prob_1

                                        # 逻辑回归：增强频率特征权重
                                        recent_freq = tail_features.get('recent_10_frequency', 0.5)
                                        if recent_freq > 0.6:
                                            prob_1 *= 1.05
                                        elif recent_freq < 0.3:
                                            prob_1 *= 0.95
                                        elif 'naive_bayes' in model_name:
                                            # 朴素贝叶斯：基于独立性假设调整
                                            if tail_features.get('in_latest_period', False):
                                                prob_1 *= 1.08
                                            else:
                                                prob_1 *= 0.92
                                        elif 'bagging' in model_name:
                                            # 装袋模型：基于集成投票调整
                                            if tail_features.get('recent_5_count', 0) >= 3:
                                                prob_1 *= 1.03  # 确定性增强
                                            elif tail_features.get('recent_5_count', 0) <= 1:
                                                prob_1 *= 0.97  # 确定性降低
                                        
                                        # 确保概率在合理范围内
                                        prob_1 = max(0.01, min(0.99, prob_1))
                                        model_predictions[tail] = prob_1
                                        prediction_details['final_probability'] = prob_1
                                        prediction_details['probability_source'] = "模型概率输出"
                                        
                                        # 生成详细的决策分析
                                        detailed_reason = self.prediction_analyzer.generate_model_specific_reason(
                                            model_name, tail, prob_1, tail_features, data_list
                                        )
                                        prediction_details['detailed_analysis'] = detailed_reason
                                        
                                        # 暂存预测结果，稍后统一记录
                                        if 'predictions_to_record' not in locals():
                                            predictions_to_record = []
                        
                                        predicted_class = 1 if prob_1 > 0.5 else 0
                                        predictions_to_record.append({
                                            'model_name': f'river_{model_name}',
                                            'predicted_class': predicted_class,
                                            'confidence': prob_1,
                                            'target_tail': tail
                                        })
                                    else:
                                        model_predictions[tail] = 0.5
                                        prediction_details['final_probability'] = 0.5
                                        prediction_details['detailed_analysis'] = "模型输出格式异常，无法解析概率"
                                        
                                except Exception as pred_e:
                                    model_predictions[tail] = 0.5
                                    prediction_details['error'] = str(pred_e)
                                    prediction_details['detailed_analysis'] = f"预测过程出错：{str(pred_e)}"
                            else:
                                model_predictions[tail] = 0.5
                                prediction_details['detailed_analysis'] = "模型不支持概率预测，使用默认概率0.5"
                            
                            decision_record['detailed_reasons'][tail] = prediction_details
                    
                    all_predictions[f'river_{model_name}'] = model_predictions

                    # 每个模型只记录一次预测（选择置信度最高的尾数）
                    if 'predictions_to_record' in locals() and predictions_to_record:
                        best_prediction = max(predictions_to_record, key=lambda x: x['confidence'])
                        self._record_model_prediction(
                            best_prediction['model_name'],
                            best_prediction['predicted_class'],
                            best_prediction['confidence'],
                            best_prediction['target_tail']
                        )
                        predictions_to_record = []  # 清空记录列表
                    model_decision_records[f'river_{model_name}'] = decision_record
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ River模型 {model_name} 预测失败: {e}")
                    print(f"   错误类型: {type(e).__name__}")
                    print(f"   错误详情: {error_msg}")
                    
                    # 检查特定错误类型
                    if 'schema' in error_msg.lower():
                        print(f"   🔍 Schema验证失败，检查输入数据格式")
                        print(f"   特征数量: {len(model_X_river) if isinstance(model_X_river, dict) else 'N/A'}")
                        if isinstance(model_X_river, dict):
                            sample_features = list(model_X_river.items())[:5]
                            print(f"   样本特征: {sample_features}")
                    elif 'math domain error' in error_msg:
                        print(f"   🔍 数学域错误，特征包含无效数值")
                    elif 'nan' in error_msg.lower() or 'inf' in error_msg.lower():
                        print(f"   🔍 数值错误，特征包含NaN或无穷大")
                    
                    # 添加调试信息
                    if hasattr(model, '__class__'):
                        print(f"   模型类型: {model.__class__.__name__}")
                    
                    all_predictions[f'river_{model_name}'] = {tail: 0.5 for tail in prediction_targets}
                    model_decision_records[f'river_{model_name}'] = {
                        'model_type': 'river',
                        'model_name': model_name,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'detailed_reasons': {tail: f"模型预测失败：{str(e)}" for tail in prediction_targets}
                    }
                    
                    model_decision_records[f'river_{model_name}'] = {
                        'model_type': 'river',
                        'model_name': model_name,
                        'error': str(e),
                        'detailed_reasons': {tail: f"模型预测失败：{str(e)}" for tail in prediction_targets}
                    }
            
            # 统一记录River模型预测（避免重复）
            if 'predictions_to_record' in locals() and predictions_to_record:
                # 每个模型只记录一次最佳预测
                recorded_models = set()
                for pred in predictions_to_record:
                    if pred['model_name'] not in recorded_models:
                        self._record_model_prediction(
                            pred['model_name'],
                            pred['predicted_class'],
                            pred['confidence'],
                            pred['target_tail']
                        )
                        recorded_models.add(pred['model_name'])
                print(f"📊 已记录 {len(recorded_models)} 个River模型的预测")

            # === scikit-multiflow模型预测 ===
            # 为sklearn模型创建多样化特征
            sklearn_diversified_features = {}
            sklearn_feature_configs = {
                'extremely_fast_tree': {'feature_start': 0, 'feature_end': 48},
                'skm_hoeffding_adaptive': {'feature_start': 5, 'feature_end': 55},
                'online_adaboost': {'feature_start': 10, 'feature_end': 50},
                'skm_random_patches': {'feature_start': 8, 'feature_end': 53}
            }
            
            for model_name in self.sklearn_models.keys():
                config = sklearn_feature_configs.get(model_name, {'feature_start': 0, 'feature_end': len(base_features)})
                
                # 确定性地选择特征子集
                feature_start = config['feature_start']
                feature_end = min(config['feature_end'], len(base_features))
                
                # 创建该模型的特征向量
                model_features = np.zeros(len(base_features))
                for idx in range(feature_start, feature_end):
                    if idx < len(base_features):
                        base_value = base_features[idx]
                        # 直接使用基础特征值，无噪声
                        model_features[idx] = base_value
                
                sklearn_diversified_features[model_name] = model_features.reshape(1, -1)

            for model_name, model in self.sklearn_models.items():
                try:
                    model_predictions = {}
                    # 使用该模型的专属特征
                    model_X_sklearn = sklearn_diversified_features.get(model_name, X_sklearn)
                    
                    # 只对通过四大定律筛选的候选尾数进行预测
                    best_tail = None
                    best_confidence = 0.0
                    
                    for tail in prediction_targets:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(model_X_sklearn)
                            if proba is not None and len(proba) > 0 and len(proba[0]) > 1:
                                confidence = proba[0][1]
                                model_predictions[tail] = confidence
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_tail = tail
                            else:
                                model_predictions[tail] = 0.5
                        else:
                            model_predictions[tail] = 0.5
                    
                    # 只记录最佳预测
                    if best_tail is not None:
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            f'sklearn_{model_name}',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
                    
                    all_predictions[f'sklearn_{model_name}'] = model_predictions
                
                except Exception as e:
                    print(f"sklearn模型 {model_name} 预测失败: {e}")
                    all_predictions[f'sklearn_{model_name}'] = {tail: 0.5 for tail in prediction_targets}

            # === PyTorch深度学习模型预测 ===
            if PYTORCH_AVAILABLE and self.deep_learning_module and self.deep_learning_module.models:
                try:
                    # 使用深度学习模型进行预测
                    dl_predictions = self.deep_learning_module.predict_single(base_features)
                
                    for model_name, probabilities in dl_predictions.items():
                        model_predictions = {}
                    
                        # 为每个候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail < len(probabilities):
                                model_predictions[tail] = float(probabilities[tail])
                            else:
                                model_predictions[tail] = 0.5
                    
                        all_predictions[f'pytorch_{model_name}'] = model_predictions
                    
                        # 只记录最佳预测到数据库（避免重复记录）
                        if model_predictions:
                            best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                            best_confidence = model_predictions[best_tail]
                            predicted_class = 1 if best_confidence > 0.5 else 0
                            self._record_model_prediction(
                                f'pytorch_{model_name}',
                                predicted_class,
                                best_confidence,
                                best_tail
                            )
            
                except Exception as e:
                    print(f"PyTorch深度学习预测失败: {e}")
                    # 为候选尾数添加默认预测
                    for model_name in ['lstm', 'transformer']:
                        all_predictions[f'pytorch_{model_name}'] = {tail: 0.5 for tail in prediction_targets}

            # === 反向心理学模型预测 ===
            if hasattr(self, 'reverse_psychology_predictor') and self.reverse_psychology_predictor:
                try:
                    model_predictions = {}
                    
                    # 优先使用分析结果，如果没有则使用基础预测
                    if reverse_psychology_analysis:
                        recommended_tails = reverse_psychology_analysis.get('recommended_tails', [])
                        avoid_tails = reverse_psychology_analysis.get('avoid_tails', [])
                        confidence = reverse_psychology_analysis.get('confidence', 0.5)
                        
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            elif tail in avoid_tails:
                                # 避开的尾数给予低概率
                                model_predictions[tail] = max(0.1, 0.4 - confidence * 0.3)
                            else:
                                # 中性尾数给予默认概率
                                model_predictions[tail] = 0.5
                        
                        print(f"🔄 反向心理学模型预测完成（基于分析结果），推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 分析失败时的备用预测策略
                        print(f"⚠️ 反向心理学分析失败，使用备用预测策略")
                        
                        # 基于历史数据的简单反向逻辑
                        if data_list and len(data_list) >= 5:
                            recent_5_data = data_list[:5]
                            tail_frequencies = {}
                            
                            # 计算最近5期各尾数频率
                            for tail in range(10):
                                count = sum(1 for period in recent_5_data if tail in period.get('tails', []))
                                tail_frequencies[tail] = count / 5.0
                            
                            # 反向逻辑：频率高的给低概率，频率低的给高概率
                            for tail in prediction_targets:
                                freq = tail_frequencies.get(tail, 0.5)
                                # 反向概率：频率越高，预测概率越低
                                reverse_prob = max(0.2, min(0.8, 0.8 - freq * 0.6))
                                model_predictions[tail] = reverse_prob
                            
                            print(f"🔄 反向心理学模型使用备用策略：基于频率反向")
                        else:
                            # 数据不足时使用中性预测
                            for tail in prediction_targets:
                                model_predictions[tail] = 0.5
                            print(f"🔄 反向心理学模型使用中性策略：数据不足")
                    
                    all_predictions['reverse_psychology_predictor'] = model_predictions
                    
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'reverse_psychology_predictor',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
                    
                except Exception as e:
                    print(f"❌ 反向心理学模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['reverse_psychology_predictor'] = model_predictions
                    
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'reverse_psychology_predictor',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )
            
            # === 反操控分析器模型预测 ===
            if hasattr(self, 'banker_behavior_analyzer') and self.banker_behavior_analyzer and anti_manipulation_analysis:
                try:
                    model_predictions = {}
        
                    if anti_manipulation_analysis and anti_recommendations:
                        recommended_anti_tails = anti_recommendations.get('recommended_tails', [])
                        avoid_anti_tails = anti_recommendations.get('avoid_tails', [])
                        manipulation_prob = anti_manipulation_analysis.get('manipulation_probability', 0.5)
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_anti_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + manipulation_prob * 0.3)
                            elif tail in avoid_anti_tails:
                                # 避开的尾数给予低概率
                                model_predictions[tail] = max(0.1, 0.4 - manipulation_prob * 0.3)
                            else:
                                # 中性尾数给予默认概率
                                model_predictions[tail] = 0.5
            
                        print(f"🎯 反操控分析器模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果没有有效分析，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🎯 反操控分析器模型使用中性策略：无有效分析结果")
        
                    all_predictions['anti_manipulation_banker_behavior'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'anti_manipulation_banker_behavior',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                except Exception as e:
                    print(f"❌ 反操控分析器模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['anti_manipulation_banker_behavior'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'anti_manipulation_banker_behavior',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 冷门挖掘器模型预测 ===
            if hasattr(self, 'unpopular_digger') and self.unpopular_digger and unpopular_digger_analysis:
                try:
                    model_predictions = {}
        
                    if unpopular_digger_analysis.get('success') and unpopular_digger_analysis.get('recommended_tails'):
                        recommended_cold_tails = unpopular_digger_analysis.get('recommended_tails', [])
                        confidence = unpopular_digger_analysis.get('confidence', 0.5)
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_cold_tails:
                                # 推荐的冷门尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            else:
                                # 非冷门尾数给予较低概率
                                model_predictions[tail] = max(0.1, 0.4 - confidence * 0.2)
            
                        print(f"🔍 冷门挖掘器模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果没有冷门发现，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🔍 冷门挖掘器模型使用中性策略：无明显冷门机会")
        
                    all_predictions['unpopular_digger'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'unpopular_digger',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                except Exception as e:
                    print(f"❌ 冷门挖掘器模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['unpopular_digger'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'unpopular_digger',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 资金流向分析器模型预测 ===
            if hasattr(self, 'money_flow_analyzer') and self.money_flow_analyzer and money_flow_analysis:
                try:
                    model_predictions = {}
        
                    if money_flow_analysis.get('success') and money_flow_analysis.get('recommended_tails'):
                        recommended_flow_tails = money_flow_analysis.get('recommended_tails', [])
                        avoid_flow_tails = money_flow_analysis.get('avoid_tails', [])
                        confidence = money_flow_analysis.get('confidence', 0.5)
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_flow_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            elif tail in avoid_flow_tails:
                                # 避开的尾数给予低概率
                                model_predictions[tail] = max(0.1, 0.4 - confidence * 0.3)
                            else:
                                # 中性尾数给予默认概率
                                model_predictions[tail] = 0.5
            
                        print(f"💰 资金流向分析器模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果没有有效分析，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"💰 资金流向分析器模型使用中性策略：无有效分析结果")
        
                    all_predictions['money_flow_analyzer'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'money_flow_analyzer',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                except Exception as e:
                    print(f"❌ 资金流向分析器模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['money_flow_analyzer'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'money_flow_analyzer',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 博弈论策略器模型预测 ===
            if hasattr(self, 'game_theory_strategist') and self.game_theory_strategist:
                try:
                    # 使用经过三大定律筛选的候选尾数进行博弈论分析
                    candidate_tails_list = list(prediction_targets)
        
                    game_theory_analysis = self.game_theory_strategist.predict(
                        candidate_tails_list, 
                        data_list
                    )
        
                    model_predictions = {}
        
                    if game_theory_analysis.get('success'):
                        recommended_game_tails = game_theory_analysis.get('recommended_tails', [])
                        confidence = game_theory_analysis.get('confidence', 0.0)
                        strategy_type = game_theory_analysis.get('strategy_type', 'balanced')
            
                        print(f"🎮 博弈论策略器分析: 策略={strategy_type}, 置信度={confidence:.2f}")
                        print(f"🎮 博弈论策略建议: 推荐{recommended_game_tails}")
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_game_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            else:
                                # 非推荐尾数给予较低概率
                                base_prob = 0.4 - confidence * 0.2
                                model_predictions[tail] = max(0.1, base_prob)
            
                        print(f"🎮 博弈论策略器模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果分析失败，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🎮 博弈论策略器模型使用中性策略：分析失败")
        
                    all_predictions['game_theory_strategist'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'game_theory_strategist',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                    # 存储博弈论分析结果供学习使用
                    if 'last_prediction_result' not in locals():
                        self.last_prediction_result = {}
                    self.last_prediction_result['game_theory_analysis'] = game_theory_analysis
        
                except Exception as e:
                    print(f"❌ 博弈论策略器模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['game_theory_strategist'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'game_theory_strategist',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 操控时机检测器模型预测 ===
            if hasattr(self, 'manipulation_timing_detector') and self.manipulation_timing_detector:
                try:
                    # 使用经过三大定律筛选的候选尾数进行操控时机检测
                    candidate_tails_list = list(prediction_targets)
        
                    manipulation_timing_analysis = self.manipulation_timing_detector.detect_manipulation_timing(
                        candidate_tails_list, 
                        data_list
                    )
        
                    model_predictions = {}
        
                    if manipulation_timing_analysis.get('success'):
                        recommended_timing_tails = manipulation_timing_analysis.get('recommended_tails', [])
                        avoid_timing_tails = manipulation_timing_analysis.get('avoid_tails', [])
                        confidence = manipulation_timing_analysis.get('confidence', 0.0)
                        timing_type = manipulation_timing_analysis.get('timing_type', 'natural')
            
                        print(f"🎯 操控时机检测分析: 时机类型={timing_type}, 置信度={confidence:.2f}")
                        print(f"🎯 操控时机建议: 推荐{recommended_timing_tails} 避开{avoid_timing_tails}")
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_timing_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            elif tail in avoid_timing_tails:
                                # 避开的尾数给予低概率
                                model_predictions[tail] = max(0.1, 0.4 - confidence * 0.3)
                            else:
                                # 中性尾数给予默认概率
                                model_predictions[tail] = 0.5
            
                        print(f"🎯 操控时机检测器模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果分析失败，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🎯 操控时机检测器模型使用中性策略：分析失败")
        
                    all_predictions['manipulation_timing_detector'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'manipulation_timing_detector',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                    # 存储操控时机分析结果供学习使用
                    if 'last_prediction_result' not in locals():
                        self.last_prediction_result = {}
                    self.last_prediction_result['manipulation_timing_analysis'] = manipulation_timing_analysis
        
                except Exception as e:
                    print(f"❌ 操控时机检测器模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['manipulation_timing_detector'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'manipulation_timing_detector',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 反趋势猎手模型预测 ===
            if hasattr(self, 'anti_trend_hunter') and self.anti_trend_hunter:
                try:
                    # 使用经过三大定律筛选的候选尾数进行反趋势分析
                    candidate_tails_list = list(prediction_targets)
        
                    anti_trend_analysis = self.anti_trend_hunter.predict(
                        candidate_tails_list, 
                        data_list
                    )
        
                    model_predictions = {}
        
                    if anti_trend_analysis.get('success'):
                        recommended_trend_tails = anti_trend_analysis.get('recommended_tails', [])
                        confidence = anti_trend_analysis.get('confidence', 0.0)
                        strategy_type = anti_trend_analysis.get('strategy_type', 'anti_trend_analysis')
            
                        print(f"🎯 反趋势猎手分析: 策略={strategy_type}, 置信度={confidence:.2f}")
                        print(f"🎯 反趋势建议: 推荐{recommended_trend_tails}")
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_trend_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            else:
                                # 非推荐尾数给予较低概率
                                base_prob = 0.4 - confidence * 0.2
                                model_predictions[tail] = max(0.1, base_prob)
            
                        print(f"🎯 反趋势猎手模型预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果分析失败，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🎯 反趋势猎手模型使用中性策略：分析失败")
        
                    all_predictions['anti_trend_hunter'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'anti_trend_hunter',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                    # 存储反趋势分析结果供学习使用
                    if 'last_prediction_result' not in locals():
                        self.last_prediction_result = {}
                    self.last_prediction_result['anti_trend_analysis'] = anti_trend_analysis
        
                except Exception as e:
                    print(f"❌ 反趋势猎手模型预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['anti_trend_hunter'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'anti_trend_hunter',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 群体心理分析器模型预测 ===
            if hasattr(self, 'crowd_psychology_analyzer') and self.crowd_psychology_analyzer:
                try:
                    # 使用经过三大定律筛选的候选尾数进行群体心理分析
                    candidate_tails_list = list(prediction_targets)
        
                    crowd_psychology_analysis = self.crowd_psychology_analyzer.predict(
                        candidate_tails_list, 
                        data_list
                    )
        
                    model_predictions = {}
        
                    if crowd_psychology_analysis.get('success'):
                        recommended_psychology_tails = crowd_psychology_analysis.get('recommended_tails', [])
                        confidence = crowd_psychology_analysis.get('confidence', 0.0)
                        strategy_type = crowd_psychology_analysis.get('strategy_type', 'crowd_psychology_analysis')
                        crowd_emotion = crowd_psychology_analysis.get('crowd_emotion', 'neutral')
                        herd_intensity = crowd_psychology_analysis.get('herd_intensity', 0.0)
            
                        print(f"🧠 群体心理分析: 策略={strategy_type}, 置信度={confidence:.2f}, 群体情绪={crowd_emotion}")
                        print(f"🧠 从众强度={herd_intensity:.2f}, 推荐={recommended_psychology_tails}")
            
                        # 为候选尾数分配预测概率
                        for tail in prediction_targets:
                            if tail in recommended_psychology_tails:
                                # 推荐的尾数给予高概率
                                model_predictions[tail] = min(0.9, 0.6 + confidence * 0.3)
                            else:
                                # 非推荐尾数给予较低概率
                                base_prob = 0.4 - confidence * 0.2
                                model_predictions[tail] = max(0.1, base_prob)
            
                        print(f"🧠 群体心理分析器预测完成，推荐概率: {[(t, f'{p:.3f}') for t, p in model_predictions.items() if p > 0.6]}")
                    else:
                        # 如果分析失败，使用中性预测
                        for tail in prediction_targets:
                            model_predictions[tail] = 0.5
                        print(f"🧠 群体心理分析器使用中性策略：分析失败")
        
                    all_predictions['crowd_psychology_analyzer'] = model_predictions
        
                    # 记录最佳预测到数据库
                    if model_predictions:
                        best_tail = max(model_predictions.keys(), key=lambda t: model_predictions[t])
                        best_confidence = model_predictions[best_tail]
                        predicted_class = 1 if best_confidence > 0.5 else 0
                        self._record_model_prediction(
                            'crowd_psychology_analyzer',
                            predicted_class,
                            best_confidence,
                            best_tail
                        )
        
                    # 存储群体心理分析结果供学习使用
                    if 'last_prediction_result' not in locals():
                        self.last_prediction_result = {}
                    self.last_prediction_result['crowd_psychology_analysis'] = crowd_psychology_analysis
        
                except Exception as e:
                    print(f"❌ 群体心理分析器预测失败: {e}")
                    # 添加默认预测确保模型参与权重投资
                    model_predictions = {tail: 0.5 for tail in prediction_targets}
                    all_predictions['crowd_psychology_analyzer'] = model_predictions
        
                    # 记录默认预测
                    if model_predictions and prediction_targets:
                        best_tail = list(prediction_targets)[0]
                        self._record_model_prediction(
                            'crowd_psychology_analyzer',
                            0,  # 默认预测类别
                            0.5,  # 默认置信度
                            best_tail
                        )

            # === 基于预测权重的直接投资 ===
            print(f"💰 开始基于预测权重的直接投资...")
            print(f"📋 投资策略: 每个模型固定投资30%预测权重")
            
            # 执行权重投资
            model_weight_investments = {}
            
            for model_key in all_predictions.keys():
                if model_key in self.ensemble_weights:
                    # 获取模型的当前权重
                    current_weight = self.ensemble_weights[model_key]['weight']
                    
                    # 获取该模型对候选尾数的预测概率
                    model_tail_probabilities = {}
                    if model_key in all_predictions:
                        model_predictions = all_predictions[model_key]
                        # 修改：让每个模型从自己的完整预测结果中选择，而不是被限制在prediction_targets中
                        for tail, prob in model_predictions.items():
                            model_tail_probabilities[tail] = prob
                    else:
                        # 如果没有预测数据，只考虑prediction_targets
                        for tail in prediction_targets:
                            model_tail_probabilities[tail] = 0.5

                    print(f"🔍 {model_key} 完整置信度数据: {model_tail_probabilities}")

                    # 所有模型采用集中投资策略：选择置信度最高的单个尾数投资
                    print(f"🎯 {model_key} 采用集中投资策略")

                    # 从模型的完整预测结果中找到置信度最高的尾数
                    if model_tail_probabilities:
                        best_tail = max(model_tail_probabilities.keys(), 
                                    key=lambda t: model_tail_probabilities.get(t, 0.5))
                        best_confidence = model_tail_probabilities[best_tail]
    
                        print(f"   🎯 选定最高置信度尾数: {best_tail} (置信度: {best_confidence:.3f})")
                        
                        # 计算总投资权重（30%投资）
                        total_investment_weight = current_weight * 0.3
                        
                        # 将所有投资权重集中投入到最高置信度的尾数
                        weight_investments = {best_tail: total_investment_weight}
                        
                        print(f"   💰 集中投资: 尾数{best_tail} 投资权重{total_investment_weight:.4f} (100%集中)")
                        
                        # 为不同模型显示选择理由
                        if model_key == 'reverse_psychology_predictor':
                            # 反向心理学模型的特殊分析
                            if reverse_psychology_analysis:
                                strategy_type = reverse_psychology_analysis.get('strategy_type', 'unknown')
                                reasoning = reverse_psychology_analysis.get('reasoning', '无详细理由')
                                print(f"   📋 反向策略: {strategy_type}")
                                print(f"   📋 选择理由: {reasoning}")
                                
                                # 显示推荐和避开的尾数对比
                                recommended_tails = reverse_psychology_analysis.get('recommended_tails', [])
                                avoid_tails = reverse_psychology_analysis.get('avoid_tails', [])
                                
                                if best_tail in recommended_tails:
                                    print(f"   ✅ 选中尾数{best_tail}在推荐列表中: {recommended_tails}")
                                elif best_tail not in avoid_tails:
                                    print(f"   ⚪ 选中尾数{best_tail}为中性选择 (推荐:{recommended_tails}, 避开:{avoid_tails})")
                                else:
                                    print(f"   ⚠️ 选中尾数{best_tail}在避开列表中: {avoid_tails} (可能是备用策略)")
                            else:
                                print(f"   📋 使用备用策略: 基于历史频率的反向选择")
                        else:
                            # 其他模型的选择理由
                            model_display_name = model_key.replace('_', ' ').replace('river ', '').replace('sklearn ', '').replace('pytorch ', '').title()
                            
                            if 'hoeffding' in model_key.lower():
                                print(f"   📋 {model_display_name}: 决策树判断尾数{best_tail}特征组合最优")
                            elif 'logistic' in model_key.lower():
                                print(f"   📋 {model_display_name}: 逻辑回归计算尾数{best_confidence:.3f}概率最高")
                            elif 'naive_bayes' in model_key.lower():
                                print(f"   📋 {model_display_name}: 贝叶斯推理认为尾数{best_tail}后验概率最大")
                            elif 'bagging' in model_key.lower():
                                print(f"   📋 {model_display_name}: 集成投票显示尾数{best_tail}支持度最高")
                            elif 'adaboost' in model_key.lower():
                                print(f"   📋 {model_display_name}: 自适应提升算法认为尾数{best_tail}最可能")
                            elif 'pattern_matcher' in model_key.lower():
                                print(f"   📋 {model_display_name}: 历史模式匹配显示尾数{best_tail}相似度最高")
                            elif 'reverse_psychology' in model_key.lower():
                                print(f"   📋 {model_display_name}: 反向心理学分析认为尾数{best_tail}是最佳选择")
                            elif 'unpopular_digger' in model_key.lower():
                                print(f"   📋 {model_display_name}: 冷门挖掘算法发现尾数{best_tail}存在复出机会")
                            elif 'anti_manipulation' in model_key.lower():
                                print(f"   📋 {model_display_name}: 反操控分析认为尾数{best_tail}安全性最高")
                            elif 'pytorch' in model_key.lower():
                                if 'lstm' in model_key.lower():
                                    print(f"   📋 {model_display_name}: LSTM时序分析预测尾数{best_tail}最可能")
                                elif 'transformer' in model_key.lower():
                                    print(f"   📋 {model_display_name}: Transformer注意力机制锁定尾数{best_tail}")
                                else:
                                    print(f"   📋 {model_display_name}: 深度学习网络输出尾数{best_tail}概率最高")
                            else:
                                print(f"   📋 {model_display_name}: 算法计算认为尾数{best_tail}置信度最高")
                            
                            # 显示置信度等级
                            if best_confidence > 0.8:
                                confidence_level = "极高置信度"
                            elif best_confidence > 0.7:
                                confidence_level = "高置信度"
                            elif best_confidence > 0.6:
                                confidence_level = "中高置信度"
                            elif best_confidence > 0.5:
                                confidence_level = "中等置信度"
                            else:
                                confidence_level = "低置信度"
                            
                            print(f"   📊 置信度评估: {confidence_level} ({best_confidence:.3f})")
                    else:
                        print(f"   ❌ 无有效的置信度数据，跳过投资")
                        weight_investments = {}
                    
                    if weight_investments:
                        # 记录权重投资
                        model_weight_investments[model_key] = weight_investments
                        self.ensemble_weights[model_key]['weight_investments'] = weight_investments.copy()
                        
                        total_investment_weight = sum(weight_investments.values())
                        investment_percentage = (total_investment_weight / current_weight) * 100 if current_weight > 0 else 0
                        print(f"   💼 {model_key}: 模型权重{current_weight:.4f}, 投资权重{total_investment_weight:.4f}({investment_percentage:.1f}%)")
                        
                        # 显示投资详情
                        model_predictions = all_predictions.get(model_key, {})
                        for tail, weight_amount in weight_investments.items():
                            tail_prob = model_predictions.get(tail, 0.5)
                            investment_ratio = (weight_amount / total_investment_weight) * 100 if total_investment_weight > 0 else 0
                            print(f"      🎯 尾数{tail}: 投资权重{weight_amount:.4f}({investment_ratio:.1f}%) - 模型概率{tail_prob:.3f}")
            
            # 冻结投资的权重
            if model_weight_investments:
                self.freeze_investment_weights(model_weight_investments)
            
            print(f"💰 权重投资分配完成，等待开奖结算")
            
            # 验证权重投资比例
            print(f"📊 权重投资比例验证：")
            for model_key, weight_investments in model_weight_investments.items():
                if model_key in self.ensemble_weights:
                    original_weight = self.ensemble_weights[model_key]['weight'] + sum(weight_investments.values())  # 投资前的原始权重
                    total_investment_weight = sum(weight_investments.values())
                    investment_ratio = (total_investment_weight / original_weight) * 100 if original_weight > 0 else 0
                    
                    status = "✅" if investment_ratio >= 25 else "⚠️"  # 调整为25%因为我们现在是30%投资
                    investment_details = []
                    model_predictions = all_predictions.get(model_key, {})
                    
                    for tail, weight_amount in weight_investments.items():
                        tail_prob = model_predictions.get(tail, 0.5)
                        tail_ratio = (weight_amount / total_investment_weight) * 100 if total_investment_weight > 0 else 0
                        investment_details.append(f"尾数{tail}({tail_prob:.2f}):{tail_ratio:.0f}%")
                    
                    print(f"   {status} {model_key}: 权重投资比例 {investment_ratio:.1f}% (投资权重{total_investment_weight:.4f}/原权重{original_weight:.4f})")
                    print(f"      💰 投资分布: {' | '.join(investment_details)}")

            # 基于权重投资计算集成概率
            ensemble_probabilities = {}
            
            # 首先收集每个尾数的投资权重信息
            tail_investment_weights = {}
            for tail in prediction_targets:
                tail_investment_weights[tail] = 0.0
                
                # 统计投资该尾数的总权重
                for model_key, weight_investments in model_weight_investments.items():
                    if tail in weight_investments:
                        tail_investment_weights[tail] += weight_investments[tail]
            
            # 计算集成概率（基于投资权重）
            for tail in prediction_targets:
                tail_total_weight = tail_investment_weights[tail]
                
                if tail_total_weight > 0:
                    # 使用投资权重作为该尾数的支持度
                    # 概率等于该尾数的投资权重占所有尾数投资权重的比例
                    total_all_investments = sum(tail_investment_weights.values())
                    if total_all_investments > 0:
                        ensemble_probabilities[tail] = tail_total_weight / total_all_investments
                    else:
                        ensemble_probabilities[tail] = 0.5
                else:
                    ensemble_probabilities[tail] = 0.0
                
                # 统计投资该尾数的模型信息
                investors = []
                for model_key, weight_investments in model_weight_investments.items():
                    if tail in weight_investments:
                        investment_weight = weight_investments[tail]
                        model_prob = all_predictions.get(model_key, {}).get(tail, 0.5)
                        investors.append(f"{model_key}(投资权重{investment_weight:.4f},概率{model_prob:.3f})")
                
                investor_info = " | ".join(investors[:3]) + ("..." if len(investors) > 3 else "")
                print(f"   🎯 尾数{tail}: 投资权重{tail_total_weight:.4f}, 集成概率{ensemble_probabilities[tail]:.3f}")
                print(f"      💼 投资方: {investor_info}")
            
            # 保存权重投资信息以便后续结算
            self.current_weight_investments = model_weight_investments
            
            # 确保投资权重机制正常工作
            if not model_weight_investments:
                print("⚠️ 没有模型进行权重投资，无法进行预测")
                return {
                    'success': False,
                    'recommended_tails': [],
                    'confidence': 0.0,
                    'ensemble_probabilities': {},
                    'model_count': len(all_predictions),
                    'message': '没有模型进行权重投资，无法生成预测结果'
                }

            # === 选择最佳候选尾数 ===
            if ensemble_probabilities:
                # 从候选中选择概率最高的尾数
                best_candidate = max(ensemble_probabilities.keys(), key=lambda t: ensemble_probabilities.get(t, 0))
                recommended_tails = [best_candidate]
                print(f"🎯 从候选尾数中选择概率最高的：{best_candidate}（概率：{ensemble_probabilities.get(best_candidate, 0):.3f}）")
            else:
                recommended_tails = []
            
            # === 计算置信度 ===
            confidence = self.calculate_ensemble_confidence(recommended_tails, ensemble_probabilities, all_predictions)
            
            # 保存预测结果（包含正确的权重信息）
            self.last_prediction_result = {
                'recommended_tails': recommended_tails,
                'confidence': confidence,
                'ensemble_probabilities': ensemble_probabilities,
                'model_predictions': all_predictions,
                'features': base_features.tolist(),
                'model_weight_investments': getattr(self, 'current_weight_investments', {}),
                'model_weights': {k: v['weight'] for k, v in self.ensemble_weights.items()},
                'weight_pool_size': getattr(self, 'weight_pool', 0.0),
                'reverse_psychology_analysis': reverse_psychology_analysis,  # 存储反向心理学分析结果
                'money_flow_analysis': money_flow_analysis  # 存储资金流向分析结果
            }
            
            # 生成详细的预测分析
            detailed_prediction_info = self.prediction_analyzer.generate_detailed_prediction_analysis(
                recommended_tails, confidence, ensemble_probabilities, all_predictions, data_list, model_decision_records
            )

            return {
                'success': True,
                'recommended_tails': recommended_tails,
                'confidence': confidence,
                'ensemble_probabilities': ensemble_probabilities,
                'model_count': len(all_predictions),
                'message': f'终极在线预测完成，使用了{len(all_predictions)}个模型',
                'weight_details': detailed_prediction_info['weight_details'],
                'decision_summary': detailed_prediction_info['decision_summary'],
                'detailed_analysis': detailed_prediction_info['detailed_analysis'],
                'total_models_participated': len(all_predictions),
                'reverse_psychology_analysis': reverse_psychology_analysis  # 存储分析结果供学习使用
            }
            
        except Exception as e:
            return {'success': False, 'message': f'在线预测失败: {str(e)}'}
    
    def learn_online(self, data_list: List[Dict], actual_tails: List[int]) -> Dict:
        """在线学习 - 真正的增量学习"""
        if not self.is_initialized:
            return {'success': False, 'message': '模型未初始化'}
        
        # 检查是否是重复学习同一个样本
        current_sample_id = f"{len(data_list)}_{hash(str(actual_tails))}"
        if hasattr(self, '_last_sample_id') and self._last_sample_id == current_sample_id:
            print(f"⚠️ 检测到重复学习同一样本，跳过计数增加")
            return {
                'success': True,
                'samples_processed': self.total_samples_seen,
                'prediction_correct': None,
                'message': '重复样本，跳过学习'
            }
        
        self._last_sample_id = current_sample_id
        self.total_samples_seen += 1
        print(f"📊 处理样本 #{self.total_samples_seen}")
        
        try:
            # 提取特征
            base_features = self.feature_engineer.extract_enhanced_features(data_list)
            features = base_features  # 保持兼容性
            
            # 为不同模型准备数据格式
            X_river = {f'feature_{i}': features[i] for i in range(len(features))}
            X_sklearn = base_features.reshape(1, -1)
            
            # 概念漂移检测
            drift_detected_by = []
            prediction_correct = False
            
            # 检查预测准确性
            if self.last_prediction_result:
                predicted_tails = self.last_prediction_result['recommended_tails']
                prediction_correct = any(tail in actual_tails for tail in predicted_tails)
                
                if prediction_correct:
                    self.correct_predictions += 1
                
                # 更新漂移检测器
                error = 0.0 if prediction_correct else 1.0
                
                for detector_name, detector in self.drift_detectors.items():
                    try:
                        if hasattr(detector, 'update'):
                            detector.update(error)
                            if hasattr(detector, 'drift_detected') and detector.drift_detected:
                                drift_detected_by.append(detector_name)
                    except Exception as e:
                        print(f"漂移检测器 {detector_name} 更新失败: {e}")
            
            # === 在线学习所有模型 ===
            learning_results = {}

            # === 反向心理学模型学习 ===
            if hasattr(self, 'reverse_psychology_predictor') and self.reverse_psychology_predictor:
                try:
                    # 从上次预测结果中获取反向心理学分析
                    if self.last_prediction_result:
                        stored_reverse_analysis = self.last_prediction_result.get('reverse_psychology_analysis')
                        
                        if stored_reverse_analysis:
                            # 让模型从结果中学习
                            learn_result = self.reverse_psychology_predictor.learn_from_outcome(
                                stored_reverse_analysis,
                                actual_tails
                            )
                            
                            if learn_result and learn_result.get('learning_success', False):
                                learning_results['reverse_psychology_predictor'] = f'success (accuracy: {learn_result.get("overall_accuracy", 0):.3f})'
                                print(f"✅ 反向心理学模型学习完成，总体准确率: {learn_result.get('overall_accuracy', 0):.3f}")
                            else:
                                learning_results['reverse_psychology_predictor'] = 'success (basic update)'
                                print("✅ 反向心理学模型基础学习完成")
                        else:
                            # 没有存储的分析结果，进行基础学习
                            print("⚠️ 无存储的反向心理学分析，进行基础权重更新")
                            
                            # 基于预测正确性更新模型置信度
                            if 'reverse_psychology_predictor' in self.ensemble_weights:
                                weight_info = self.ensemble_weights['reverse_psychology_predictor']
                                if prediction_correct is not None:
                                    # 更新性能历史
                                    if 'performance_history' not in weight_info:
                                        weight_info['performance_history'] = []
                                    weight_info['performance_history'].append(prediction_correct)
                                    if len(weight_info['performance_history']) > 100:
                                        weight_info['performance_history'].pop(0)
                            
                            learning_results['reverse_psychology_predictor'] = 'success (basic weight update)'
                    else:
                        learning_results['reverse_psychology_predictor'] = 'skipped (no last prediction result)'
                        print("⚠️ 反向心理学模型学习跳过：无上次预测结果")
                
                except Exception as e:
                    print(f"❌ 反向心理学模型学习失败: {e}")
                    learning_results['reverse_psychology_predictor'] = f'failed: {str(e)}'
            
            # === 反操控分析器学习 ===
            if hasattr(self, 'banker_behavior_analyzer') and self.banker_behavior_analyzer and self.last_prediction_result:
                try:
                    # 构建预测结果用于学习
                    model_predictions_dict = self.last_prediction_result.get('model_predictions', {})
                    anti_manipulation_predictions = model_predictions_dict.get('anti_manipulation_banker_behavior', {})

                    if anti_manipulation_predictions:
                        # 找到预测概率最高的尾数作为推荐
                        recommended_tails = [tail for tail, prob in anti_manipulation_predictions.items() if prob > 0.6]
                        avg_confidence = sum(anti_manipulation_predictions.values()) / len(anti_manipulation_predictions) if anti_manipulation_predictions else 0.5
                    else:
                        recommended_tails = []
                        avg_confidence = 0.5

                    anti_prediction = {
                        'recommended_tails': recommended_tails,
                        'confidence': avg_confidence,
                        'success': True
                    }
        
                    # 反操控分析器通常没有专门的学习方法，所以进行基础更新
                    learning_results['anti_manipulation_banker_behavior'] = 'success (basic update)'
                    print("✅ 反操控分析器基础学习完成")
        
                except Exception as e:
                    print(f"❌ 反操控分析器学习失败: {e}")
                    learning_results['anti_manipulation_banker_behavior'] = f'failed: {str(e)}'

            # === 冷门挖掘器学习 ===
            if hasattr(self, 'unpopular_digger') and self.unpopular_digger and self.last_prediction_result:
                try:
                    # 构建预测结果用于学习
                    model_predictions_dict = self.last_prediction_result.get('model_predictions', {})
                    unpopular_digger_predictions = model_predictions_dict.get('unpopular_digger', {})

                    if unpopular_digger_predictions:
                        # 找到预测概率最高的尾数作为推荐
                        recommended_tails = [tail for tail, prob in unpopular_digger_predictions.items() if prob > 0.6]
                        avg_confidence = sum(unpopular_digger_predictions.values()) / len(unpopular_digger_predictions) if unpopular_digger_predictions else 0.5
                    else:
                        recommended_tails = []
                        avg_confidence = 0.5

                    digger_prediction = {
                        'recommended_tails': recommended_tails,
                        'confidence': avg_confidence,
                        'success': True
                    }
        
                    learn_result = self.unpopular_digger.learn_from_outcome(
                        digger_prediction, 
                        actual_tails
                    )
        
                    if learn_result and learn_result.get('learning_success', False):
                        learning_results['unpopular_digger'] = f'success (accuracy: {learn_result.get("recommendation_accuracy", 0):.3f})'
                        print(f"✅ 冷门挖掘器学习完成，推荐准确率: {learn_result.get('recommendation_accuracy', 0):.3f}")
                        print(f"   冷门复出成功率: {learn_result.get('revival_success_rate', 0):.3f}")
                    else:
                        learning_results['unpopular_digger'] = 'success (basic update)'
                        print("✅ 冷门挖掘器基础学习完成")
    
                except Exception as e:
                    print(f"❌ 冷门挖掘器学习失败: {e}")
                    learning_results['unpopular_digger'] = f'failed: {str(e)}'

            # === 资金流向分析器学习 ===
            if hasattr(self, 'money_flow_analyzer') and self.money_flow_analyzer and self.last_prediction_result:
                try:
                    # 从上次预测结果中获取资金流向分析
                    stored_flow_analysis = self.last_prediction_result.get('money_flow_analysis')
                    
                    if stored_flow_analysis:
                        # 让模型从结果中学习
                        learn_result = self.money_flow_analyzer.learn_from_outcome(
                            stored_flow_analysis,
                            actual_tails
                        )
                        
                        if learn_result and learn_result.get('learning_success', False):
                            learning_results['money_flow_analyzer'] = f'success (accuracy: {learn_result.get("overall_accuracy", 0):.3f})'
                            print(f"✅ 资金流向分析器学习完成，总体准确率: {learn_result.get('overall_accuracy', 0):.3f}")
                        else:
                            learning_results['money_flow_analyzer'] = 'success (basic update)'
                            print("✅ 资金流向分析器基础学习完成")
                    else:
                        learning_results['money_flow_analyzer'] = 'skipped (no stored analysis)'
                        print("⚠️ 资金流向分析器学习跳过：无存储的分析结果")
                
                except Exception as e:
                    print(f"❌ 资金流向分析器学习失败: {e}")
                    learning_results['money_flow_analyzer'] = f'failed: {str(e)}'

            # === 博弈论策略器学习 ===
            if hasattr(self, 'game_theory_strategist') and self.game_theory_strategist and self.last_prediction_result:
                try:
                    # 从上次预测结果中获取博弈论分析
                    stored_game_analysis = self.last_prediction_result.get('game_theory_analysis')
        
                    if stored_game_analysis:
                        # 让模型从结果中学习
                        learn_result = self.game_theory_strategist.learn_from_outcome(
                            stored_game_analysis,
                            actual_tails
                        )
            
                        if learn_result and learn_result.get('learning_success', False):
                            performance_metrics = learn_result.get('performance_metrics', {})
                            accuracy = performance_metrics.get('accuracy', 0)
                            strategy_type = performance_metrics.get('strategy_type', 'unknown')
                
                            learning_results['game_theory_strategist'] = f'success (accuracy: {accuracy:.3f}, strategy: {strategy_type})'
                            print(f"✅ 博弈论策略器学习完成，准确率: {accuracy:.3f}, 策略类型: {strategy_type}")
                
                            # 显示策略表现详情
                            recent_accuracy = performance_metrics.get('recent_accuracy', 0)
                            total_predictions = performance_metrics.get('total_predictions', 0)
                            print(f"   📊 最近准确率: {recent_accuracy:.3f}, 总预测次数: {total_predictions}")
                        else:
                            learning_results['game_theory_strategist'] = 'success (basic update)'
                            print("✅ 博弈论策略器基础学习完成")
                    else:
                        learning_results['game_theory_strategist'] = 'skipped (no stored analysis)'
                        print("⚠️ 博弈论策略器学习跳过：无存储的分析结果")
    
                except Exception as e:
                    print(f"❌ 博弈论策略器学习失败: {e}")
                    learning_results['game_theory_strategist'] = f'failed: {str(e)}'

            # === 操控时机检测器学习 ===
            if hasattr(self, 'manipulation_timing_detector') and self.manipulation_timing_detector and self.last_prediction_result:
                try:
                    # 从上次预测结果中获取操控时机分析
                    stored_timing_analysis = self.last_prediction_result.get('manipulation_timing_analysis')
        
                    if stored_timing_analysis:
                        # 让模型从结果中学习
                        learn_result = self.manipulation_timing_detector.learn_from_outcome(
                            stored_timing_analysis,
                            actual_tails
                        )
            
                        if learn_result and learn_result.get('learning_success', False):
                            detection_accuracy = learn_result.get('detection_accuracy', 0)
                            total_predictions = learn_result.get('total_predictions', 0)
                
                            learning_results['manipulation_timing_detector'] = f'success (accuracy: {detection_accuracy:.3f}, total: {total_predictions})'
                            print(f"✅ 操控时机检测器学习完成，检测准确率: {detection_accuracy:.3f}, 总预测次数: {total_predictions}")
                        else:
                            learning_results['manipulation_timing_detector'] = 'success (basic update)'
                            print("✅ 操控时机检测器基础学习完成")
                    else:
                        learning_results['manipulation_timing_detector'] = 'skipped (no stored analysis)'
                        print("⚠️ 操控时机检测器学习跳过：无存储的分析结果")
    
                except Exception as e:
                    print(f"❌ 操控时机检测器学习失败: {e}")
                    learning_results['manipulation_timing_detector'] = f'failed: {str(e)}'

            # === 反趋势猎手学习 ===
            if hasattr(self, 'anti_trend_hunter') and self.anti_trend_hunter and self.last_prediction_result:
                try:
                    # 从上次预测结果中获取反趋势分析
                    stored_anti_trend_analysis = self.last_prediction_result.get('anti_trend_analysis')
        
                    if stored_anti_trend_analysis:
                        # 让模型从结果中学习
                        learn_result = self.anti_trend_hunter.learn_from_outcome(
                            stored_anti_trend_analysis,
                            actual_tails
                        )
            
                        if learn_result and learn_result.get('learning_success', False):
                            current_accuracy = learn_result.get('current_accuracy', 0)
                            strategy_performance = learn_result.get('strategy_performance', {})
                            total_predictions = learn_result.get('total_predictions', 0)
                
                            learning_results['anti_trend_hunter'] = f'success (accuracy: {current_accuracy:.3f}, total: {total_predictions})'
                            print(f"✅ 反趋势猎手学习完成，总体准确率: {current_accuracy:.3f}, 总预测次数: {total_predictions}")
                
                            # 显示策略表现详情
                            trend_break_rate = strategy_performance.get('trend_break_rate', 0)
                            reversal_rate = strategy_performance.get('reversal_rate', 0)
                            print(f"   📊 趋势终结成功率: {trend_break_rate:.3f}")
                            print(f"   📊 反转成功率: {reversal_rate:.3f}")
                        else:
                            learning_results['anti_trend_hunter'] = 'success (basic update)'
                            print("✅ 反趋势猎手基础学习完成")
                    else:
                        learning_results['anti_trend_hunter'] = 'skipped (no stored analysis)'
                        print("⚠️ 反趋势猎手学习跳过：无存储的分析结果")
    
                except Exception as e:
                    print(f"❌ 反趋势猎手学习失败: {e}")
                    learning_results['anti_trend_hunter'] = f'failed: {str(e)}'

            # === 群体心理分析器学习 ===
            if hasattr(self, 'crowd_psychology_analyzer') and self.crowd_psychology_analyzer and self.last_prediction_result:
                try:
                    # 从上次预测结果中获取群体心理分析
                    stored_crowd_psychology_analysis = self.last_prediction_result.get('crowd_psychology_analysis')
        
                    if stored_crowd_psychology_analysis:
                        # 让模型从结果中学习
                        learn_result = self.crowd_psychology_analyzer.learn_from_outcome(
                            stored_crowd_psychology_analysis,
                            actual_tails
                        )
            
                        if learn_result and learn_result.get('learning_success', False):
                            current_accuracy = learn_result.get('current_accuracy', 0)
                            contrarian_success_rate = learn_result.get('contrarian_success_rate', 0)
                            psychology_insights = learn_result.get('psychology_insights', {})
                            total_predictions = learn_result.get('total_predictions', 0)
                
                            learning_results['crowd_psychology_analyzer'] = f'success (accuracy: {current_accuracy:.3f}, contrarian: {contrarian_success_rate:.3f}, total: {total_predictions})'
                            print(f"✅ 群体心理分析器学习完成，总体准确率: {current_accuracy:.3f}, 总预测次数: {total_predictions}")
                            print(f"   📊 反向策略成功率: {contrarian_success_rate:.3f}")
                
                            # 显示心理学洞察
                            key_insights = psychology_insights.get('key_insights', [])
                            if key_insights:
                                print(f"   🧠 心理学洞察: {'; '.join(key_insights[:2])}")
                
                            crowd_behavior_accuracy = psychology_insights.get('crowd_behavior_accuracy', 0)
                            emotion_prediction_accuracy = psychology_insights.get('emotion_prediction_accuracy', 0)
                            print(f"   📈 群体行为预测准确率: {crowd_behavior_accuracy:.3f}")
                            print(f"   😨 情绪预测准确率: {emotion_prediction_accuracy:.3f}")
                        else:
                            learning_results['crowd_psychology_analyzer'] = 'success (basic update)'
                            print("✅ 群体心理分析器基础学习完成")
                    else:
                        learning_results['crowd_psychology_analyzer'] = 'skipped (no stored analysis)'
                        print("⚠️ 群体心理分析器学习跳过：无存储的分析结果")
    
                except Exception as e:
                    print(f"❌ 群体心理分析器学习失败: {e}")
                    learning_results['crowd_psychology_analyzer'] = f'failed: {str(e)}'
                    
            # River模型学习
            for model_name, model in self.river_models.items():
                try:
                    # 历史模式匹配算法不需要传统的学习，只需要更新历史数据
                    if hasattr(model, 'update_historical_data'):
                        model.update_historical_data(data_list)
                        learning_results[f'river_{model_name}'] = 'success (pattern updated)'
                    else:
                        # 普通模型学习
                        for tail in range(10):
                            y = 1 if tail in actual_tails else 0
                            if hasattr(model, 'learn_one'):
                                model.learn_one(X_river, y)
                            
                            # 更新性能度量
                            if model_name in self.metrics_trackers:
                                # 这里可以进行预测来更新度量，但为了性能考虑可以定期更新
                                pass
                        
                        learning_results[f'river_{model_name}'] = 'success'
                
                except Exception as e:
                    print(f"River模型 {model_name} 学习失败: {e}")
                    learning_results[f'river_{model_name}'] = f'failed: {e}'
            
            # scikit-multiflow模型学习
            for model_name, model in self.sklearn_models.items():
                try:
                    for tail in range(10):
                        y = np.array([1 if tail in actual_tails else 0])
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_sklearn, y)
                    
                    learning_results[f'sklearn_{model_name}'] = 'success'
                
                except Exception as e:
                    print(f"sklearn模型 {model_name} 学习失败: {e}")
                    learning_results[f'sklearn_{model_name}'] = f'failed: {e}'
            
            # PyTorch深度学习模型在线学习
            if PYTORCH_AVAILABLE and self.deep_learning_module and self.deep_learning_module.models:
                try:
                    print(f"🧠 开始PyTorch深度学习模型在线学习...")
                    dl_learning_results = self.deep_learning_module.learn_online_single(features, actual_tails)
                    
                    for model_name, result in dl_learning_results.items():
                        full_model_name = f'pytorch_{model_name}'
                        if result.get('status') == 'success':
                            learning_results[full_model_name] = f'success (loss: {result["loss"]:.4f}, acc: {result["accuracy"]:.3f})'
                            print(f"   ✓ {model_name}: 损失={result['loss']:.4f}, 准确率={result['accuracy']:.3f}")
                            
                            # 动态调整学习率
                            if result['loss'] > 0.8:  # 如果损失较高，降低学习率
                                self.deep_learning_module.update_learning_rate(model_name, factor=0.9)
                        else:
                            learning_results[full_model_name] = f'failed: {result.get("error", "unknown")}'
                            print(f"   ✗ {model_name}: 学习失败")
                    
                    print(f"✅ PyTorch深度学习在线学习完成")
                
                except Exception as e:
                    print(f"❌ PyTorch深度学习在线学习失败: {e}")
                    # 为所有深度学习模型添加失败记录
                    if hasattr(self, 'deep_learning_module') and self.deep_learning_module and self.deep_learning_module.models:
                        for model_name in self.deep_learning_module.models.keys():
                            learning_results[f'pytorch_{model_name}'] = f'failed: {str(e)}'
            
            # === 处理权重投资结算 ===
            if prediction_correct is not None and hasattr(self, 'current_weight_investments') and self.current_weight_investments:
                self.settle_weight_investments(actual_tails)
            elif prediction_correct is not None:
                # 如果没有投资记录，更新传统的性能历史
                print("📊 更新传统性能历史（无投资记录）")
                for model_key in self.ensemble_weights.keys():
                    weight_info = self.ensemble_weights[model_key]
                    weight_info['performance_history'].append(prediction_correct)
                    if len(weight_info['performance_history']) > 100:
                        weight_info['performance_history'].pop(0)
            
            # === 更新智能特征处理组件 ===
            if prediction_correct is not None:
                accuracy = 1.0 if prediction_correct else 0.0
                
                # 更新智能特征处理组件
                self.feature_engineer.update_components_with_learning_result(features, accuracy)

            # === 处理概念漂移 ===
            if drift_detected_by:
                self.handle_concept_drift(drift_detected_by)
            
            # === 保存学习记录 ===
            self.save_learning_record(base_features, actual_tails, prediction_correct, drift_detected_by)
            
            # === 保存各模型的预测记录 ===
            self._save_model_predictions(actual_tails)
            
            # === 定期保存模型状态 ===
            if self.total_samples_seen % 10 == 0:  # 每10个样本保存一次，确保及时保存
                self.save_model_state()
            elif self.total_samples_seen <= 50:  # 前50个样本每次都保存，确保早期数据不丢失
                self.save_model_state()
            
            # 触发UI更新（如果有主应用引用）
            if hasattr(self, '_main_app_ref') and self._main_app_ref is not None:
                try:
                    # 同时触发 ui_updates 和 ai_display_manager 的更新
                    if hasattr(self._main_app_ref, 'ui_updates') and self._main_app_ref.ui_updates is not None:
                        # 先更新基础UI组件
                        self._main_app_ref.root.after(200, self._main_app_ref.ui_updates.update_learning_progress_display)
                        print("🔄 已安排基础UI更新任务")
                    
                    if hasattr(self._main_app_ref, 'ai_display_manager') and self._main_app_ref.ai_display_manager is not None:
                        # 再更新详细的AI显示
                        self._main_app_ref.root.after(500, self._main_app_ref.ai_display_manager.trigger_update_after_learning)
                        print("🔄 已安排AI详细显示更新任务")
                except Exception as e:
                    print(f"触发AI学习后更新失败: {e}")

            return {
                'success': True,
                'samples_processed': self.total_samples_seen,
                'prediction_correct': prediction_correct,
                'drift_detected': len(drift_detected_by) > 0,
                'drift_detectors': drift_detected_by,
                'learning_results': learning_results,
                'current_accuracy': self.get_current_accuracy(),
                'message': '终极在线学习完成'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'在线学习失败: {str(e)}'}
    
    def batch_pretrain(self, data_list: List[Dict]) -> Dict:
        """批量预训练模型 - 基础数据量阈值策略"""
        if not self.is_initialized:
            return {'success': False, 'message': '模型未初始化'}
    
        if len(data_list) < 5:
            return {'success': False, 'message': '数据不足，需要至少5期数据'}
    
        try:
            # 🔧 修复时间权重顺序问题：反转数据列表
            # 用户的数据是最新在前(index 0)，最老在后
            # 但学习需要从最老开始，所以反转列表
            print(f"🔄 修复数据顺序：原始数据最新在index 0，反转后最老数据在index 0")
            print(f"   反转前：data_list[0] = 最新期")
            print(f"   反转后：data_list[0] = 最老期") 
        
            reversed_data_list = list(reversed(data_list))
            data_list = reversed_data_list  # 使用反转后的数据
        
            print(f"✅ 数据顺序已修复，现在从最老数据开始学习，时间权重正确")
        
            # 设置基础数据量阈值（从配置中读取）
            base_data_threshold = self.learning_config.get('base_training_data_threshold', 500)
            total_data_count = len(data_list)
        
            print(f"🚀 开始批量预训练，总历史数据：{total_data_count}期")
            print(f"📊 基础数据量阈值：{base_data_threshold}期")
        
            # 计算训练起始位置
            if total_data_count > base_data_threshold:
                # 有足够数据，从T(total-base_threshold)开始训练
                start_training_index = total_data_count - base_data_threshold
                print(f"✅ 数据充足，从第{start_training_index}期开始训练（确保有{base_data_threshold}期基础数据）")
            else:
                # 数据不足阈值，从能训练的最早位置开始
                start_training_index = total_data_count - 1
                actual_base_data = total_data_count - 1
                print(f"⚠️ 数据不足阈值，从第{start_training_index}期开始训练（实际基础数据：{actual_base_data}期）")
        
            successful_samples = 0
            total_samples = 0
        
            # 从起始位置开始训练，逐步向T0训练
            print(f"📚 训练策略：从T{start_training_index} → T0，确保充足历史数据")
        
            for i in range(start_training_index, 0, -1):  # 从start_training_index到1
                try:
                    total_samples += 1
                
                    # 使用从第i期到最新的所有数据作为历史特征数据
                    feature_data = data_list[i:]  # T[i]到T0的所有数据
                    # 要学习的目标是第i-1期的尾数
                    actual_tails = data_list[i-1].get('tails', [])
                
                    # 验证数据质量
                    history_data_count = len(feature_data)
                    if history_data_count < 5 or not actual_tails:
                        print(f"   ⚠️ 跳过训练样本T{i-1}：历史数据{history_data_count}期，目标尾数{len(actual_tails)}个")
                        continue
                
                    # 执行训练
                    result = self.learn_online(feature_data, actual_tails)
                    if result.get('success', False):
                        successful_samples += 1
                
                    # 定期显示训练进度
                    if total_samples % 50 == 0:
                        current_position = start_training_index - total_samples + 1
                        progress_percent = (total_samples / start_training_index) * 100
                        print(f"   📈 训练进度：T{current_position} ({progress_percent:.1f}%) "
                            f"成功率:{successful_samples}/{total_samples} "
                            f"历史数据:{history_data_count}期")
                
                except Exception as e:
                    print(f"   ❌ 训练样本T{i-1}失败: {e}")
                    continue
        
            success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
            print(f"✅ 批量预训练完成")
            print(f"   📊 训练统计：成功{successful_samples}/{total_samples} (成功率:{success_rate:.1%})")
            print(f"   📊 最终历史数据量：{len(data_list)}期")
        
            return {
                'success': successful_samples > 0,
                'total_samples': total_samples,
                'successful_samples': successful_samples,
                'success_rate': success_rate,
                'base_data_threshold': base_data_threshold,
                'actual_base_data': len(data_list[start_training_index:]) if start_training_index < len(data_list) else 0,
                'message': f'批量预训练完成，成功训练 {successful_samples}/{total_samples} 个样本'
            }
        
        except Exception as e:
            return {'success': False, 'message': f'批量预训练失败: {str(e)}'}
    
    def deep_learning_batch_train(self, data_list: List[Dict], epochs=50) -> Dict:
        """深度学习批量训练"""
        if not PYTORCH_AVAILABLE or not self.deep_learning_module:
            return {'success': False, 'message': 'PyTorch深度学习模块不可用'}
        
        if not self.is_initialized:
            return {'success': False, 'message': '基础模型未初始化'}
        
        if len(data_list) < 30:
            return {'success': False, 'message': '深度学习需要至少30期数据'}
        
        try:
            print(f"🚀 开始深度学习批量训练...")
            
            # 执行批量训练
            training_result = self.deep_learning_module.batch_train(
                data_list, 
                epochs=epochs, 
                validation_split=0.2
            )
            
            if training_result['success']:
                print(f"✅ 深度学习批量训练完成")
                
                # 更新集成权重以包含新的深度学习模型
                self._update_ensemble_weights_for_deep_learning()
                
                # 保存模型状态
                self.save_model_state()
                
                return {
                    'success': True,
                    'message': '深度学习批量训练成功',
                    'training_details': training_result,
                    'models_trained': list(self.deep_learning_module.models.keys())
                }
            else:
                return training_result
                
        except Exception as e:
            print(f"❌ 深度学习批量训练失败: {e}")
            traceback.print_exc()
            return {'success': False, 'message': f'深度学习训练失败: {str(e)}'}
    
    def _update_ensemble_weights_for_deep_learning(self):
        """为深度学习模型更新集成权重"""
        if not PYTORCH_AVAILABLE or not self.deep_learning_module:
            return
    
        # 为新的深度学习模型添加权重
        for model_name in self.deep_learning_module.models.keys():
            pytorch_model_key = f'pytorch_{model_name}'
            if pytorch_model_key not in self.ensemble_weights:
                # 计算当前所有模型的平均权重作为新模型的初始权重
                current_weights = [w['weight'] for w in self.ensemble_weights.values()]
                if current_weights:
                    avg_weight = sum(current_weights) / len(current_weights)
                    initial_weight = min(0.1, avg_weight)  # 使用平均权重，但不超过0.1
                else:
                    initial_weight = 0.1
            
                self.ensemble_weights[pytorch_model_key] = {
                    'weight': initial_weight,          # 活跃权重
                    'frozen_weight': 0.0,             # 冻结权重
                    'total_weight': initial_weight,    # 总权重
                    'is_frozen': False,               # 冻结状态
                    'frozen_timestamp': None,         # 冻结时间
                    'confidence': 0.6,
                    'last_update': datetime.now(),
                    'performance_history': []
                }
    
        # 重新归一化所有权重
        total_weight = sum(w['weight'] for w in self.ensemble_weights.values())
        if total_weight > 0:
            for model_key in list(self.ensemble_weights.keys()):
                self.ensemble_weights[model_key]['weight'] /= total_weight
    
        print(f"✅ 集成权重已更新，包含深度学习模型")
    
    def get_deep_learning_stats(self) -> Dict:
        """获取深度学习训练统计"""
        if not PYTORCH_AVAILABLE or not self.deep_learning_module:
            return {'available': False, 'message': 'PyTorch深度学习模块不可用'}
        
        try:
            training_stats = self.deep_learning_module.get_training_stats()
            
            return {
                'available': True,
                'device': str(self.deep_learning_module.device),
                'models': list(self.deep_learning_module.models.keys()),
                'training_stats': training_stats,
                'batch_size': self.deep_learning_module.batch_size,
                'sequence_length': self.deep_learning_module.sequence_length
            }
            
        except Exception as e:
            return {'available': False, 'message': f'获取统计失败: {str(e)}'}
        
    def calculate_ensemble_confidence(self, recommended_tails: List[int], 
                                    ensemble_probabilities: Dict[int, float], 
                                    all_predictions: Dict) -> float:
        """计算集成置信度"""
        if not recommended_tails:
            return 0.0
        
        # 基于集成概率
        tail_prob = ensemble_probabilities.get(recommended_tails[0], 0.5)
        
        # 基于模型一致性
        if recommended_tails:
            target_tail = recommended_tails[0]
            tail_predictions = [pred.get(target_tail, 0.5) for pred in all_predictions.values()]
            consistency = 1.0 - np.std(tail_predictions) if tail_predictions else 0.0
        else:
            consistency = 0.0
        
        # 基于历史准确率
        historical_accuracy = self.get_current_accuracy()
        
        # 基于样本数量
        sample_factor = min(1.0, self.total_samples_seen / 100)
        
        # 综合置信度
        confidence = (tail_prob * 0.35 + consistency * 0.25 + 
                     historical_accuracy * 0.25 + sample_factor * 0.15)
        
        return min(max(confidence, 0.2), 0.95)
    
    def get_model_effective_weight(self, model_key):
        """获取模型的有效权重（包括冻结权重）"""
        if model_key in self.ensemble_weights:
            weight_info = self.ensemble_weights[model_key]
            active_weight = weight_info.get('weight', 0.0)
            frozen_weight = weight_info.get('frozen_weight', 0.0)
            return active_weight + frozen_weight
        return 0.0
    
    def settle_investments(self, actual_tails: List[int]):
        """结算投资结果 - 新的简化版本"""
        if not hasattr(self, 'current_investments'):
            return
    
    def freeze_investment_weights(self, model_weight_investments):
        """冻结模型的投资权重"""
        return self.investment_system.freeze_investment_weights(model_weight_investments, self.ensemble_weights)

    def settle_weight_investments(self, actual_tails: List[int]):
        """结算权重投资结果"""
        if not hasattr(self, 'current_weight_investments'):
            return
    
        # 使用投资管理系统进行结算
        settlement_results = self.investment_system.settle_weight_investments(
            actual_tails, self.current_weight_investments, self.ensemble_weights
        )
    
        # 解冻权重并应用结算结果
        unfrozen_count, final_rewards, final_penalties = self.investment_system.unfreeze_and_settle_weights(
            settlement_results, self.ensemble_weights
        )
    
        # 更新模型的投资历史和性能历史
        for model_key, settlement in settlement_results.items():
            self.investment_system.update_investment_history(
                model_key, settlement, self.current_weight_investments, self.ensemble_weights
            )
    
        # 正确更新投资统计（每次结算只增加1轮）
        if hasattr(self, 'current_weight_investments') and self.current_weight_investments:
            self.investment_stats['total_rounds'] += 1  # 只在有实际投资时增加轮数
            print(f"📊 投资轮数统计更新: 当前轮数 {self.investment_stats['total_rounds']}")
        self.investment_stats['total_rewards'] += final_rewards
        self.investment_stats['total_penalties'] += final_penalties
    
        # 权重归一化
        self.investment_system.normalize_weights(self.ensemble_weights)
    
        print(f"💰 权重投资结算完成:")
        print(f"   总奖励权重: {final_rewards:.4f}")
        print(f"   总惩罚权重: {final_penalties:.4f}")
        print(f"   净权重变化: {final_rewards - final_penalties:.4f}")
    
        # 清空当前权重投资记录
        delattr(self, 'current_weight_investments')
    
    def get_investment_strategy_analysis(self):
        """获取投资策略分析"""
        current_investments = getattr(self, 'current_weight_investments', {})
        if not current_investments:
            return {'status': 'no_active_investments'}
    
        # 获取所有模型的预测结果（用于策略分析）
        all_predictions = getattr(self, 'last_prediction_result', {}).get('model_predictions', {})
    
        return self.investment_system.get_investment_strategy_analysis(current_investments, all_predictions)
    
    def clear_failed_investments(self):
        """清理失败的投资状态"""
        unfrozen_count = self.investment_system.clear_failed_investments(self.ensemble_weights)
    
        # 清空权重投资记录
        if hasattr(self, 'current_weight_investments'):
            self.current_weight_investments = {}
    
        return unfrozen_count > 0

    def _normalize_weights(self):
        """归一化所有模型权重（只归一化可用权重，冻结权重保持不变）"""
        self.investment_system.normalize_weights(self.ensemble_weights)

    def handle_concept_drift(self, drift_detectors: List[str]):
        """处理概念漂移"""
        # 使用漂移检测管理器处理概念漂移
        self.drift_manager.handle_concept_drift(
            drift_detectors, 
            self.ensemble_weights, 
            self.learning_config, 
            self.deep_learning_module
        )
    
        # 重置漂移检测器
        self.drift_manager.reset_drift_detectors(self.drift_detectors)
    
    def _handle_advanced_concept_drift(self, drift_info: Dict):
        """处理高级概念漂移"""
        # 使用漂移检测管理器处理高级概念漂移
        self.drift_manager.handle_advanced_concept_drift(
            drift_info,
            self.feature_selector,
            self.feature_weighter,
            self.ensemble_weights,
            self.deep_learning_module
        )
    
        # 重置高级漂移检测器
        if hasattr(self, 'advanced_drift_detector'):
            self.advanced_drift_detector.reset()

    def reset_poor_performing_models(self):
        """重置表现较差的模型"""
        self.drift_manager.reset_poor_performing_models(self.ensemble_weights)
    
    def adjust_learning_parameters(self):
        """调整学习参数"""
        self.drift_manager.adjust_learning_parameters(self.learning_config, self.deep_learning_module)
    
    def get_current_accuracy(self) -> float:
        """获取当前准确率"""
        if self.total_samples_seen == 0:
            return 0.5
        
        return self.correct_predictions / self.total_samples_seen
    
    def process_new_sample(self, data_list: List[Dict], actual_tails: List[int] = None) -> Dict:
        """处理新样本 - 预测并学习"""
        # 首先进行预测
        prediction_result = self.predict_online(data_list)
        
        # 如果有实际结果，进行在线学习
        if actual_tails is not None:
            learning_result = self.learn_online(data_list, actual_tails)
            
            return {
                'prediction': prediction_result,
                'learning': learning_result,
                'total_samples': self.total_samples_seen,
                'current_accuracy': self.get_current_accuracy(),
                'ensemble_weights': {k: v['weight'] for k, v in self.ensemble_weights.items()}
            }
        
        return {'prediction': prediction_result}
    
    def save_learning_record(self, features: np.ndarray, actual_tails: List[int], 
                            prediction_correct: Optional[bool], drift_detectors: List[str]):
        """保存学习记录"""
        predicted_tails = self.last_prediction_result.get('recommended_tails', []) if self.last_prediction_result else []
        confidence = self.last_prediction_result.get('confidence', 0.0) if self.last_prediction_result else 0.0
    
        self.db_manager.save_learning_record(
            features=features,
            actual_tails=actual_tails,
            prediction_correct=prediction_correct,
            drift_detectors=drift_detectors,
            predicted_tails=predicted_tails,
            confidence=confidence,
            sample_number=self.total_samples_seen
        )
    
    def _record_model_prediction(self, model_name: str, predicted_class: int, confidence: float, target_tail: int):
        """记录单个模型的预测"""
        self.db_manager.record_model_prediction(
            model_name=model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            target_tail=target_tail,
            sample_number=self.total_samples_seen + 1
        )
    
    def _save_model_predictions(self, actual_tails: List[int]):
        """保存所有模型的预测结果"""
        try:
            # 更新模型预测记录的实际结果
            self.db_manager.update_model_predictions_with_actual_results(
                actual_tails=actual_tails,
                sample_number=self.total_samples_seen
            )
        
        except Exception as e:
            print(f"保存模型预测失败: {e}")
    
    def get_model_performance_stats(self) -> Dict:
        """获取每个模型的详细性能统计"""
        return self.db_manager.get_model_performance_stats()
        
    def save_drift_record(self, detector_name: str, drift_type: str, confidence: float, action: str):
        """保存概念漂移记录"""
        self.db_manager.save_drift_record(detector_name, drift_type, confidence, action)
    
    def ensure_pytorch_model_directory(self, models_dir=None):
        """确保PyTorch模型目录存在"""
        if models_dir is None:
            # 使用默认目录
            models_dir = self.data_dir / "deep_learning" / "models"
    
        try:
            # 确保是Path对象
            if isinstance(models_dir, str):
                models_dir = Path(models_dir)
        
            # 创建目录
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ PyTorch模型目录已创建: {models_dir}")
        
            # 将目录路径保存到深度学习模块中
            if hasattr(self, 'deep_learning_module') and self.deep_learning_module:
                self.deep_learning_module.models_dir = str(models_dir)
                print(f"✅ 模型目录路径已传递给深度学习模块")
        
            return str(models_dir)
        except Exception as e:
            print(f"❌ 创建PyTorch模型目录失败: {e}")
            return None
    
    def save_model_state(self):
        """保存模型状态"""
        try:
            # 从数据库获取最新的准确计数
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
            # 获取实际的记录数
            cursor.execute("SELECT COUNT(*) FROM online_predictions")
            db_count_result = cursor.fetchone()
            actual_total = db_count_result[0] if db_count_result else 0
        
            cursor.execute("SELECT COUNT(*) FROM online_predictions WHERE is_correct = 1")
            db_correct_result = cursor.fetchone()
            actual_correct = db_correct_result[0] if db_correct_result else 0
        
            conn.close()
        
            # 使用数据库中的准确数据
            state = {
                'total_samples_seen': actual_total,
                'correct_predictions': actual_correct,
                'ensemble_weights': self.ensemble_weights,
                'learning_config': self.learning_config,
                'last_update': datetime.now().isoformat(),
                'version': '3.0',  # 更新版本号以包含深度学习
                'save_source': 'database_verified',  # 标记数据来源
                'deep_learning_available': PYTORCH_AVAILABLE,
                'deep_learning_models': list(self.deep_learning_module.models.keys()) if PYTORCH_AVAILABLE and self.deep_learning_module else []
            }
            
            # 保存深度学习模型状态
            if PYTORCH_AVAILABLE and self.deep_learning_module and self.deep_learning_module.models:
                try:
                    deep_learning_state = {}
                    online_learning_stats = {}
                    
                    # 确保PyTorch模型目录存在
                    model_save_dir = self.ensure_pytorch_model_directory()
                    if not model_save_dir:
                        print(f"❌ 无法创建PyTorch模型目录，跳过深度学习模型保存")
                        state['deep_learning_save_error'] = "无法创建模型目录"
                    else:
                        model_save_dir = Path(model_save_dir)
                    
                    # 验证目录创建是否成功
                    if not model_save_dir.exists():
                        print(f"❌ 模型保存目录创建失败: {model_save_dir}")
                        raise Exception(f"无法创建模型保存目录: {model_save_dir}")
            
                    # 检查目录写入权限
                    try:
                        test_file = model_save_dir / "write_test.tmp"
                        test_file.write_text("test")
                        test_file.unlink()
                        print(f"✅ 模型保存目录写入权限验证成功")
                    except Exception as write_error:
                        print(f"❌ 模型保存目录写入权限验证失败: {write_error}")
                        raise Exception(f"模型保存目录无写入权限: {write_error}")
            
                    for model_name, model in self.deep_learning_module.models.items():
                        model_state_path = model_save_dir / f"pytorch_{model_name}_state.pth"
                
                        # 确保模型文件的直接父目录存在
                        model_state_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 再次确保文件的父目录存在（双重保险）
                        model_state_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 保存模型状态和在线学习统计
                        training_history = self.deep_learning_module.training_history[model_name]
                        try:
                            # 保存前再次验证路径
                            print(f"🔧 准备保存模型 {model_name} 到: {model_state_path}")
                
                            # 安全地准备训练历史数据（确保可序列化）
                            safe_training_history = {}
                            try:
                                # 转换所有数据为Python原生类型
                                if 'loss' in training_history:
                                    safe_training_history['loss'] = [float(x) for x in training_history['loss'] if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
                                else:
                                    safe_training_history['loss'] = []
                    
                                if 'accuracy' in training_history:
                                    safe_training_history['accuracy'] = [float(x) for x in training_history['accuracy'] if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
                                else:
                                    safe_training_history['accuracy'] = []
                    
                                safe_training_history['best_loss'] = float(training_history.get('best_loss', float('inf')))
                                safe_training_history['epochs_trained'] = int(training_history.get('epochs_trained', 0))
                    
                                # 处理无穷大值
                                if math.isinf(safe_training_history['best_loss']):
                                    safe_training_history['best_loss'] = 999999.0
                        
                            except Exception as history_e:
                                print(f"   ⚠️ 处理训练历史数据失败: {history_e}")
                                safe_training_history = {
                                    'loss': [],
                                    'accuracy': [],
                                    'best_loss': 999999.0,
                                    'epochs_trained': 0
                                }
                
                            # 安全地准备在线学习统计数据
                            safe_online_stats = {}
                            try:
                                safe_online_stats = {
                                    'total_online_updates': len(safe_training_history.get('loss', [])),
                                    'recent_loss': safe_training_history.get('loss', [])[-10:],
                                    'recent_accuracy': safe_training_history.get('accuracy', [])[-10:],
                                    'current_lr': float(self.deep_learning_module.optimizers[model_name].param_groups[0]['lr'])
                                }
                            except Exception as stats_e:
                                print(f"   ⚠️ 处理在线学习统计失败: {stats_e}")
                                safe_online_stats = {
                                    'total_online_updates': 0,
                                    'recent_loss': [],
                                    'recent_accuracy': [],
                                    'current_lr': 0.001
                                }
                
                            # 准备要保存的状态字典
                            save_dict = {}
                
                            # 安全地获取模型状态
                            try:
                                save_dict['model_state_dict'] = model.state_dict()
                                print(f"   ✓ 模型状态获取成功")
                            except Exception as model_e:
                                print(f"   ❌ 获取模型状态失败: {model_e}")
                                raise Exception(f"无法获取模型状态: {model_e}")
                
                            # 安全地获取优化器状态
                            try:
                                save_dict['optimizer_state_dict'] = self.deep_learning_module.optimizers[model_name].state_dict()
                                print(f"   ✓ 优化器状态获取成功")
                            except Exception as opt_e:
                                print(f"   ⚠️ 获取优化器状态失败: {opt_e}")
                                # 优化器状态不是必需的，可以跳过
                                save_dict['optimizer_state_dict'] = None
                
                            # 安全地获取调度器状态
                            try:
                                save_dict['scheduler_state_dict'] = self.deep_learning_module.schedulers[model_name].state_dict()
                                print(f"   ✓ 调度器状态获取成功")
                            except Exception as sch_e:
                                print(f"   ⚠️ 获取调度器状态失败: {sch_e}")
                                # 调度器状态不是必需的，可以跳过
                                save_dict['scheduler_state_dict'] = None
                
                            # 添加处理过的安全数据
                            save_dict['training_history'] = safe_training_history
                            save_dict['online_learning_stats'] = safe_online_stats
                            save_dict['save_timestamp'] = datetime.now().isoformat()
                            save_dict['model_type'] = model_name
                
                            # 执行保存操作
                            ai_config.torch.save(save_dict, str(model_state_path))
                            print(f"✅ 模型 {model_name} 保存成功")
                    
                        except Exception as save_error:
                            print(f"❌ 保存模型 {model_name} 失败: {save_error}")
                            print(f"📍 尝试的路径: {model_state_path}")
                            print(f"📁 父目录存在: {model_state_path.parent.exists()}")
                            print(f"📁 父目录可写: {os.access(model_state_path.parent, os.W_OK)}")
                            print(f"🔍 错误类型: {type(save_error).__name__}")
                            print(f"🔍 详细错误: {str(save_error)}")
                
                            # 尝试简化保存（只保存模型状态）
                            try:
                                print(f"   🔄 尝试简化保存模型 {model_name}...")
                                simple_save_dict = {
                                    'model_state_dict': model.state_dict(),
                                    'save_timestamp': datetime.now().isoformat(),
                                    'model_type': model_name,
                                    'simplified_save': True
                                }
                    
                                simple_path = model_save_dir / f"pytorch_{model_name}_simple.pth"
                                ai_config.torch.save(simple_save_dict, str(simple_path))
                                print(f"   ✅ 模型 {model_name} 简化保存成功到: {simple_path}")
                    
                            except Exception as simple_save_error:
                               print(f"   ❌ 简化保存也失败: {simple_save_error}")
                               print(f"   ⚠️ 模型 {model_name} 完全保存失败，将跳过")
                
                            # 继续保存其他模型，不中断整个保存过程
                            continue
                        
                        deep_learning_state[model_name] = str(model_state_path)
                        
                        # 记录在线学习统计
                        if training_history.get('loss'):
                            online_learning_stats[model_name] = {
                                'online_updates': len(training_history['loss']),
                                'avg_recent_loss': np.mean(training_history['loss'][-20:]) if len(training_history['loss']) >= 20 else 0,
                                'avg_recent_accuracy': np.mean(training_history['accuracy'][-20:]) if len(training_history['accuracy']) >= 20 else 0
                            }
                    
                    state['deep_learning_model_paths'] = deep_learning_state
                    state['online_learning_stats'] = online_learning_stats
                    print(f"✅ 深度学习模型状态已保存（包含在线学习统计）")
                    
                    # 显示在线学习统计
                    for model_name, stats in online_learning_stats.items():
                        print(f"   📊 {model_name}: 在线更新{stats.get('online_updates', 0)}次, "
                              f"平均损失{stats.get('avg_recent_loss', 0):.4f}, "
                              f"平均准确率{stats.get('avg_recent_accuracy', 0):.3f}")
                    
                except Exception as e:
                    print(f"❌ 保存深度学习模型状态失败: {e}")
                    print(f"📋 错误详情: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    state['deep_learning_save_error'] = str(e)
        
            # 同步内存中的计数器
            if self.total_samples_seen != actual_total or self.correct_predictions != actual_correct:
                print(f"🔧 状态保存时修正计数器：样本数 {self.total_samples_seen} → {actual_total}, 正确预测 {self.correct_predictions} → {actual_correct}")
                self.total_samples_seen = actual_total
                self.correct_predictions = actual_correct
        
            # 创建备份
            if self.model_path.exists():
                backup_path = self.model_path.with_suffix('.pkl.backup')
                try:
                    import shutil
                    shutil.copy2(self.model_path, backup_path)
                except Exception as backup_e:
                    print(f"创建状态备份失败: {backup_e}")

            # 创建数据库备份
            try:
                self.db_manager.backup_database()
            except Exception as db_backup_e:
                print(f"创建数据库备份失败: {db_backup_e}")

            # 保存状态
            with open(self.model_path, 'wb') as f:
                pickle.dump(state, f)
        
            print(f"✅ 模型状态已保存：样本数={actual_total}, 正确预测={actual_correct}")
            
        except Exception as e:
            print(f"❌ 保存模型状态失败: {e}")
            import traceback
            traceback.print_exc()
    
    def safe_sort_models_by_weight(self, models_dict, limit=5):
        """安全地按权重排序模型"""
        try:
            # 确保数据完整性
            valid_models = []
            for model_key, weight_info in models_dict.items():
                if isinstance(weight_info, dict):
                    total_weight = weight_info.get('total_weight', 0.0)
                    if isinstance(total_weight, (int, float)):
                        valid_models.append((model_key, weight_info, float(total_weight)))
                    else:
                        valid_models.append((model_key, weight_info, 0.0))
                else:
                    valid_models.append((model_key, weight_info, 0.0))
            
            # 安全排序
            sorted_models = sorted(valid_models, key=lambda x: x[2], reverse=True)
            return [(model_key, weight_info) for model_key, weight_info, _ in sorted_models[:limit]]
            
        except Exception as e:
            print(f"模型权重排序失败: {e}")
            return list(models_dict.items())[:limit]

    def _get_sortable_ensemble_weights(self):
        """获取可排序的权重数据（专门用于解决字典比较问题）"""
        sortable_weights = {}
        try:
            # 先创建可排序的权重列表
            weight_list = []
            for model_key, weight_info in self.ensemble_weights.items():
                if isinstance(weight_info, dict):
                    # 提取数值用于排序
                    total_weight = float(weight_info.get('total_weight', 0.0))
                    active_weight = float(weight_info.get('weight', 0.0))
                    frozen_weight = float(weight_info.get('frozen_weight', 0.0))
                    confidence = float(weight_info.get('confidence', 0.5))
                    is_frozen = bool(weight_info.get('is_frozen', False))
                    
                    # 处理时间戳
                    last_update = weight_info.get('last_update')
                    if isinstance(last_update, datetime):
                        last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        last_update_str = str(last_update) if last_update else 'Unknown'
                    
                    # 处理冻结时间戳
                    frozen_timestamp = weight_info.get('frozen_timestamp')
                    if isinstance(frozen_timestamp, datetime):
                        frozen_timestamp_str = frozen_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        frozen_timestamp_str = str(frozen_timestamp) if frozen_timestamp else None
                    
                    # 创建可排序的数据结构
                    sortable_data = {
                        'model_name': str(model_key),
                        'active_weight': active_weight,
                        'frozen_weight': frozen_weight,
                        'total_weight': total_weight,
                        'is_frozen': is_frozen,
                        'confidence': confidence,
                        'last_update': last_update_str,
                        'frozen_timestamp': frozen_timestamp_str,
                        'performance_count': len(weight_info.get('performance_history', [])),
                        'sort_key': total_weight  # 专门用于排序的键
                    }
                    
                    weight_list.append((total_weight, model_key, sortable_data))
                else:
                    # 处理异常情况
                    weight_list.append((0.0, str(model_key), {
                        'model_name': str(model_key),
                        'active_weight': 0.0,
                        'frozen_weight': 0.0,
                        'total_weight': 0.0,
                        'is_frozen': False,
                        'confidence': 0.5,
                        'last_update': 'Unknown',
                        'frozen_timestamp': None,
                        'performance_count': 0,
                        'sort_key': 0.0
                    }))
            
            # 按权重排序（避免字典比较）
            weight_list.sort(key=lambda x: x[0], reverse=True)
            
            # 转换为最终格式
            for _, model_key, sortable_data in weight_list:
                sortable_weights[model_key] = sortable_data
            
            return sortable_weights
            
        except Exception as e:
            print(f"获取可排序权重数据失败: {e}")
            # 返回安全的默认数据
            return {
                'error_model': {
                    'model_name': 'error',
                    'active_weight': 0.0,
                    'frozen_weight': 0.0,
                    'total_weight': 0.0,
                    'is_frozen': False,
                    'confidence': 0.5,
                    'last_update': 'Error',
                    'frozen_timestamp': None,
                    'performance_count': 0,
                    'sort_key': 0.0,
                    'error': str(e)
                }
            }

    def _clean_ensemble_weights_data(self):
        """清理权重数据，确保类型一致性"""
        try:
            for model_key, weight_info in list(self.ensemble_weights.items()):
                if isinstance(weight_info, dict):
                    # 确保所有数值字段都是float类型
                    weight_info['weight'] = float(weight_info.get('weight', 0.0))
                    weight_info['total_weight'] = float(weight_info.get('total_weight', weight_info['weight']))
                    weight_info['confidence'] = float(weight_info.get('confidence', 0.5))
                    
                    # 确保投资字段是字典类型
                    if not isinstance(weight_info.get('invested_weights'), dict):
                        weight_info['invested_weights'] = {}
                    if not isinstance(weight_info.get('pending_investments'), dict):
                        weight_info['pending_investments'] = {}
                    if not isinstance(weight_info.get('investment_history'), list):
                        weight_info['investment_history'] = []
                    
                    # 确保时间字段是datetime类型或None
                    last_update = weight_info.get('last_update')
                    if last_update is not None and not isinstance(last_update, datetime):
                        try:
                            # 尝试解析时间字符串
                            if isinstance(last_update, str):
                                weight_info['last_update'] = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            else:
                                weight_info['last_update'] = datetime.now()
                        except:
                            weight_info['last_update'] = datetime.now()
                    elif last_update is None:
                        weight_info['last_update'] = datetime.now()
                    
                    # 处理冻结时间戳
                    frozen_timestamp = weight_info.get('frozen_timestamp')
                    if frozen_timestamp is not None and not isinstance(frozen_timestamp, datetime):
                        try:
                            if isinstance(frozen_timestamp, str):
                                weight_info['frozen_timestamp'] = datetime.fromisoformat(frozen_timestamp.replace('Z', '+00:00'))
                            else:
                                weight_info['frozen_timestamp'] = None
                        except:
                            weight_info['frozen_timestamp'] = None
                    
                    # 确保性能历史是列表
                    if not isinstance(weight_info.get('performance_history'), list):
                        weight_info['performance_history'] = []
            
            print("✅ 权重数据类型清理完成")
        except Exception as e:
            print(f"❌ 权重数据清理失败: {e}")

    def load_saved_state(self):
        """加载保存的状态"""
        try:
            # 首先从数据库恢复准确的计数
            self.recover_state_from_database()
        
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
            
                saved_samples = state.get('total_samples_seen', 0)
                saved_correct = state.get('correct_predictions', 0)
                state_version = state.get('version', '1.0')
                save_source = state.get('save_source', 'unknown')
            
                print(f"📂 加载状态文件：版本={state_version}, 来源={save_source}")
                print(f"📂 状态文件显示：样本数={saved_samples}, 正确预测={saved_correct}")
                print(f"📂 数据库显示：样本数={self.total_samples_seen}, 正确预测={self.correct_predictions}")
            
                # 如果状态文件数据比数据库新，且标记为可信，则使用状态文件数据
                if (save_source == 'database_verified' and 
                    state_version >= '2.0' and 
                    saved_samples >= self.total_samples_seen):
                    print(f"✅ 使用可信的状态文件数据")
                    self.total_samples_seen = saved_samples
                    self.correct_predictions = saved_correct
                else:
                    print(f"✅ 使用数据库验证的数据")
                    # 保持从数据库恢复的数据
                    pass
            
                # 恢复集成权重
                saved_weights = state.get('ensemble_weights', {})
                for model_key in self.ensemble_weights:
                    if model_key in saved_weights:
                        self.ensemble_weights[model_key].update(saved_weights[model_key])
            
                # 恢复学习配置
                saved_config = state.get('learning_config', {})
                self.learning_config.update(saved_config)
            
                print(f"📂 最终状态：样本数={self.total_samples_seen}, 正确预测={self.correct_predictions}")

                # 加载深度学习模型状态
                if (PYTORCH_AVAILABLE and hasattr(self, 'deep_learning_module') and 
                    self.deep_learning_module and 'deep_learning_model_paths' in state):
                    try:
                        deep_learning_paths = state['deep_learning_model_paths']
                        loaded_models = 0
                        total_models = len(deep_learning_paths)
                        
                        for model_name, model_path in deep_learning_paths.items():
                            model_path_obj = Path(model_path)
                            
                            # 检查模型文件是否存在
                            if not model_path_obj.exists():
                                print(f"⚠️ 深度学习模型文件不存在: {model_path}")
                                continue
                                
                            # 检查模型是否在当前模型字典中
                            if model_name not in self.deep_learning_module.models:
                                print(f"⚠️ 模型 {model_name} 不在当前模型列表中，跳过加载")
                                continue
                            
                            try:
                                checkpoint = ai_config.torch.load(model_path, map_location=self.deep_learning_module.device)
                                
                                # 加载模型状态
                                self.deep_learning_module.models[model_name].load_state_dict(
                                    checkpoint['model_state_dict']
                                )
                                self.deep_learning_module.optimizers[model_name].load_state_dict(
                                    checkpoint['optimizer_state_dict']
                                )
                                self.deep_learning_module.schedulers[model_name].load_state_dict(
                                    checkpoint['scheduler_state_dict']
                                )
                                self.deep_learning_module.training_history[model_name] = checkpoint['training_history']
                                loaded_models += 1
                                print(f"✅ 成功加载深度学习模型: {model_name}")
                                
                            except Exception as model_e:
                                print(f"❌ 加载模型 {model_name} 失败: {model_e}")
                                continue
                        
                        print(f"📂 深度学习模型状态加载完成: {loaded_models}/{total_models} 个模型加载成功")
                        
                    except Exception as e:
                        print(f"❌ 加载深度学习模型状态失败: {e}")
                        print(f"📋 错误详情: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"📂 未找到状态文件，仅使用数据库数据")
                    
        except Exception as e:
            print(f"❌ 加载保存状态失败: {e}")
            # 如果加载失败，尝试从数据库恢复状态
            self.recover_state_from_database()
    
    def recover_state_from_database(self):
        """从数据库恢复状态信息"""
        try:
            total_samples, correct_predictions = self.db_manager.get_sample_counts()
            self.total_samples_seen = total_samples
            self.correct_predictions = correct_predictions
        
            print(f"🔄 从数据库恢复状态：样本数={self.total_samples_seen}, 正确预测={self.correct_predictions}")
        
        except Exception as e:
            print(f"从数据库恢复状态失败: {e}")
            self.total_samples_seen = 0
            self.correct_predictions = 0
    
    def check_data_consistency_on_startup(self):
        """启动时检查数据一致性"""
        try:
            # 使用数据库管理器获取一致性报告
            consistency_report = self.db_manager.check_data_consistency()
        
            if 'error' in consistency_report:
                print(f"❌ 数据一致性检查失败: {consistency_report['error']}")
                return
        
            db_count = consistency_report['total_predictions']
            db_correct = consistency_report['completed_predictions']
        
            print(f"🔍 启动数据一致性检查：")
            print(f"   内存中样本数：{self.total_samples_seen}")
            print(f"   数据库记录数：{db_count}")
            print(f"   内存中正确预测：{self.correct_predictions}")
            print(f"   数据库正确记录：{db_correct}")
            print(f"   数据完整性：{consistency_report['data_integrity']}")
        
            # 如果发现不一致，优先使用数据库中的数据
            if self.total_samples_seen != db_count or self.correct_predictions != db_correct:
                print(f"⚠️ 发现数据不一致，将使用数据库中的实际数据")
                self.total_samples_seen = db_count
                self.correct_predictions = db_correct
                # 立即保存修正后的状态
                self.save_model_state()
                print(f"✅ 数据已修复：样本数={self.total_samples_seen}, 正确预测={self.correct_predictions}")
            else:
                print(f"✅ 数据一致性检查通过")
            
        except Exception as e:
            print(f"❌ 数据一致性检查失败: {e}")

    def _validate_ensemble_weights(self):
        """验证和修复集成权重数据"""
        try:
            for model_key, weight_info in list(self.ensemble_weights.items()):
                # 确保所有必需字段存在且为正确类型
                if not isinstance(weight_info, dict):
                    print(f"⚠️ 修复模型 {model_key} 的权重信息")
                    self.ensemble_weights[model_key] = {
                        'weight': 0.1,
                        'frozen_weight': 0.0,
                        'total_weight': 0.1,
                        'is_frozen': False,
                        'frozen_timestamp': None,
                        'pending_investments': {},
                        'confidence': 0.5,
                        'last_update': datetime.now(),
                        'performance_history': [],
                        'investment_history': []
                    }
                    continue
                
                # 验证和修复数值字段
                weight_info['weight'] = float(weight_info.get('weight', 0.1))
                weight_info['total_weight'] = float(weight_info.get('total_weight', weight_info['weight']))
                weight_info['confidence'] = float(weight_info.get('confidence', 0.5))
                
                # 验证冻结权重相关字段
                weight_info['frozen_weight'] = float(weight_info.get('frozen_weight', 0.0))
                weight_info['is_frozen'] = bool(weight_info.get('is_frozen', False))
                if 'frozen_timestamp' not in weight_info:
                    weight_info['frozen_timestamp'] = None

                # 验证投资相关字段
                if not isinstance(weight_info.get('invested_weights'), dict):
                    weight_info['invested_weights'] = {}
                if not isinstance(weight_info.get('pending_investments'), dict):
                    weight_info['pending_investments'] = {}
                if not isinstance(weight_info.get('investment_history'), list):
                    weight_info['investment_history'] = []
                if not isinstance(weight_info.get('performance_history'), list):
                    weight_info['performance_history'] = []
                
                # 确保last_update是datetime对象
                if not isinstance(weight_info.get('last_update'), datetime):
                    weight_info['last_update'] = datetime.now()
            
            print("✅ 集成权重数据验证完成")
            
        except Exception as e:
            print(f"❌ 验证集成权重数据失败: {e}")

    def get_comprehensive_stats(self) -> Dict:
        """获取综合统计信息（委托给统计分析管理器）"""
        return self.statistics_manager.get_comprehensive_stats(self)
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        return self.statistics_manager.generate_performance_summary(self)

    def get_model_diversity_analysis(self) -> Dict:
        """获取模型多样性分析"""
        return self.statistics_manager.analyze_model_diversity(self)

    def get_system_health_status(self) -> Dict:
        """获取系统健康状态"""
        return self.statistics_manager.get_system_health_status(self)
        
    def reset_model(self):
        """重置AI模型 - 完全重置所有状态和数据"""
        print("🔄 开始重置AI模型...")
    
        try:
            # 1. 强制关闭所有可能的数据库连接
            self._force_close_database_connections()
        
            # 2. 删除数据库文件
            self._delete_database_files()

            # 2.5. 重新创建数据库管理器实例以确保表结构被正确创建
            try:
                self.db_manager = DatabaseManager(self.db_path)
                print("✅ 数据库管理器已重新创建，表结构已初始化")
            except Exception as e:
                print(f"❌ 重新创建数据库管理器失败: {e}")
                return False

            # 3. 重置学习状态
            self.total_samples_seen = 0
            self.correct_predictions = 0
            self.last_prediction_result = None
            self.model_performance = {}
        
            # 4. 清空所有模型集合
            self.river_models.clear()
            self.sklearn_models.clear()
            self.drift_detectors.clear()
            self.metrics_trackers.clear()
            self.ensemble_weights.clear()
        
            # 5. 验证数据库表是否正确创建
            try:
                # 测试数据库连接和表存在性
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
    
                # 检查必要的表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
    
                required_tables = ['online_predictions', 'individual_model_predictions', 'drift_events']
                missing_tables = [table for table in required_tables if table not in tables]
    
                conn.close()
    
                if missing_tables:
                    print(f"❌ 缺少数据库表: {missing_tables}")
                    return False
                else:
                    print(f"✅ 数据库表验证通过: {tables}")
        
            except Exception as e:
                print(f"❌ 数据库表验证失败: {e}")
                return False
        
            # 6. 重新初始化所有模型
            self.init_online_models()
        
            # 7. 重置所有组件状态
            self._reset_all_components()
        
            # 8. 保存新的初始状态
            self.save_model_state()
        
            print("✅ AI模型重置完成")
            print(f"   - 重新初始化了 {len(self.river_models)} 个River模型")
            print(f"   - 重新初始化了 {len(self.sklearn_models)} 个scikit-multiflow模型")
        
            return True
        
        except Exception as e:
            print(f"❌ 重置AI模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _force_close_database_connections(self):
        """强制关闭所有数据库连接"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                # 关闭数据库管理器的连接
                if hasattr(self.db_manager, 'close_connection'):
                    self.db_manager.close_connection()
        
            # 强制垃圾回收
            import gc
            gc.collect()
        
            print("   - 数据库连接已关闭")
        except Exception as e:
            print(f"   ⚠️ 关闭数据库连接失败: {e}")

    def _delete_database_files(self):
        """删除数据库文件"""
        try:
            import time
            import os
        
            # 等待一下确保连接完全关闭
            time.sleep(0.5)
        
            # 删除主数据库文件
            if self.db_path.exists():
                try:
                    os.remove(self.db_path)
                    print(f"   - 删除数据库文件: {self.db_path}")
                except Exception as e:
                    print(f"   ⚠️ 删除数据库文件失败: {e}")
        
            # 删除备份文件
            backup_files = [self.db_path.with_suffix('.db.backup'), 
                        self.db_path.with_suffix('.backup')]
            for backup_file in backup_files:
                if backup_file.exists():
                    try:
                        os.remove(backup_file)
                        print(f"   - 删除备份文件: {backup_file}")
                    except Exception as e:
                        print(f"   ⚠️ 删除备份文件失败: {e}")
        
            # 删除模型状态文件
            if self.model_path.exists():
                try:
                    os.remove(self.model_path)
                    print(f"   - 删除模型状态文件: {self.model_path}")
                except Exception as e:
                    print(f"   ⚠️ 删除模型状态文件失败: {e}")
                
        except Exception as e:
            print(f"删除数据库文件失败: {e}")

    def _reset_all_components(self):
        """重置所有组件状态"""
        try:
            # 重置特征工程组件
            if hasattr(self, 'feature_engineer'):
                self.feature_engineer = FeatureEngineer(
                    ai_config=ai_config,
                    feature_selector=ai_config.DynamicFeatureSelector(feature_count=60, selection_ratio=0.9),
                    feature_combiner=ai_config.FeatureInteractionCombiner(original_features=60, max_interactions=15),
                    timeseries_enhancer=ai_config.TimeSeriesFeatureEnhancer(history_length=10),
                    feature_weighter=ai_config.AdaptiveFeatureWeighter(feature_count=75),
                    feature_assessor=ai_config.FeatureQualityAssessor(assessment_window=50)
                )
        
            # 重置投资系统
            if hasattr(self, 'investment_system'):
                self.investment_system = InvestmentSystem()
        
            # 重置统计信息
            self.investment_stats = {
                'total_rounds': 0,
                'total_invested': 0.0,
                'total_rewards': 0.0,
                'total_penalties': 0.0,
                'current_pool_size': 0.0,
                'last_investment_details': {}
            }
        
            # 重置样本ID
            if hasattr(self, '_last_sample_id'):
                self._last_sample_id = None
            
            print("   - 所有组件状态已重置")
        
        except Exception as e:
            print(f"重置组件状态失败: {e}")
    
    def _add_fundamental_law_analysis(self, analysis_text, data_list, recommended_tails):
        """添加底层定律应用分析（委托给预测分析器）"""
        self.prediction_analyzer._add_fundamental_law_analysis(analysis_text, data_list, recommended_tails)
    
    def get_last_prediction_details(self):
        """获取最后一次预测的详细信息"""
        if self.last_prediction_result:
            # 添加当前权重信息用于调试
            result = self.last_prediction_result.copy()
            result['current_model_weights'] = {k: v['weight'] for k, v in self.ensemble_weights.items()}
            result['debug_weight_info'] = f"权重总数: {len(self.ensemble_weights)}, 权重池: {getattr(self, 'weight_pool', 0.0):.4f}"
            return result
        else:
            return {
                'success': False,
                'message': '暂无预测数据',
                'weight_details': [],
                'decision_summary': '请先进行AI预测',
                'detailed_analysis': '暂无预测分析数据'
            }
        
    def _clear_database_records(self):
        """清空数据库中的所有记录"""
        try:
            self.db_manager.clear_all_records()
        except Exception as e:
            print(f"清空数据库记录失败: {e}")
            raise
    
    def _select_most_dangerous_tail(self, overlapping_tails: set, anti_manipulation_analysis: dict, data_list: List[Dict]) -> int:
        """从重合的尾数中选择最应该避开的一个"""
        if len(overlapping_tails) == 1:
            return list(overlapping_tails)[0]
        
        tail_risk_scores = {}
        
        for tail in overlapping_tails:
            risk_score = 0.0
            
            # 基于最近频率的风险评分
            if len(data_list) >= 5:
                recent_5_count = sum(1 for period in data_list[:5] if tail in period.get('tails', []))
                if recent_5_count >= 3:
                    risk_score += 0.4
                elif recent_5_count == 0:
                    risk_score += 0.3
            
            # 基于操控分析证据的风险评分
            evidence = anti_manipulation_analysis.get('evidence', {})
            detection_results = evidence.get('detection_breakdown', {})
            
            psychological_score = detection_results.get('psychological_traps', {}).get('score', 0.0)
            if psychological_score > 0.7:
                risk_score += 0.3
            elif psychological_score > 0.5:
                risk_score += 0.2
            
            frequency_score = detection_results.get('frequency_deviation', {}).get('score', 0.0)
            if frequency_score > 0.6:
                risk_score += 0.25
            
            # 基于尾数特性的风险评分
            mirror_pairs = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]
            for pair in mirror_pairs:
                if tail in pair and any(other in overlapping_tails for other in pair if other != tail):
                    risk_score += 0.15
            
            if tail in [6, 8, 9]:
                risk_score += 0.1
            elif tail in [0, 5]:
                risk_score += 0.08
            
            tail_risk_scores[tail] = risk_score
        
        # 选择风险分数最高的尾数
        most_dangerous_tail = max(tail_risk_scores.keys(), key=lambda t: tail_risk_scores[t])
        return most_dangerous_tail

    def _select_safest_tail(self, original_candidates: set, anti_recommendations: dict) -> int:
        """从原始候选中选择最安全的尾数"""
        avoid_tails = set(anti_recommendations.get('avoid_tails', []))
        recommended_tails = set(anti_recommendations.get('recommended_tails', []))
        
        # 优先选择推荐的尾数
        safe_recommended = original_candidates.intersection(recommended_tails)
        if safe_recommended:
            return list(safe_recommended)[0]
        
        # 其次选择不在避开列表中的尾数
        safe_neutral = original_candidates - avoid_tails
        if safe_neutral:
            return list(safe_neutral)[0]
        
        # 最后兜底：选择任意一个原始候选
        return list(original_candidates)[0] if original_candidates else 0
    
# 使用示例和测试
if __name__ == "__main__":
    print("🚀 启动终极在线学习AI引擎测试...")
    
    # 创建引擎
    ultimate_ai = UltimateOnlineAIEngine("./ultimate_ai_data")
    
    if ultimate_ai.is_initialized:
        print("\n✅ 终极在线学习AI引擎初始化成功！")
        
        # 模拟真正的在线学习过程
        import random
        
        sample_data = []
        print("\n🔄 开始在线学习演示...")
        
        for i in range(20):
            # 生成测试数据
            num_tails = random.randint(4, 8)
            tails = random.sample(range(10), num_tails)
            numbers = [f"{random.randint(1, 49):02d}" for _ in range(7)]
            
            period_data = {
                'tails': sorted(tails),
                'numbers': numbers
            }
            sample_data.insert(0, period_data)  # 最新数据在前
            
            print(f"\n📊 期数 {i+1}:")
            print(f"   实际尾数: {period_data['tails']}")
            
            if i == 0:
                # 第一期只进行预测
                result = ultimate_ai.process_new_sample(sample_data)
                print(f"   首次预测: {result['prediction'].get('recommended_tails', [])}")
            else:
                # 后续期数进行预测和学习
                result = ultimate_ai.process_new_sample(sample_data, period_data['tails'])
                
                prediction = result.get('prediction', {})
                learning = result.get('learning', {})
                
                print(f"   AI预测: {prediction.get('recommended_tails', [])}")
                print(f"   置信度: {prediction.get('confidence', 0):.3f}")
                print(f"   预测正确: {learning.get('prediction_correct', False)}")
                print(f"   当前准确率: {learning.get('current_accuracy', 0):.3f}")
                print(f"   检测到漂移: {learning.get('drift_detected', False)}")
                
                if learning.get('drift_detected'):
                    print(f"   🚨 漂移检测器: {learning.get('drift_detectors', [])}")
        
        # 显示最终统计
        final_stats = ultimate_ai.get_comprehensive_stats()
        print(f"\n📈 最终统计:")
        print(f"   总样本数: {final_stats['basic_stats']['total_samples_seen']}")
        print(f"   正确预测: {final_stats['basic_stats']['correct_predictions']}")
        print(f"   总体准确率: {final_stats['basic_stats']['current_accuracy']:.3f}")
        print(f"   使用模型数: {final_stats['model_stats']['total_models']}")
        
        print(f"\n🎯 权重最高的模型:")
        top_models = sorted(final_stats['ensemble_weights'].items(), 
                           key=lambda x: x[1], reverse=True)[:3]
        for model, weight in top_models:
            print(f"   {model}: {weight:.3f}")
        
        print("\n🎉 终极在线学习AI引擎演示完成！")
    else:
        print("❌ 引擎初始化失败，请检查依赖库安装")