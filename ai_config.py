# ai_config.py - AI引擎配置管理模块

import json
import numpy as np
import os
import math
import warnings
warnings.filterwarnings('ignore')

class AIConfig:
    """AI引擎配置管理器"""
    
    def __init__(self):
        self.setup_dependencies()
    
    def setup_dependencies(self):
        """设置和检查所有依赖"""
        # 检查NumPy版本兼容性
        try:
            numpy_version = np.__version__
            if numpy_version.startswith('2.'):
                print(f"⚠️ 检测到NumPy {numpy_version}，建议降级到1.x版本以获得最佳兼容性")
                print("执行: pip install 'numpy<2.0'")
            else:
                print(f"✅ NumPy版本 {numpy_version} 兼容")
        except Exception as e:
            print(f"NumPy版本检查失败: {e}")

        # 设置PyTorch相关配置
        self._setup_pytorch()
        
        # 设置River相关配置
        self._setup_river()
        
        # 设置scikit-multiflow相关配置
        self._setup_skmultiflow()
        
        # 设置Deep-River配置（已禁用）
        self._setup_deep_river()
        
        # 设置本地模型配置
        self._setup_local_models()
    
    def _setup_pytorch(self):
        """设置PyTorch配置"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import MinMaxScaler
            
            self.PYTORCH_AVAILABLE = True
            # 保存torch模块为实例属性
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.F = F
            self.DataLoader = DataLoader
            self.TensorDataset = TensorDataset
            self.MinMaxScaler = MinMaxScaler
            print("✅ PyTorch导入成功")
            
            # 检查GPU可用性
            try:
                if torch.cuda.is_available():
                    self.DEVICE = torch.device('cuda')
                    print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
                else:
                    self.DEVICE = torch.device('cpu')
                    print("ℹ️ 使用CPU进行训练")
            except Exception as e:
                self.DEVICE = torch.device('cpu')
                print(f"⚠️ CUDA检测失败，使用CPU: {e}")
                
        except ImportError as e:
            self.PYTORCH_AVAILABLE = False
            self.DEVICE = 'cpu'
            self.torch = None
            print(f"❌ PyTorch导入失败: {e}")
            if "numpy" in str(e).lower():
                print("💡 这可能是NumPy版本兼容性问题，请尝试:")
                print("   pip uninstall numpy")
                print("   pip install 'numpy<2.0'")
                print("   pip install torch --force-reinstall")
        except Exception as e:
            self.PYTORCH_AVAILABLE = False
            self.DEVICE = 'cpu'
            self.torch = None
            print(f"❌ PyTorch初始化失败: {e}")
    
    def _setup_river(self):
        """设置River配置"""
        try:
            from river import (
                tree, linear_model, ensemble, drift, metrics, compose, 
                preprocessing, naive_bayes, neighbors, anomaly
            )
            
            # 基础分类器导入
            from river.tree import HoeffdingTreeClassifier
            
            # 漂移检测器导入（兼容不同版本）
            self.drift_detectors = {}
            try:
                from river.drift import ADWIN
                self.drift_detectors['ADWIN'] = ADWIN
            except ImportError:
                pass
            
            try:
                from river.drift import PageHinkley
                self.drift_detectors['PageHinkley'] = PageHinkley
            except ImportError:
                pass
            
            try:
                from river.drift import KSWIN
                self.drift_detectors['KSWIN'] = KSWIN
            except ImportError:
                pass
            
            try:
                from river.drift import HDDM_A
                self.drift_detectors['HDDM_A'] = HDDM_A
            except ImportError:
                pass
            
            try:
                from river.drift import HDDM_W
                self.drift_detectors['HDDM_W'] = HDDM_W
            except ImportError:
                pass
            
            # 集成学习器导入
            self.ensemble_models = {}

            # 尝试不同的导入路径
            ensemble_imports = [
                ('AdaptiveRandomForestClassifier', ['river.ensemble', 'river.forest']),
                ('OnlineBaggingClassifier', ['river.ensemble']),
                ('StreamingRandomPatchesClassifier', ['river.ensemble']),
                ('BaggingClassifier', ['river.ensemble']),  # 备选名称
            ]

            for model_name, import_paths in ensemble_imports:
                for import_path in import_paths:
                    try:
                        module = __import__(import_path, fromlist=[model_name])
                        if hasattr(module, model_name):
                            self.ensemble_models[model_name] = getattr(module, model_name)
                            break
                    except ImportError:
                        continue

            print(f"   发现集成模型: {list(self.ensemble_models.keys())}")
            
            # 尝试导入River的基础集成模型
            try:
                from river.ensemble import BaggingClassifier as RiverBaggingClassifier
                self.ensemble_models['RiverBaggingClassifier'] = RiverBaggingClassifier
                print(f"   ✓ 成功导入River基础装袋分类器")
            except ImportError:
                print(f"   ✗ River基础装袋分类器不可用")
            
            # 尝试其他可能的集成模型
            try:
                from river.ensemble import AdaBoostClassifier
                self.ensemble_models['AdaBoostClassifier'] = AdaBoostClassifier
                print(f"   ✓ 成功导入AdaBoost分类器")
            except ImportError:
                print(f"   ✗ AdaBoost分类器不可用")

            # 尝试多种装袋分类器名称（不同River版本可能不同）
            bagging_names = ['BaggingClassifier', 'OnlineBaggingClassifier', 'StreamingBaggingClassifier']
            for bagging_name in bagging_names:
                try:
                    from river.ensemble import BaggingClassifier as RiverBagging
                    self.ensemble_models['OnlineBaggingClassifier'] = RiverBagging
                    print(f"   ✓ 成功导入装袋分类器: {bagging_name}")
                    break
                except ImportError:
                    try:
                        # 尝试直接从ensemble导入
                        ensemble_module = __import__('river.ensemble', fromlist=[bagging_name])
                        if hasattr(ensemble_module, bagging_name):
                            self.ensemble_models['OnlineBaggingClassifier'] = getattr(ensemble_module, bagging_name)
                            print(f"   ✓ 成功导入装袋分类器: {bagging_name}")
                            break
                    except ImportError:
                        continue
            else:
                print(f"   ⚠️ 所有装袋分类器导入尝试都失败，将跳过该模型")
            
            try:
                from river.ensemble import StreamingRandomPatchesClassifier
                self.ensemble_models['StreamingRandomPatchesClassifier'] = StreamingRandomPatchesClassifier
            except ImportError:
                pass
            
            # 线性模型导入
            from river.linear_model import LogisticRegression
            self.LogisticRegression = LogisticRegression
            try:
                from river.linear_model import Perceptron
                self.Perceptron = Perceptron
            except ImportError:
                self.Perceptron = LogisticRegression  # fallback
            
            # KNN导入
            try:
                from river.neighbors import KNNClassifier
                self.KNNClassifier = KNNClassifier
            except ImportError:
                self.KNNClassifier = None
            
            # 其他必要的导入
            self.HoeffdingTreeClassifier = HoeffdingTreeClassifier
            self.tree = tree
            self.linear_model = linear_model
            self.ensemble = ensemble
            self.drift = drift
            self.metrics = metrics
            self.compose = compose
            self.preprocessing = preprocessing
            self.naive_bayes = naive_bayes
            self.neighbors = neighbors
            self.anomaly = anomaly
            
            self.RIVER_AVAILABLE = True
            print("✅ River在线学习库导入成功")
            print(f"   可用漂移检测器: {list(self.drift_detectors.keys())}")
            print(f"   可用集成模型: {list(self.ensemble_models.keys())}")
            print(f"   ⚠️ 已禁用River原生朴素贝叶斯，使用本地实现替代")
            
        except ImportError as e:
            self.RIVER_AVAILABLE = False
            print(f"ℹ️ River库不可用，将使用基础实现: {e}")
            if "numpy" in str(e).lower():
                print("💡 这可能是NumPy版本兼容性问题，请尝试:")
                print("   pip install 'numpy<2.0' river --force-reinstall")
            else:
                print("如需完整功能，请安装River库: pip install river")
        except Exception as e:
            self.RIVER_AVAILABLE = False
            print(f"ℹ️ River库加载异常，将使用基础实现: {e}")
    
    def _setup_skmultiflow(self):
        """设置scikit-multiflow配置"""
        try:
            import importlib
            
            # 动态导入以避免IDE错误
            skmultiflow_trees = importlib.import_module('skmultiflow.trees')
            skmultiflow_drift = importlib.import_module('skmultiflow.drift_detection')
            skmultiflow_lazy = importlib.import_module('skmultiflow.lazy')
            skmultiflow_ensemble = importlib.import_module('skmultiflow.ensemble')
            
            # 获取需要的类
            self.SKMHoeffdingTree = getattr(skmultiflow_trees, 'HoeffdingTreeClassifier')
            self.SKMHoeffdingAdaptive = getattr(skmultiflow_trees, 'HoeffdingAdaptiveTreeClassifier')
            self.ExtremelyFastDecisionTreeClassifier = getattr(skmultiflow_trees, 'ExtremelyFastDecisionTreeClassifier')
            
            self.SKADWIN = getattr(skmultiflow_drift, 'ADWIN')
            self.DDM = getattr(skmultiflow_drift, 'DDM')
            self.EDDM = getattr(skmultiflow_drift, 'EDDM')
            self.SKPageHinkley = getattr(skmultiflow_drift, 'PageHinkley')
            
            self.SKMKNNClassifier = getattr(skmultiflow_lazy, 'KNNClassifier')
            self.OnlineAdaBoostClassifier = getattr(skmultiflow_ensemble, 'OnlineAdaBoostClassifier')
            self.SKMRandomPatches = getattr(skmultiflow_ensemble, 'StreamingRandomPatchesClassifier')
            
            self.SKMULTIFLOW_AVAILABLE = True
            print("✅ scikit-multiflow导入成功")
        except ImportError as e:
            self.SKMULTIFLOW_AVAILABLE = False
            print(f"ℹ️ scikit-multiflow库不可用（可选）")
    
    def _setup_deep_river(self):
        """设置Deep River配置（已禁用）"""
        self.DEEP_RIVER_AVAILABLE = False
        print("ℹ️ Deep-River库已禁用，使用稳定的在线学习模型替代")
    
    def _setup_local_models(self):
        """设置本地模型配置"""
        try:
            from sf import (
                LocalHoeffdingTree, LocalLogisticRegression, LocalNaiveBayes,
                LocalBaggingClassifier, LocalAdaBoostClassifier, LocalDriftDetector,
                LocalStandardScaler, LocalPipeline, LocalHistoricalPatternMatcher,
                MultiLevelDriftDetector, StatisticalDriftDetector,
                EnsembleDriftDetector, DynamicFeatureSelector, FeatureInteractionCombiner,
                TimeSeriesFeatureEnhancer, AdaptiveFeatureWeighter, FeatureQualityAssessor,
                ADWINDriftDetector, PageHinkleyDriftDetector, CUSUMDriftDetector
            )
            self.LOCAL_MODELS_AVAILABLE = True
            
            # 存储本地模型类
            self.LocalHoeffdingTree = LocalHoeffdingTree
            self.LocalLogisticRegression = LocalLogisticRegression
            self.LocalNaiveBayes = LocalNaiveBayes
            self.LocalBaggingClassifier = LocalBaggingClassifier
            self.LocalAdaBoostClassifier = LocalAdaBoostClassifier
            self.LocalDriftDetector = LocalDriftDetector
            self.LocalStandardScaler = LocalStandardScaler
            self.LocalPipeline = LocalPipeline
            self.LocalHistoricalPatternMatcher = LocalHistoricalPatternMatcher
            self.MultiLevelDriftDetector = MultiLevelDriftDetector
            self.StatisticalDriftDetector = StatisticalDriftDetector
            self.EnsembleDriftDetector = EnsembleDriftDetector
            self.DynamicFeatureSelector = DynamicFeatureSelector
            self.FeatureInteractionCombiner = FeatureInteractionCombiner
            self.TimeSeriesFeatureEnhancer = TimeSeriesFeatureEnhancer
            self.AdaptiveFeatureWeighter = AdaptiveFeatureWeighter
            self.FeatureQualityAssessor = FeatureQualityAssessor
            self.ADWINDriftDetector = ADWINDriftDetector
            self.PageHinkleyDriftDetector = PageHinkleyDriftDetector
            self.CUSUMDriftDetector = CUSUMDriftDetector
            
            print("✅ 本地在线学习模型导入成功")
        except ImportError as e:
            self.LOCAL_MODELS_AVAILABLE = False
            print(f"ℹ️ 本地在线学习模型不可用，将使用备用实现: {e}")
            self._setup_fallback_classes()
        except Exception as e:
            self.LOCAL_MODELS_AVAILABLE = False
            print(f"ℹ️ 本地在线学习模型加载异常，将使用备用实现: {e}")
            self._setup_fallback_classes()
    
    def _setup_fallback_classes(self):
        """设置备用类"""
        class EnsembleDriftDetector:
            def __init__(self):
                self.detected = False
            def update(self, error_rate, performance_metrics=None):
                pass
            def get_detailed_report(self):
                return {'ensemble_detected': False, 'individual_detectors': {}}
            def reset(self):
                pass
        
        class FeatureQualityAssessor:
            def __init__(self, assessment_window=50):
                pass
            def assess_feature_quality(self, features, accuracy):
                return {'overall_quality': 0.5, 'recommendations': ['备用模式：特征处理功能有限']}
        
        self.EnsembleDriftDetector = EnsembleDriftDetector
        self.FeatureQualityAssessor = FeatureQualityAssessor

# 创建全局配置实例
ai_config = AIConfig()

# 导出常用的配置变量
PYTORCH_AVAILABLE = ai_config.PYTORCH_AVAILABLE
DEVICE = ai_config.DEVICE
RIVER_AVAILABLE = ai_config.RIVER_AVAILABLE
SKMULTIFLOW_AVAILABLE = ai_config.SKMULTIFLOW_AVAILABLE
DEEP_RIVER_AVAILABLE = ai_config.DEEP_RIVER_AVAILABLE
LOCAL_MODELS_AVAILABLE = ai_config.LOCAL_MODELS_AVAILABLE

# === 特殊模型配置管理 ===

class SpecialModelsConfig:
    """特殊模型配置管理器"""
    
    def __init__(self):
        """初始化特殊模型配置"""
        # 默认启用的特殊模型列表
        self.default_enabled_models = [
            'anti_manipulation_banker_behavior',  # 反操控分析器
            'reverse_psychology_predictor',       # 反向心理学预测器
            'unpopular_digger',                   # 冷门挖掘器
            'money_flow_analyzer',                # 资金流向分析器
            'game_theory_strategist',             # 博弈论策略器  
            'manipulation_timing_detector',       # 操控时机检测器
            # 'anti_trend_hunter',                # 反趋势猎手（待实现）
            # 'crowd_psychology_analyzer'         # 群体心理分析器（待实现）
        ]
        
        # 加载用户自定义配置
        self.enabled_models = self._load_user_config()
        
        print(f"🎯 特殊模型配置加载完成")
        print(f"   启用的模型: {self.enabled_models}")
    
    def _load_user_config(self) -> list:
        """加载用户自定义配置"""
        config_file = "special_models_config.json"
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    enabled_models = user_config.get('enabled_special_models', self.default_enabled_models)
                    print(f"✅ 从配置文件加载特殊模型设置: {config_file}")
                    return enabled_models
            else:
                # 创建默认配置文件
                self._save_default_config(config_file)
                return self.default_enabled_models.copy()
                
        except Exception as e:
            print(f"⚠️ 加载特殊模型配置失败，使用默认配置: {e}")
            return self.default_enabled_models.copy()
    
    def _save_default_config(self, config_file: str):
        """保存默认配置文件"""
        try:
            default_config = {
                "enabled_special_models": self.default_enabled_models,
                "model_descriptions": {
                    "anti_manipulation_banker_behavior": "反操控分析器 - 分析庄家行为模式",
                    "reverse_psychology_predictor": "反向心理学预测器 - 基于反向思维的预测",
                    "unpopular_digger": "冷门挖掘器 - 发现被忽视的机会",
                    "money_flow_analyzer": "资金流向分析器 - 分析虚拟资金流向和赔付压力",
                    "game_theory_strategist": "博弈论策略器 - 基于博弈论的多方博弈分析",
                    "manipulation_timing_detector": "操控时机检测器 - 识别操控发生的时机",
                    "anti_trend_hunter": "反趋势猎手 - 专门打破趋势的预测",
                    "crowd_psychology_analyzer": "群体心理分析器 - 分析群体投注心理"
                },
                "configuration_notes": [
                    "可以通过修改 enabled_special_models 列表来启用/禁用特定模型",
                    "每个模型都有其独特的分析角度和策略逻辑",
                    "建议保持至少2-3个模型启用以确保预测多样性"
                ]
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            print(f"📄 已创建默认特殊模型配置文件: {config_file}")
            
        except Exception as e:
            print(f"⚠️ 创建默认配置文件失败: {e}")
    
    def is_model_enabled(self, model_name: str) -> bool:
        """检查模型是否启用"""
        return model_name in self.enabled_models
    
    def enable_model(self, model_name: str):
        """启用模型"""
        if model_name not in self.enabled_models:
            self.enabled_models.append(model_name)
            print(f"✅ 已启用特殊模型: {model_name}")
    
    def disable_model(self, model_name: str):
        """禁用模型"""
        if model_name in self.enabled_models:
            self.enabled_models.remove(model_name)
            print(f"❌ 已禁用特殊模型: {model_name}")
    
    def get_enabled_models(self) -> list:
        """获取启用的模型列表"""
        return self.enabled_models.copy()
    
    def save_current_config(self):
        """保存当前配置"""
        config_file = "special_models_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            config['enabled_special_models'] = self.enabled_models
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print(f"💾 特殊模型配置已保存")
            
        except Exception as e:
            print(f"⚠️ 保存特殊模型配置失败: {e}")

# 创建特殊模型配置实例
special_models_config = SpecialModelsConfig()

# 在 AIConfig 类中添加方法
def get_enabled_special_models():
    """获取启用的特殊模型列表"""
    return special_models_config.get_enabled_models()

# 将方法添加到全局配置实例
ai_config.get_enabled_special_models = get_enabled_special_models
ai_config.special_models_config = special_models_config

# 导出特殊模型配置
ENABLED_SPECIAL_MODELS = special_models_config.get_enabled_models()

print(f"🎯 特殊模型配置系统初始化完成")
print(f"   当前启用的特殊模型数量: {len(ENABLED_SPECIAL_MODELS)}")