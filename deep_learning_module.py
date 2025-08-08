#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习模块管理
管理LSTM和Transformer模型的训练、预测和优化
"""

import os
import math
import traceback
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from .deep_learning_models import LSTMPredictor, TransformerPredictor

class DeepLearningModule:
    """深度学习模块 - 管理LSTM和Transformer模型"""
    
    def __init__(self, input_size=60, device='cpu'):
        # 确保device是正确的torch.device对象
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
    
        self.input_size = input_size
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.scalers = {}
        self.training_history = {}
        self.batch_size = 32
        self.sequence_length = 10
        
        # 初始化模型
        self._init_models()
        
        print(f"🤖 深度学习模块初始化完成，使用设备: {device}")
    
    def _init_models(self):
        """初始化深度学习模型"""
        try:
            print(f"🔧 开始初始化深度学习模型，使用设备: {self.device}")
        
            # 确保设备可用
            if not torch.cuda.is_available() and str(self.device) != 'cpu':
                print("⚠️ CUDA不可用，切换到CPU")
                self.device = torch.device('cpu')
        
            # LSTM模型
            print("🔧 正在初始化LSTM模型...")
            lstm_model = LSTMPredictor(
                input_size=self.input_size,
                hidden_size=128,
                num_layers=2,
                output_size=10,
                dropout=0.2
            )
            self.models['lstm'] = lstm_model.to(self.device)
            print("✅ LSTM模型初始化成功")
            
            self.optimizers['lstm'] = optim.AdamW(
                self.models['lstm'].parameters(),
                lr=0.001,
                weight_decay=0.01
            )
        
            self.schedulers['lstm'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['lstm'],
                mode='min',
                factor=0.5,
                patience=10
            )
        
            # Transformer模型
            print("🔧 正在初始化Transformer模型...")
            transformer_model = TransformerPredictor(
                input_size=self.input_size,
                d_model=128,
                nhead=8,
                num_encoder_layers=4,
                dim_feedforward=512,
                dropout=0.1,
                output_size=10
            )
            self.models['transformer'] = transformer_model.to(self.device)
            print("✅ Transformer模型初始化成功")
            
            self.optimizers['transformer'] = optim.AdamW(
                self.models['transformer'].parameters(),
                lr=0.0005,
                weight_decay=0.01
            )
        
            self.schedulers['transformer'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['transformer'],
                mode='min',
                factor=0.5,
                patience=10
            )
        
            # 数据标准化器
            print("🔧 正在初始化数据标准化器...")
            for model_name in self.models.keys():
                self.scalers[model_name] = MinMaxScaler()
                self.training_history[model_name] = {
                    'loss': [],
                    'accuracy': [],
                    'best_loss': float('inf'),
                    'epochs_trained': 0
                }
        
            print(f"✅ 深度学习模型初始化成功: {list(self.models.keys())}")
            print(f"📊 初始化了 {len(self.models)} 个深度学习模型")
        
        except Exception as e:
            print(f"❌ 深度学习模型初始化失败: {e}")
            print(f"📋 错误详情: {str(e)}")
            traceback.print_exc()
            self.models = {}
    
    def prepare_sequence_data(self, features_list, labels_list):
        """准备序列数据"""
        if len(features_list) < self.sequence_length:
            return None, None
        
        sequences = []
        targets = []
        
        for i in range(len(features_list) - self.sequence_length + 1):
            # 创建序列
            sequence = features_list[i:i + self.sequence_length]
            target = labels_list[i + self.sequence_length - 1]
            
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def batch_train(self, data_list, epochs=50, validation_split=0.2):
        """批量训练深度学习模型"""
        if not self.models:
            return {'success': False, 'message': f'深度学习模型未初始化。当前模型数量: {len(self.models)}。请检查模型初始化过程是否有错误。'}
        
        if len(data_list) < self.sequence_length + 10:
            return {'success': False, 'message': f'数据不足，需要至少{self.sequence_length + 10}期数据'}
        
        try:
            print(f"🚀 开始深度学习批量训练，数据量: {len(data_list)} 期")
            
            # 准备训练数据
            features_list = []
            labels_list = []
            
            for i in range(len(data_list) - 1):
                # 使用前i+1期数据作为特征
                feature_data = data_list[i+1:]
                features = self._extract_deep_features(feature_data)
                features_list.append(features)
                
                # 使用下一期的尾数作为标签
                actual_tails = data_list[i].get('tails', [])
                # 创建多标签（10个尾数的二分类）
                label = np.zeros(10)
                for tail in actual_tails:
                    if 0 <= tail <= 9:
                        label[tail] = 1
                labels_list.append(label)
            
            # 准备序列数据
            sequences, targets = self.prepare_sequence_data(features_list, labels_list)
            if sequences is None:
                return {'success': False, 'message': '序列数据准备失败'}
            
            # 划分训练和验证集
            split_idx = int(len(sequences) * (1 - validation_split))
            train_sequences = sequences[:split_idx]
            train_targets = targets[:split_idx]
            val_sequences = sequences[split_idx:]
            val_targets = targets[split_idx:]
            
            results = {}
            
            # 训练每个模型
            for model_name, model in self.models.items():
                print(f"📚 训练 {model_name.upper()} 模型...")
                
                # 数据标准化
                scaler = self.scalers[model_name]
                train_seq_scaled = scaler.fit_transform(
                    train_sequences.reshape(-1, self.input_size)
                ).reshape(train_sequences.shape)
                val_seq_scaled = scaler.transform(
                    val_sequences.reshape(-1, self.input_size)
                ).reshape(val_sequences.shape)
                
                # 转换为张量
                train_dataset = TensorDataset(
                    torch.FloatTensor(train_seq_scaled).to(self.device),
                    torch.FloatTensor(train_targets).to(self.device)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(val_seq_scaled).to(self.device),
                    torch.FloatTensor(val_targets).to(self.device)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                
                # 训练模型
                model_result = self._train_single_model(
                    model_name, model, train_loader, val_loader, epochs
                )
                results[model_name] = model_result
            
            return {
                'success': True,
                'message': f'深度学习批量训练完成',
                'results': results,
                'train_samples': len(train_sequences),
                'val_samples': len(val_sequences)
            }
            
        except Exception as e:
            print(f"❌ 深度学习批量训练失败: {e}")
            traceback.print_exc()
            return {'success': False, 'message': f'训练失败: {str(e)}'}
    
    def _train_single_model(self, model_name, model, train_loader, val_loader, epochs):
        """训练单个模型"""
        criterion = nn.BCELoss()
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算准确率
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_targets).sum().item()
                train_total += batch_targets.numel()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == batch_targets).sum().item()
                    val_total += batch_targets.numel()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # 更新学习率
            try:
                scheduler.step(avg_val_loss)
            except Exception as scheduler_e:
                print(f"   学习率调度器更新失败: {scheduler_e}")
            
            # 记录训练历史
            self.training_history[model_name]['loss'].append(avg_val_loss)
            self.training_history[model_name]['accuracy'].append(val_acc)
            self.training_history[model_name]['epochs_trained'] += 1
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.training_history[model_name]['best_loss'] = best_val_loss
                patience_counter = 0

                # 保存最佳模型状态
                models_dir = getattr(self, 'models_dir', None)
                if models_dir:
                    try:
                        os.makedirs(models_dir, exist_ok=True)
                        model_path = os.path.join(models_dir, f'{model_name}_best.pth')
                        torch.save(model.state_dict(), model_path)
                        print(f"✅ 模型 {model_name} 最佳状态已保存到: {model_path}")
                    except Exception as save_e:
                        print(f"❌ 保存模型 {model_name} 最佳状态失败: {save_e}")
                else:
                    print(f"⚠️ 模型目录未设置，跳过模型保存")

                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"   早停触发于第 {epoch+1} 轮")
                break
            
            # 每10轮打印一次进度
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'final_val_accuracy': val_acc,
            'best_val_loss': best_val_loss,
            'epochs_completed': epoch + 1
        }
    
    def _extract_deep_features(self, data_list, window_size=15):
        """为深度学习提取特征"""
        window_size = min(window_size, len(data_list))
        recent_data = data_list[:window_size]
        
        features = []
        
        # 1. 基础频率特征（10维）
        for tail in range(10):
            count = sum(1 for period in recent_data if tail in period.get('tails', []))
            features.append(count / len(recent_data))
        
        # 2. 短期频率特征（10维）- 最近5期
        recent_5 = recent_data[:5] if len(recent_data) >= 5 else recent_data
        for tail in range(10):
            count = sum(1 for period in recent_5 if tail in period.get('tails', []))
            features.append(count / len(recent_5) if recent_5 else 0)
        
        # 3. 连续性特征（10维）
        for tail in range(10):
            consecutive = 0
            for period in recent_data:
                if tail in period.get('tails', []):
                    consecutive += 1
                else:
                    break
            features.append(consecutive / len(recent_data))
        
        # 4. 间隔特征（10维）
        for tail in range(10):
            last_seen = -1
            for i, period in enumerate(recent_data):
                if tail in period.get('tails', []):
                    last_seen = i
                    break
            features.append((last_seen + 1) / len(recent_data) if last_seen >= 0 else 1.0)
        
        # 5. 趋势特征（10维）
        mid = len(recent_data) // 2
        for tail in range(10):
            early_count = sum(1 for period in recent_data[mid:] if tail in period.get('tails', []))
            late_count = sum(1 for period in recent_data[:mid] if tail in period.get('tails', []))
            trend = (late_count - early_count) / max(mid, 1) if mid > 0 else 0
            features.append(trend)
        
        # 6. 统计特征（10维）
        tail_counts_per_period = [len(period.get('tails', [])) for period in recent_data]
        if tail_counts_per_period:
            features.extend([
                np.mean(tail_counts_per_period),
                np.std(tail_counts_per_period),
                np.min(tail_counts_per_period),
                np.max(tail_counts_per_period),
                np.median(tail_counts_per_period)
            ])
        else:
            features.extend([0] * 5)
        
        # 补充特征到固定维度（60维）
        for tail in range(5):  # 额外5维特征
            features.append(0.0)
        
        # 确保特征维度为60
        while len(features) < 60:
            features.append(0.0)
        
        return np.array(features[:60])
    
    def predict_single(self, features):
        """使用深度学习模型进行单次预测"""
        if not self.models:
            return {}
        
        predictions = {}
        
        try:
            with torch.no_grad():
                for model_name, model in self.models.items():
                    model.eval()
                    
                    # 标准化特征
                    scaler = self.scalers[model_name]
                    if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                        features_scaled = scaler.transform(features.reshape(1, -1))
                    else:
                        features_scaled = features.reshape(1, -1)
                    
                    # 转换为张量
                    input_tensor = torch.FloatTensor(features_scaled).to(self.device)
                    
                    # 预测
                    output = model(input_tensor)
                    probabilities = output.cpu().numpy().flatten()
                    
                    predictions[model_name] = probabilities
            
            return predictions
            
        except Exception as e:
            print(f"深度学习预测失败: {e}")
            return {}
    
    def get_training_stats(self):
        """获取训练统计信息"""
        stats = {}
        for model_name, history in self.training_history.items():
            stats[model_name] = {
                'epochs_trained': history['epochs_trained'],
                'best_loss': history['best_loss'],
                'current_accuracy': history['accuracy'][-1] if history['accuracy'] else 0.0,
                'loss_history': history['loss'][-10:],  # 最近10轮的损失
                'accuracy_history': history['accuracy'][-10:]  # 最近10轮的准确率
            }
        return stats
    
    def learn_online_single(self, features, actual_tails):
        """在线学习单个样本"""
        if not self.models:
            return {}
        
        learning_results = {}
        
        try:
            # 创建多标签目标（10个尾数的二分类）
            target = np.zeros(10)
            for tail in actual_tails:
                if 0 <= tail <= 9:
                    target[tail] = 1
            
            for model_name, model in self.models.items():
                    try:
                        model.train()  # 设置为训练模式
                        
                        # 标准化特征
                        scaler = self.scalers[model_name]
                        if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                            # 在线更新标准化器
                            features_scaled = scaler.transform(features.reshape(1, -1))
                            # 简单的在线更新（可选）
                            scaler.partial_fit(features.reshape(1, -1))
                        else:
                            # 初次学习，先拟合标准化器
                            scaler.fit(features.reshape(1, -1))
                            features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # 转换为张量
                        input_tensor = torch.FloatTensor(features_scaled).to(self.device)
                        input_tensor.requires_grad_(True)
                        target_tensor = torch.FloatTensor(target).unsqueeze(0).to(self.device)
                        
                        # 前向传播
                        model.train()
                        optimizer = self.optimizers[model_name]
                        optimizer.zero_grad()
                        
                        output = model(input_tensor)
                        
                        # 计算损失
                        criterion = nn.BCELoss()
                        loss = criterion(output, target_tensor)
                        
                        # 反向传播和参数更新
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # 计算准确率
                        predicted = (output > 0.5).float()
                        accuracy = (predicted == target_tensor).float().mean().item()
                        
                        # 更新训练历史
                        self.training_history[model_name]['loss'].append(loss.item())
                        self.training_history[model_name]['accuracy'].append(accuracy)
                        
                        # 保持历史记录长度
                        if len(self.training_history[model_name]['loss']) > 1000:
                            self.training_history[model_name]['loss'].pop(0)
                        if len(self.training_history[model_name]['accuracy']) > 1000:
                            self.training_history[model_name]['accuracy'].pop(0)
                        
                        learning_results[model_name] = {
                            'loss': loss.item(),
                            'accuracy': accuracy,
                            'status': 'success'
                        }
                        
                    except Exception as e:
                        print(f"深度学习模型 {model_name} 在线学习失败: {e}")
                        learning_results[model_name] = {
                            'status': 'failed',
                            'error': str(e)
                        }
            
            return learning_results
            
        except Exception as e:
            print(f"深度学习在线学习失败: {e}")
            return {}
    
    def update_learning_rate(self, model_name, factor=0.95):
        """动态调整学习率"""
        if model_name in self.optimizers:
            scheduler = self.schedulers[model_name]
            current_lr = self.optimizers[model_name].param_groups[0]['lr']
            
            # 基于最近的损失调整学习率
            if model_name in self.training_history:
                recent_losses = self.training_history[model_name]['loss'][-10:]
                if len(recent_losses) >= 5:
                    # 如果最近损失没有下降，降低学习率
                    if recent_losses[-1] >= recent_losses[-5]:
                        for param_group in self.optimizers[model_name].param_groups:
                            param_group['lr'] *= factor
                        print(f"调整 {model_name} 学习率: {current_lr:.6f} -> {param_group['lr']:.6f}")