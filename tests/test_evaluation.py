"""
评估模块单元测试

测试内容：
- 稳定性分析
- 化学键分析
- 评估器功能
"""

import pytest
import torch
import numpy as np

from src.evaluation.bond_analyze import get_bond_order, allowed_bonds
from src.evaluation.stability_analyze import check_stability


# ============ 化学键分析测试 ============

class TestBondAnalyze:
    """测试化学键分析"""
    
    def test_single_bond(self):
        """测试单键检测"""
        # C-C 单键长度约 1.54 Å
        order = get_bond_order("C", "C", 1.50)
        assert order == 1, "Should detect single bond"
    
    def test_double_bond(self):
        """测试双键检测"""
        # C=C 双键长度约 1.34 Å
        order = get_bond_order("C", "C", 1.30)
        assert order == 2, "Should detect double bond"
    
    def test_triple_bond(self):
        """测试三键检测"""
        # C≡C 三键长度约 1.20 Å
        order = get_bond_order("C", "C", 1.15)
        assert order == 3, "Should detect triple bond"
    
    def test_no_bond(self):
        """测试无键检测"""
        # 距离太远，无键
        order = get_bond_order("C", "C", 3.0)
        assert order == 0, "Should detect no bond"
    
    def test_h_bond(self):
        """测试 H 键"""
        # C-H 键长约 1.09 Å
        order = get_bond_order("C", "H", 1.10)
        assert order == 1, "Should detect C-H single bond"
    
    def test_allowed_bonds(self):
        """测试允许的键数"""
        assert allowed_bonds["H"] == 1
        assert allowed_bonds["C"] == 4
        assert allowed_bonds["N"] == 3
        assert allowed_bonds["O"] == 2
        assert allowed_bonds["F"] == 1


# ============ 稳定性分析测试 ============

class TestStabilityAnalyze:
    """测试分子稳定性分析"""
    
    @pytest.fixture
    def methane_molecule(self):
        """创建甲烷分子 (CH4)"""
        # C 原子在中心，4 个 H 原子在四面体顶点
        positions = np.array([
            [0.0, 0.0, 0.0],      # C
            [1.09, 0.0, 0.0],     # H
            [-0.36, 1.03, 0.0],   # H
            [-0.36, -0.51, 0.89], # H
            [-0.36, -0.51, -0.89] # H
        ])
        atom_types = np.array([1, 0, 0, 0, 0])  # 1=C, 0=H
        
        return positions, atom_types
    
    @pytest.fixture
    def dataset_info(self):
        """数据集信息"""
        return {
            "atom_decoder": ["H", "C", "N", "O", "F"]
        }
    
    def test_stable_molecule(self, methane_molecule, dataset_info):
        """测试稳定分子检测"""
        positions, atom_types = methane_molecule
        
        is_stable, num_stable, total = check_stability(
            positions, atom_types, dataset_info
        )
        
        # 甲烷是稳定分子
        assert is_stable, "Methane should be stable"
        assert num_stable == total, "All atoms should be stable"
    
    def test_output_format(self, methane_molecule, dataset_info):
        """测试输出格式"""
        positions, atom_types = methane_molecule
        
        result = check_stability(positions, atom_types, dataset_info)
        
        assert len(result) == 3, "Should return 3 values"
        assert isinstance(result[0], bool), "First value should be bool"
        assert isinstance(result[1], int), "Second value should be int"
        assert isinstance(result[2], int), "Third value should be int"


# ============ 评估器测试 ============

class TestEvaluator:
    """测试评估器"""
    
    # 注意：完整的 Evaluator 测试需要真实的模型和数据
    # 这里只测试基本功能
    pass
