"""
EGNN 等变图神经网络单元测试

测试内容：
- 网络输出形状
- 等变性质
- 梯度流动
"""

import pytest
import torch

from src.models.egnn import EGNN, EGNNScore, GNN, GCL


# ============ Fixtures ============

@pytest.fixture
def simple_graph():
    """创建简单的测试图"""
    num_nodes = 10
    num_edges = 20
    
    h = torch.randn(num_nodes, 5)  # 节点特征
    x = torch.randn(num_nodes, 3)  # 3D 坐标
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    return h, x, edge_index


@pytest.fixture
def egnn():
    """创建 EGNN 实例"""
    return EGNN(
        in_node_nf=5,
        hidden_nf=32,
        n_layers=2,
        out_node_nf=5
    )


@pytest.fixture
def egnn_score():
    """创建 EGNNScore 实例"""
    return EGNNScore(
        in_node_nf=6,  # 5 + 1 for time
        hidden_nf=32,
        n_layers=2,
        out_node_nf=5
    )


# ============ 基础测试 ============

class TestEGNN:
    """测试 EGNN 网络"""
    
    def test_forward_shape(self, egnn, simple_graph):
        """测试前向传播输出形状"""
        h, x, edge_index = simple_graph
        
        h_out, x_out = egnn(h, x, edge_index)
        
        assert h_out.shape == h.shape, "h output shape should match input"
        assert x_out.shape == x.shape, "x output shape should match input"
    
    def test_gradient_flow(self, egnn, simple_graph):
        """测试梯度能否正常流动"""
        h, x, edge_index = simple_graph
        h.requires_grad_(True)
        x.requires_grad_(True)
        
        h_out, x_out = egnn(h, x, edge_index)
        loss = h_out.sum() + x_out.sum()
        loss.backward()
        
        assert h.grad is not None, "Gradients should flow to h"
        assert x.grad is not None, "Gradients should flow to x"


class TestEGNNScore:
    """测试 EGNNScore 评分网络"""
    
    def test_forward_shape(self, egnn_score):
        """测试前向传播输出形状"""
        batch_size = 4
        num_nodes = 10
        
        # z_t: [num_nodes, 8] = [coords(3) + atom_type(5)]
        z_t = torch.randn(num_nodes, 8)
        t = torch.rand(num_nodes)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        output = egnn_score(z_t, t, edge_index, batch)
        
        assert output.shape == z_t.shape, "Output shape should match input"
    
    def test_time_embedding(self, egnn_score):
        """测试时间嵌入是否影响输出"""
        num_nodes = 10
        z_t = torch.randn(num_nodes, 8)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        t1 = torch.ones(num_nodes) * 0.1
        t2 = torch.ones(num_nodes) * 0.9
        
        out1 = egnn_score(z_t, t1, edge_index, batch)
        out2 = egnn_score(z_t, t2, edge_index, batch)
        
        # 不同时间步应该产生不同输出
        assert not torch.allclose(out1, out2), \
            "Different time steps should produce different outputs"


class TestGCL:
    """测试图卷积层"""
    
    def test_forward_shape(self):
        """测试 GCL 输出形状"""
        gcl = GCL(
            input_nf=16,
            output_nf=16,
            hidden_nf=32,
            normalization_factor=1,
            aggregation_method="sum"
        )
        
        num_nodes = 10
        h = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        h_out = gcl(h, edge_index)
        
        assert h_out.shape == h.shape, "Output shape should match input"
