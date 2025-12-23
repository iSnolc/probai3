"""
DDPM 扩散模型单元测试

测试内容：
- 噪声调度系数计算
- 前向扩散过程 q(z_t | z_0)
- 后验分布 p(z_{t-1} | z_t)
- 损失函数计算
"""

import pytest
import torch
import numpy as np

from src.models.ddpm import DDPM


# ============ Fixtures ============

@pytest.fixture
def mock_model():
    """创建一个简单的 mock 模型用于测试"""
    class MockScoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)
        
        def forward(self, z_t, t, edge_index=None, batch=None, context=None):
            return self.linear(z_t)
    
    return MockScoreModel()


@pytest.fixture
def ddpm(mock_model):
    """创建 DDPM 实例"""
    return DDPM(model=mock_model, noise_schedule_type="linear", N=100)


@pytest.fixture
def ddpm_cosine(mock_model):
    """创建使用 cosine 调度的 DDPM 实例"""
    return DDPM(model=mock_model, noise_schedule_type="cosine", N=100)


# ============ 噪声调度系数测试 ============

class TestNoiseSchedule:
    """测试噪声调度系数"""
    
    @pytest.mark.parametrize("N", [10, 100, 1000])
    def test_get_coefs_linear_shape(self, N):
        """测试线性调度系数的形状"""
        betas, alphas, alpha_bars = DDPM.get_coefs(N, "linear")
        
        assert betas.shape == (N,), f"betas shape should be ({N},)"
        assert alphas.shape == (N,), f"alphas shape should be ({N},)"
        assert alpha_bars.shape == (N,), f"alpha_bars shape should be ({N},)"
    
    @pytest.mark.parametrize("N", [10, 100, 1000])
    def test_get_coefs_cosine_shape(self, N):
        """测试 cosine 调度系数的形状"""
        betas, alphas, alpha_bars = DDPM.get_coefs(N, "cosine")
        
        assert betas.shape == (N,)
        assert alphas.shape == (N,)
        assert alpha_bars.shape == (N,)
    
    def test_coefs_values_range(self):
        """测试系数值的范围"""
        betas, alphas, alpha_bars = DDPM.get_coefs(100, "linear")
        
        # betas 应该在 (0, 1) 之间
        assert torch.all(betas > 0), "betas should be positive"
        assert torch.all(betas < 1), "betas should be less than 1"
        
        # alphas = 1 - betas，所以也在 (0, 1)
        assert torch.all(alphas > 0), "alphas should be positive"
        assert torch.all(alphas < 1), "alphas should be less than 1"
        
        # alpha_bars 应该单调递减
        assert torch.all(alpha_bars[1:] < alpha_bars[:-1]), \
            "alpha_bars should be monotonically decreasing"
    
    def test_coefs_relationship(self):
        """测试系数之间的关系"""
        betas, alphas, alpha_bars = DDPM.get_coefs(100, "linear")
        
        # alphas = 1 - betas
        assert torch.allclose(alphas, 1 - betas), \
            "alphas should equal 1 - betas"
        
        # alpha_bars = cumprod(alphas)
        assert torch.allclose(alpha_bars, torch.cumprod(alphas, dim=0)), \
            "alpha_bars should equal cumulative product of alphas"


# ============ 前向扩散测试 ============

class TestForwardProcess:
    """测试前向扩散过程 q(z_t | z_0)"""
    
    def test_q_mean_shape(self, ddpm):
        """测试 _q_mean 输出形状"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        mean = ddpm._q_mean(z, t)
        assert mean.shape == z.shape, "q_mean output shape should match input"
    
    def test_q_std_shape(self, ddpm):
        """测试 _q_std 输出形状（支持广播）"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        std = ddpm._q_std(z, t)
        # std 可以是 (batch, 1) 或 (batch, features)，都可以广播
        assert std.shape[0] == z.shape[0], "batch dimension should match"
        # 验证广播兼容性
        result = std * torch.randn_like(z)
        assert result.shape == z.shape, "std should be broadcastable to z shape"
    
    def test_q_sample_shape(self, ddpm):
        """测试 q_sample 输出形状"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        z_t, epsilon = ddpm.q_sample(z, t)
        
        assert z_t.shape == z.shape, "z_t shape should match input"
        assert epsilon.shape == z.shape, "epsilon shape should match input"
    
    def test_q_sample_at_t0(self, ddpm):
        """测试 t=0 时的采样（应该接近原始数据）"""
        z = torch.randn(4, 8)
        t = torch.zeros(4, dtype=torch.long)
        
        mean = ddpm._q_mean(z, t)
        # 当 t=0 时，alpha_bar 接近 1，所以 mean ≈ z
        assert torch.allclose(mean, z, atol=0.01), \
            "At t=0, q_mean should be close to z"


# ============ 损失函数测试 ============

class TestLoss:
    """测试损失函数"""
    
    def test_losses_shape(self, ddpm):
        """测试 _losses 输出形状"""
        epsilon_pred = torch.randn(4, 8)
        epsilon = torch.randn(4, 8)
        
        losses = ddpm._losses(epsilon_pred, epsilon)
        
        # 应该对特征维度求和，保留 batch 维度
        assert losses.shape == (4,), \
            "_losses should return (batch_size,) shape"
    
    def test_losses_value(self, ddpm):
        """测试损失值计算"""
        epsilon = torch.randn(4, 8)
        epsilon_pred = epsilon.clone()  # 完美预测
        
        losses = ddpm._losses(epsilon_pred, epsilon)
        
        # 完美预测时损失应该为 0
        assert torch.allclose(losses, torch.zeros_like(losses)), \
            "Perfect prediction should have zero loss"
    
    def test_losses_positive(self, ddpm):
        """测试损失值为非负"""
        epsilon_pred = torch.randn(4, 8)
        epsilon = torch.randn(4, 8)
        
        losses = ddpm._losses(epsilon_pred, epsilon)
        
        assert torch.all(losses >= 0), "Losses should be non-negative"


# ============ 后验分布测试 ============

class TestReverseProcess:
    """测试反向扩散过程 p(z_{t-1} | z_t)"""
    
    def test_p_mean_shape(self, ddpm):
        """测试 _p_mean 输出形状"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        mean = ddpm._p_mean(z, t)
        assert mean.shape == z.shape, "p_mean output shape should match input"
    
    def test_p_std_shape(self, ddpm):
        """测试 _p_std 输出形状（支持广播）"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        std = ddpm._p_std(z, t)
        # std 可以是 (batch, 1) 或 (batch, features)，都可以广播
        assert std.shape[0] == z.shape[0], "batch dimension should match"
        # 验证广播兼容性
        result = std * torch.randn_like(z)
        assert result.shape == z.shape, "std should be broadcastable to z shape"
    
    def test_p_std_positive(self, ddpm):
        """测试 _p_std 返回正值"""
        z = torch.randn(4, 8)
        t = torch.randint(1, 100, (4,))
        
        std = ddpm._p_std(z, t)
        assert torch.all(std > 0), "p_std should be positive"


# ============ 设备兼容性测试 ============

class TestDevice:
    """测试设备兼容性"""
    
    def test_device_property(self, ddpm):
        """测试 device 属性"""
        assert ddpm.device == torch.device("cpu")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, mock_model):
        """测试 CUDA 兼容性"""
        ddpm = DDPM(model=mock_model, N=10).cuda()
        assert ddpm.device == torch.device("cuda", 0)
