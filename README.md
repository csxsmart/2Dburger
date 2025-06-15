# 2Dburger
# 有限差分算子使用指南

基于PyTorch的有限差分算子库，用于求解偏微分方程。
## 安装依赖
```bash
pip install torch 
```
## 基本使用

### 1D算子

#### 扩散算子 (二阶导数)
```python
from finite_diff_operators import diffusion1D

# 创建扩散算子
diffusion_op = diffusion1D(scheme='Central', accuracy=4, device='cuda')

# 输入张量形状: [batch_size, channels, length]
u = torch.randn(10, 1, 100, device='cuda')  # 10个样本，长度100
result = diffusion_op(u)  # 计算 ∂²u/∂x²
```

#### 一阶导数算子 (上风格式)
```python
from finite_diff_operators import dudx_1D

# 创建上风格式一阶导数算子
dudx_op = dudx_1D(scheme='Upwind', accuracy=2, device='cuda')

u = torch.randn(10, 1, 100, device='cuda')
result = dudx_op(u)  # 根据u的符号自动选择前向或后向差分
```

### 2D算子

#### 二阶导数算子
```python
from finite_diff_operators import d2udx2_2D, d2udy2_2D

# x方向二阶导数
d2udx2_op = d2udx2_2D(scheme='Central2', accuracy=6, device='cuda')

# y方向二阶导数  
d2udy2_op = d2udy2_2D(scheme='Central2', accuracy=6, device='cuda')

# 输入张量形状: [batch_size, channels, height, width]
u = torch.randn(5, 1, 64, 64, device='cuda')
d2udx2 = d2udx2_op(u)  # ∂²u/∂x²
d2udy2 = d2udy2_op(u)  # ∂²u/∂y²
```

#### 一阶导数算子
```python
from finite_diff_operators import dudx_2D, dudy_2D

# 创建2D一阶导数算子
dudx_op = dudx_2D(scheme='Upwind1', accuracy=3, device='cuda')
dudy_op = dudy_2D(scheme='Upwind1', accuracy=3, device='cuda')

u = torch.randn(5, 1, 64, 64, device='cuda')
dudx = dudx_op(u)  # ∂u/∂x
dudy = dudy_op(u)  # ∂u/∂y
```

## 边界条件处理

代码假设使用周期边界条件，需要手动填充边界：

```python
def padbcx(u, pad_width=3):
    """x方向周期边界条件"""
    return torch.cat((u[:, :, -pad_width:], u, u[:, :, :pad_width]), dim=2)

def padbcy(u, pad_width=3):
    """y方向周期边界条件"""
    return torch.cat((u[:, :, :, -pad_width:], u, u[:, :, :, :pad_width]), dim=3)

# 使用示例
u_padded_x = padbcx(u)  # 对原始数据进行x方向填充
result = dudx_op(u_padded_x)  # 在填充后的数据上计算导数
```
## 参数选择指南

### 数值格式选择
- **Central**: 中心差分，精度高但可能不稳定
- **Upwind**: 上风格式，适合对流占主导的问题
- **Forward/Backward**: 前向/后向差分，适合边界处理

### 精度选择
- **accuracy=2**: 标准二阶精度，计算快
- **accuracy=4**: 四阶精度，平衡精度和效率
- **accuracy=6,8**: 高精度，适合高质量求解

### 张量维度要求
- **1D**: `[batch_size, channels, length]`
- **2D**: `[batch_size, channels, height, width]`

## 注意事项

1. **GPU内存**: 高精度算子需要更多内存
2. **边界处理**: 必须正确处理边界条件
3. **数值稳定性**: 时间步长需要满足CFL条件
4. **批处理**: 支持同时处理多个样本，提高效率
