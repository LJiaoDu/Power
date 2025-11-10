"""
优化建议：使用KV Cache加速自回归生成

当前问题：
每次迭代都重新计算所有之前token的Key和Value，造成大量重复计算。

解决方案：
缓存之前步骤计算的K、V，每次只计算新token的K、V。

预期加速：
理论上可以加速 48/2 = 24倍（对于48步自回归）

实现复杂度：
需要修改PyTorch的TransformerDecoder，使用自定义Attention层。

是否值得：
- 如果只是训练：不值得（因为训练需要teacher forcing，不用自回归）
- 如果是推理：非常值得实现

注意：
当前代码在训练时也使用自回归生成，这是性能瓶颈！
建议在训练时使用teacher forcing（并行计算），推理时才用自回归。
"""

# 推荐的优化方案（需要重构forward_loop）：

def forward_loop_optimized(self, x, t_true):
    """
    训练模式：使用teacher forcing（并行，快速）
    推理模式：使用自回归生成（串行，慢但准确）
    """
    if self.training:
        # 训练时：使用真实的目标序列（teacher forcing）
        # 可以并行计算，不需要循环
        t_embed = self.embedding_fc_t(t_true)
        # ... RoPE处理 ...
        t_out = self.decoder(t_embed, x)
        return self.fc(t_out)
    else:
        # 推理时：使用自回归生成（当前的实现）
        # 这里可以进一步优化使用KV cache
        return self.forward_loop_autoregressive(x)
