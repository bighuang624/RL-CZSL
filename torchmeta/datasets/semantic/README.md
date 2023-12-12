为某一数据集加入新 semantic 时需要做的改动：

1. 在 XXClassDataset 的 self.semantic_type_limitation 中加入对应 semantic_type
2. 在 XXClassDataset 中加入对应的处理代码
3. 可能需要修改 global_utils 中 get_inputs_and_outputs 函数