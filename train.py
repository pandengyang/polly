import numpy as np

from dataset import Dataset
from activation_function.sigmoid import Sigmoid as ActivationFunction

if __name__ == '__main__':
    # 加载数据集：64 张黑白图像，每张图像 12 个像素，图像内容为 0 或 1
    dataset = Dataset('./dataset')
    # X：64x12 矩阵，每行是一个样本的 12 个像素值，黑色为 1，白色为 0
    # Y：64x2 矩阵，每行是一个样本的为 0 和 1 的概率
    X, Y = dataset.toMatrix()
    print(X)
    print(Y)

    np.random.seed(42)

    # 隐藏层的权重和偏置
    # W1：12x3矩阵，隐藏层权重
    W1 = np.random.randn(12, 3) * 0.1
    # b1：1x3 矩阵，隐藏层偏置
    b1 = np.random.randn(1, 3) * 0.1
    print(W1)
    print(b1)

    # 输出层的权重和偏置
    # W2：3x2 矩阵，输出层权重
    W2 = np.random.randn(3, 2) * 0.1
    # b2：1x2 矩阵，输出层偏置
    b2 = np.random.randn(1, 2) * 0.1
    print(W2)
    print(b2)

    # 学习率
    LEARNING_RATE = 0.5

    # 激活函数
    actfunc = ActivationFunction()
    print(actfunc)

    # 训练
    EPOCHS = 10000
    for epoch in range(EPOCHS):
        # 正向传播

        # Z1：64x3 矩阵，隐藏层输入：加权求和、含偏置
        # 每行为一个样本的隐藏层输入，每列为一个隐藏层神经元的输入，共 64 个样本
        Z1 = X @ W1 + b1
        # A1：64x3 矩阵，隐藏层激活后的输出
        # 每行为一个样本的隐藏层输出，每列为一个隐藏层神经元的输出，共 64 个样本
        A1 = actfunc.function(Z1)

        # Z2：64x2 矩阵，输出层输入：加权求和、含偏置
        # 每行为一个样本的输出层输入，每列为一个输出层神经元的输入，共 64 个样本
        Z2 = A1 @ W2 + b2
        # A2：64x2 矩阵，输出层激活后输出
        # 每行为一个样本的输出层输出，每列为一个输出层神经元的输出，共 64 个样本
        A2 = actfunc.function(Z2)

        # 计算损失：损失函数为均方误差
        L = np.mean((Y - A2)**2)

        # 打印损失
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, L: {L}')

        # 反向传播

        # dL_dA2：64x2 矩阵
        # 损失函数对输出层输出的导数
        dL_dA2 = 2 * (A2 - Y) / len(X)
        # delta2：64x2 矩阵
        # 损失函数对输出层输入的导数
        delta2 = dL_dA2 * actfunc.derivative(A2)

        # dL_dW2：3x2矩阵
        # 损失函数对输出层权重的导数
        dL_dW2 = A1.T @ delta2
        # dL_db2：1x2 矩阵
        # 损失函数对输出层偏置的导数
        dL_db2 = np.sum(delta2, axis=0, keepdims=True)

        # dL_dA1：64x3 矩阵
        # 损失函数对隐藏层输出的导数
        dL_dA1 = delta2 @ W2.T
        # delta1：64x3 矩阵
        # 损失函数对隐藏层输入的导数
        delta1 = dL_dA1 * actfunc.derivative(A1)

        # dL_dW1：12x3 矩阵
        # 损失函数对隐藏层权重的导数
        dL_dW1 = X.T @ delta1
        # dL_db1：1x3 矩阵
        # 损失函数对隐藏层偏置的导数
        dL_db1 = np.sum(delta1, axis=0, keepdims=True)

        # 更新权重和偏置
        W2 -= LEARNING_RATE * dL_dW2
        b2 -= LEARNING_RATE * dL_db2
        W1 -= LEARNING_RATE * dL_dW1
        b1 -= LEARNING_RATE * dL_db1

    # 打印模型参数
    print("训练完成！最终模型参数：")
    print("隐藏层权重：")
    print(W1)
    print("隐藏层偏置：")
    print(b1)
    print("输出层权重：")
    print(W2)
    print("输出层偏置：")
    print(b2)

    # 保存模型参数
    np.savez('polly.npz', W1=W1, b1=b1, W2=W2, b2=b2)
