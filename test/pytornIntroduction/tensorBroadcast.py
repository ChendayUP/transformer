# 张量传播
import numpy as np

# 创建张量 A 和 B
A = np.array([3, 4, 5])
B = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])

# C = A * B
# print('C shape:', C.shape)
print("A shape:", A.shape)
print("B shape:", B.shape)

# 执行 broadcasting 加法
C = A[:, np.newaxis, np.newaxis] + B[np.newaxis, :, :]

print("\nResult shape:", C.shape)
print("\nResult:")
print(C)

# 验证每个切片
print("\nFirst slice (C[0]):")
print(C[0])
print("\nSecond slice (C[1]):")
print(C[1])