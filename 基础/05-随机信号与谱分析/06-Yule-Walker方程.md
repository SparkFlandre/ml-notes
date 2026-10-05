# Yule-Walker

AR 模型 Σ a_k x[n-k] = w[n]，a0=1。

推导：差分方程乘 x[n-l] 取期望。l>0 时白噪声和历史输出独立，E[w[n]x[n-l]]=0，随机项消掉。剩 R a = -r。

R 是 r_xx[|i-j|] 的 Toeplitz 矩阵（对称正定）。Toeplitz 结构 = 平稳性指纹。能用 Levinson-Durbin 从 O(N³) 降到 O(N²)。

阶数方程不决定，用 AIC/BIC。

例子：N=2 一个共振峰，N=4 两个峰残差 0.73→0.25，N=6 系数小残差微降 → 过拟合。阶数不是越高越好。

（消白噪声那步挺妙，乘 x[n-l] 取期望就没了）
