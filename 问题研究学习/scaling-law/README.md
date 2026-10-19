# Scaling Law（大模型缩放定律）

> 学习日期：2026-10-19｜五课完整单元｜闪卡 29 张

## 课程地图

| 课 | 主题 | 核心结论 |
|---|---|---|
| 一 | 幂律现象 | $L \propto C^{-\alpha}$（$\alpha\approx0.076$），log-log 坐标上是直线，可用小实验外推预言大模型 |
| 二 | 双瓶颈与计算最优分配 | $L(N,D)=(N_c/N)^{\alpha_N}+(D_c/D)^{\alpha_D}+L_\infty$；$C\approx6ND$ 约束下呈 U 形；最优约 20 token/参数（Chinchilla） |
| 三 | 推理时 Scaling | $\text{Acc}\approx a+b\log T_{\text{infer}}$；RL 训出长思维链；小模型+长思考可追平大模型 |
| 四 | 数据墙 | 需求（每年约×3）追上存量（≈300 万亿 token）约 2026–2028；四条逃生路线：合成数据、多模态/私有、换轴、提高利用率 |
| 五 | 幂律为什么存在 | 幂律从数据自身结构遗传：重尾特征（量化模型）与流形维度（$\alpha\approx4/d$） |

## 文件说明

- [notes.md](./notes.md) —— 完整笔记（九节，TeX 公式）
- [flashcards.html](./flashcards.html) —— 29 张闪卡，下载后浏览器打开即测
- [cheatsheet.pdf](./cheatsheet.pdf) —— 打印版速查卡（23 条要点）
- [scaling_law-schedule.md](./scaling_law-schedule.md) / [.ics](./scaling_law-schedule.ics) —— 间隔复习日程
- 示意图：[幂律曲线](./scaling_law_powerlaw.png)、[Chinchilla U 形](./chinchilla_optimal.png)、[推理时 scaling](./inference_scaling.png)、[数据墙](./data_wall.png)、[幂律起源](./powerlaw_origin.png)

## 关键参考文献

- Kaplan et al. 2020, *Scaling Laws for Neural Language Models*
- Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models*（Chinchilla）
- Muennighoff et al. 2023, *Scaling Data-Constrained Language Models*
- Michaud et al. 2023, *The Quantization Model of Neural Scaling*
- Sharma & Kaplan 2020, *A Neural Scaling Law from the Dimension of the Data Manifold*
- Shumailov et al. 2024, *AI models collapse when trained on recursively generated data*（Nature）
- Villalobos et al., *Will we run out of data?*（数据存量估算）
