# 触觉传感器硅胶层：从有限元仿真到代理模型

**状态：进行中，2026年11月开始，目前还没有结果。**
English: [README.md](README.md)

视触觉传感器靠一层软硅胶的变形来感知接触。用有限元计算这层硅胶的变形很准，但很慢；机器人策略在仿真里训练时每秒需要成千上万次接触计算，有限元跟不上。这个项目要做出一个快速的替代模型，并且验证它：

1. **标定**：用同一批硅胶浇注试样，做压缩试验，得到材料参数。
2. **仿真**：在 ANSYS 里建立超弹性接触模型，用 Hertz 解析解和网格收敛性分析校核。
3. **学习**：训练代理模型，同时预测表面位移场和接触力。
4. **验证**：用自制传感器的压痕实验，对比有限元、代理模型和实测。
5. **写清楚模型在哪里失效**，并给出数据。

## 和最接近的已有工作的区别

2026年10月的论文 [FEM-Validated Real-Time Modelling of Soft Tactile Deformation](https://arxiv.org/pdf/2610.16667) 走的是同一条路线：先用实验验证 ANSYS 模型，再用它训练代理模型。那篇论文研究的是TPU材料的二维平面结构，材料参数借用自相近材料，并把三维传感器和力的估计列为后续工作。本项目做三维的硅胶胶层，材料参数来自对传感器所用硅胶的标定，同时预测力和位移，并公开代码、数据和失效工况。

## 路线图

| 版本 | 目标日期 | 内容 | 完成标准 |
| --- | --- | --- | --- |
| v0 最小闭环 | 2027年1月10日 | 二维球压痕仿真、硅胶块测力、预测力的小模型 | 三条力与压深曲线画在一张图上 |
| v1 仿真加代理模型 | 2027年2月28日 | 标定并收敛的三维模型、数据集、预测位移场和力的代理模型 | 一句话说清快了多少倍、误差多少、适用范围 |
| v2 完整发布 | 2027年5月23日 | 自制传感器的实测对比、失效工况表、文章、视频 | 新读者只看 README，3分钟后能复述结论 |

## 目录

```
fea/          Hertz 与薄层解析解、开源有限元求解器、PyMAPDL 接触模型、批量生成数据集
materials/    由压缩试验拟合超弹性本构
surrogate/    代理模型（力模型、表面形貌基线）
experiments/  测量记录模板、对比脚本、合成示例
docs/         有限元验证报告、实验方案、采购清单、开发日志
tests/        解析部分的单元测试
```

## 不装 ANSYS 也能试

下面的命令全部跑在合成数据上，只能说明流程是通的，不代表任何结果。文件名带 `example_synthetic_` 的都不是实测数据。

```bash
pip install -r requirements.txt
python -m pytest tests
python experiments/make_synthetic_examples.py
cd fea && python run_batch.py --backend hertz --n 300 --out ../data/dataset_hertz.npz && cd ..
cd surrogate && python train_force_v0.py ../data/dataset_hertz.npz && cd ..
cd materials && python fit_hyperelastic.py ../experiments/example_synthetic_compression.csv --height 20 --diameter 15 && cd ..
```

## 哪些验证过，哪些没有

| 部分 | 状态 |
| --- | --- |
| Hertz 解、本构拟合、批处理、v0 力模型 | 已在合成数据上运行并通过单元测试 |
| 开源有限元求解器 `fea/axisym_fem.py` | 半无限体极限下与 Hertz 解相差1%以内，薄层工况与解析解相差2%到4%，见 [docs/fem_verification.md](docs/fem_verification.md) |
| `fea/axisym_indentation_pymapdl.py` | 参考草稿，从未在 ANSYS 上运行 |
| 材料参数、ANSYS 仿真结果、实测数据 | 暂无 |

## 许可证

v1 发布前确定。在此之前保留所有权利。
