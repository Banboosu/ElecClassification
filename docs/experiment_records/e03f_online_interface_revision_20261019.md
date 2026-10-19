# E03F-v2：在线冻结控制的训练器接口修订

2026-10-19。原科学协议仍见 `e03f_strong_head_lora_protocol_20261019.md`；本记录不变更任何阈值、LR、batch、epoch、数据、父checkpoint、rank或选择规则。

## v1事故与证据

v1于UTC02:50:45冻结218文件，协议SHA `ec050c33e559b81c6e4b1939aad4dfbcec40ec7dbace74bccf59b3bdf3e47774`。全部128测试/19子测试及两支路strong-head batch64合成梯度检查通过，随后进入在线frozen_fusion控制。

新卡epoch0迁移核验通过：父validation的预测0差异、Macro-F1漂移0。第一次训练窗口在原 `train_e03.train` 的adapter梯度见证处报：

```
AttributeError: 'int' object has no attribute 'sqrt'
```

旧训练器假设 `model.backbone is not None` 就存在adapter，rank0在线冻结encoder不符合这个旧假设；空adapter集合求和是int0，不能sqrt。报错发生在第一次optimizer.step之前，history为空，last checkpoint仍epoch0；没有已提交训练epoch，不是LoRA效果失败或数值溢出。

v1协议、源码、失败状态、日志、epoch0 checkpoint及smoke保留，不回写成功。

## 修订

新增独立 `strong_head_lora_online.py`、`e03f_strong_head_lora_online_protocol.json`、`run_e03f_online_remote.sh`，套件后缀v2；四条件全部在新目录从原父checkpoint重启，不接续v1失败内存状态。

修复仅为PeftClassifier接口兼容：

- rank0的真实encoder注册为 `online_frozen_encoder`，训练器用于adapter角色判断的 `.backbone=None`；`forward`仍在每批执行同一冻结encoder，**不是缓存或旁路常量特征**。
- rank0 encoder强制全部requires_grad=False、eval；只训练原分类头。冻结参数哈希仍包含该注册encoder。
- 原训练器在 `.backbone=None` 时选择frozen_head_batch_size，因此显式设为已冻结的物理batch64，有效batch仍128。不是让控制悄悄使用128物理batch。
- rank8路径不变，仍保留真实q/v梯度见证。没有伪造梯度、加入dummy adapter或跳过有效梯度检查。
- 仅在新模块进程内以scoped hooks替换模型构造和训练协议适配；原 `train_e03.py`、v1源码及历史输入不编辑。

补充回归覆盖两种在线包装输出逐位相同、冻结/训练参数角色、base eval/head train、rank0一个完整GPU训练epoch（微批64/64/2）能提交且无adapter见证、冻结参数不变、固定scale/无跳步。原epoch0/重启测试也继续执行。

v2合成检查与冻结同时绑定原核心和新兼容层源码SHA，新增冻结文件包括本修订、v2脚本/测试、v1已冻结协议和失败现场。v2四条件仍为seed43、40%、每条件追加30epoch、epoch0可选；科学门槛不变，不评价test或自动扩展。
