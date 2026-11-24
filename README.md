# pairwise-pulldown-dedup

## 领域

游戏、图形与多媒体

## 主题

3:2 Pulldown 去重复帧

## 说明

周期自适应、容差判定与拼接延续。

初始状态下 `media/pulldown_dedup.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
