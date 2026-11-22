# pairwise-dialogue-branch

## 领域

游戏、图形与多媒体

## 主题

对话分支

## 说明

返回恢复、服务器判定与存档中断恢复。

初始状态下 `gameplay/dialogue_branch.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
