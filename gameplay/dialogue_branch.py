"""对话分支核心。

把客户端、服务端与存档层各自维护的对话状态收敛为一套纯函数：输入不被
修改，返回值可直接序列化。对应题目中的十个问题：

==========================  ======================================
函数                        解决的问题
==========================  ======================================
``restore_options``         返回上一层时恢复选项的解锁/已选状态
``server_authority``        联机时以服务端判定为准
``seen_marker``             给选过的选项加持久化标记
``mid_save``/``load_save``  存档记录游标，读档从中间继续
``voice_wait``/``auto_play``  自动播放等语音和字幕都结束
``layout_stable``           本地化后槽位与点击区域保持稳定
``speaker_lock``            同一时刻只允许一个发言人
``timeline_unify``          语音与字幕共用一条时间轴
``weighted_random``         权重来自数据，随机种子可控
``skip_compensate``         跳过对话时补执行任务事件与条件
==========================  ======================================
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SAVE_VERSION = 1

#: 服务端判定时允许保留客户端值的字段（纯显示偏好，不影响分支）。
LOCAL_ONLY_KEYS = frozenset(
    {
        "ui",
        "settings",
        "volume",
        "voice_volume",
        "text_speed",
        "auto_play_speed",
        "language",
        "accessibility",
    }
)

#: 进度字段；服务端缺失时不能用客户端存档补缺。
PROGRESS_KEYS = ("node", "node_id", "line", "line_index", "option", "selected")

_CHOICE_KEYS = ("chosen", "selected", "selection", "option")


# ---------------------------------------------------------------------------
# 输入规整
# ---------------------------------------------------------------------------


class _Missing:
    """条件求值中缺失变量的占位符；任何比较都视为不成立。"""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()


def _as_mapping(value: Any) -> dict[str, Any]:
    """把任意输入规整为字典；标量放在 ``value`` 键下。"""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return {"value": copy.deepcopy(value)}


def _is_list(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _normalize_option(option: Any, index: int) -> dict[str, Any]:
    """把映射/字符串/标量统一成带 ``id``、``text``、``order`` 的选项。"""

    if isinstance(option, Mapping):
        data = copy.deepcopy(dict(option))
    elif isinstance(option, str):
        data = {"text": option}
    else:
        data = {"value": copy.deepcopy(option)}

    oid = next((data[key] for key in ("id", "option_id", "key") if data.get(key) is not None), None)
    if oid is None:
        oid = data.get("value", f"opt_{index}")
    data["id"] = oid
    data.setdefault("text", str(oid))
    data.setdefault("order", index)
    data.setdefault("locked", False)
    return data


def _options_of(source: Any) -> list[dict[str, Any]]:
    """从节点映射、选项序列或选项数量得到规整后的选项列表。"""

    if isinstance(source, int) and not isinstance(source, bool):
        return [_normalize_option({"id": index}, index) for index in range(max(0, source))]
    if isinstance(source, Mapping):
        raw = source.get("options")
        if raw is None:
            raw = source.get("choices", [])
    elif _is_list(source):
        raw = list(source)
    else:
        raw = []
    if not _is_list(raw):
        return []
    return [_normalize_option(item, index) for index, item in enumerate(raw)]


def _is_unlocked(option: Mapping[str, Any]) -> bool:
    """``unlocked`` 显式给出时以它为准，否则未锁定即可选。"""

    return bool(option.get("unlocked", not option.get("locked")))


def _flatten_ids(value: Any) -> list[Any]:
    """把 id、id 列表或带 ``chosen`` 的历史项压平成有序、去重的 id 列表。"""

    result: list[Any] = []
    seen: set[str] = set()

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            for key in _CHOICE_KEYS:
                if key in item:
                    add(item[key])
                    return
            if "id" in item:
                add(item["id"])
            return
        if _is_list(item):
            for sub in item:
                add(sub)
            return
        if str(item) not in seen:
            seen.add(str(item))
            result.append(item)

    add(value)
    return result


def _history_state(history: Any) -> tuple[list[Any], list[Any]]:
    """从历史栈取出 (已选, 已解锁) 两组 id。"""

    chosen: list[Any] = []
    unlocked: list[Any] = []

    def visit(entry: Any) -> None:
        if isinstance(entry, Mapping):
            for key in ("history", "frames"):
                if key in entry:
                    visit(entry[key])
            for key in _CHOICE_KEYS:
                if key in entry:
                    chosen.extend(_flatten_ids(entry[key]))
                    break
            unlocked.extend(_flatten_ids(entry.get("unlocked")))
            for option in _options_of(entry):
                if option.get("seen"):
                    chosen.append(option["id"])
                if _is_unlocked(option):
                    unlocked.append(option["id"])
        elif _is_list(entry):
            for sub in entry:
                visit(sub)
        else:
            chosen.extend(_flatten_ids(entry))

    visit(history)
    return _unique(chosen), _unique(unlocked)


def _unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if str(value) not in seen:
            seen.add(str(value))
            result.append(value)
    return result


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


# ---------------------------------------------------------------------------
# 条件求值
# ---------------------------------------------------------------------------


def condition_eval(condition: Any, variables: Any = None) -> bool:
    """确定性的条件求值器，客户端与服务端共用同一套语义。

    支持布尔值、条件列表（全部成立）、``all``/``any``/``not`` 组合、
    ``{"var": 路径, "op": 运算符, "value": 期望值}``、``{路径: 期望值}``
    映射，以及受限表达式（不允许函数调用与赋值）。未知变量一律不成立，
    因此两端不会因本地环境差异看到不同分支。
    """

    env = variables if isinstance(variables, Mapping) else {}
    return _eval(condition, env)


def _eval(condition: Any, env: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    if isinstance(condition, (int, float)):
        return bool(condition)
    if isinstance(condition, str):
        expression = condition.strip()
        if not expression:
            return True
        try:
            return bool(_eval_ast(ast.parse(expression, mode="eval").body, env))
        except Exception:
            return False
    if _is_list(condition):
        return all(_eval(item, env) for item in condition)
    if not isinstance(condition, Mapping):
        return bool(condition)

    for key in ("all", "and"):
        if key in condition:
            return all(_eval(item, env) for item in _as_conditions(condition[key]))
    for key in ("any", "or"):
        if key in condition:
            return any(_eval(item, env) for item in _as_conditions(condition[key]))
    if "not" in condition:
        return not _eval(condition["not"], env)
    for key in ("expr", "expression"):  # 嵌套表达式
        if key in condition:
            return _eval(condition[key], env)

    if any(key in condition for key in ("var", "path", "key")):
        path = condition.get("var", condition.get("path", condition.get("key")))
        actual = _lookup(path, env)
        if "op" in condition or "operator" in condition:
            return _compare(actual, condition.get("op", condition.get("operator")), condition.get("value"))
        if "value" in condition:
            return _compare(actual, "eq", condition["value"])
        return bool(actual)

    # 形如 {"quest.stage": 3} 或 {"quest.stage": {"gte": 2}} 的简写。
    results: list[bool] = []
    for path, expected in condition.items():
        if path in {"condition", "conditions", "requires"}:
            results.append(_eval(expected, env))
        elif isinstance(expected, Mapping):
            results.append(_compare_spec(_lookup(path, env), expected))
        else:
            results.append(_compare(_lookup(path, env), "eq", expected))
    return all(results)


def _as_conditions(value: Any) -> list[Any]:
    return list(value) if _is_list(value) else [value]


def _lookup(path: Any, env: Mapping[str, Any]) -> Any:
    if path is None:
        return _MISSING
    if not isinstance(path, str):
        try:
            return env.get(path, _MISSING)
        except TypeError:  # 不可哈希的键
            return _MISSING
    if path in env:
        return env[path]
    current: Any = env
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.lstrip("-").isdigit():
            try:
                current = current[int(part)]
            except IndexError:
                return _MISSING
        else:
            return _MISSING
    return current


def _compare(actual: Any, op: Any, expected: Any) -> bool:
    if actual is _MISSING:
        return False
    operator = str(op or "eq").lower()
    try:
        if operator in {"eq", "==", "equals"}:
            return actual == expected
        if operator in {"ne", "!=", "not_equals"}:
            return actual != expected
        if operator in {"gt", ">"}:
            return actual > expected
        if operator in {"gte", "ge", ">="}:
            return actual >= expected
        if operator in {"lt", "<"}:
            return actual < expected
        if operator in {"lte", "le", "<="}:
            return actual <= expected
        if operator in {"in", "contains"}:
            return actual in expected
        if operator in {"not_in", "notin"}:
            return actual not in expected
    except (TypeError, ValueError):
        return False
    return False


def _compare_spec(actual: Any, spec: Mapping[str, Any]) -> bool:
    if not any(op in spec for op in ("eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in")):
        return _compare(actual, "eq", spec)
    return all(_compare(actual, op, expected) for op, expected in spec.items())


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _eval_ast(node: ast.AST, env: Mapping[str, Any]) -> Any:
    """只允许字面量、变量、四则运算和比较，杜绝任意代码执行。"""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, _MISSING)
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast(value, env) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        operation = _BINOPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported binary operator")
        left, right = _eval_ast(node.left, env), _eval_ast(node.right, env)
        if left is _MISSING or right is _MISSING:
            return _MISSING
        return operation(left, right)
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator, env)
            operation = _CMPOPS.get(type(op))
            if operation is None or left is _MISSING or right is _MISSING or not operation(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Attribute):
        base = _eval_ast(node.value, env)
        return base.get(node.attr, _MISSING) if isinstance(base, Mapping) else _MISSING
    if isinstance(node, ast.Subscript):
        base, index = _eval_ast(node.value, env), _eval_ast(node.slice, env)
        if base is _MISSING or index is _MISSING:
            return _MISSING
        try:
            return base[index]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast(element, env) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval_ast(key, env): _eval_ast(value, env)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.IfExp):
        return _eval_ast(node.body, env) if _eval_ast(node.test, env) else _eval_ast(node.orelse, env)
    raise ValueError("unsupported expression")


# ---------------------------------------------------------------------------
# 服务器判定
# ---------------------------------------------------------------------------


def server_authority(local_state: Any, server_state: Any) -> dict[str, Any]:
    """以服务端判定为准合并客户端预测状态。

    服务端的进度字段覆盖客户端存档，客户端只保留显示偏好。两边进度不一致
    时返回 ``resync=True`` 与冲突列表，客户端据此回滚到服务端画面；服务端
    缺失的进度字段不会被客户端值顶替。
    """

    local = _as_mapping(local_state)
    server = _as_mapping(server_state)
    merged = copy.deepcopy(server)

    client_only: dict[str, Any] = {}
    for key, value in local.items():
        if key in LOCAL_ONLY_KEYS:
            merged[key] = copy.deepcopy(value)
        elif key in PROGRESS_KEYS and key not in server:
            client_only[key] = copy.deepcopy(value)
        elif key not in merged:
            merged[key] = copy.deepcopy(value)

    conflicts = [
        {"field": key, "client": copy.deepcopy(value), "server": copy.deepcopy(server[key])}
        for key, value in local.items()
        if key not in LOCAL_ONLY_KEYS and key in server and value != server[key]
    ]

    local_rev = local.get("revision", local.get("rev"))
    server_rev = server.get("revision", server.get("rev"))
    resync = bool(conflicts) or bool(client_only)
    if local_rev is not None and server_rev is not None:
        resync = resync or local_rev != server_rev

    # 选项可见性也由服务端统一求值：本地已解锁列表不足以推翻服务端条件。
    env = _eval_env(merged)
    raw_options = merged.get("options")
    if _is_list(raw_options):
        unlocked = {str(item) for item in _flatten_ids(merged.get("unlocked"))}
        visible = []
        for index, raw in enumerate(raw_options):
            option = _normalize_option(raw, index)
            shown = bool(option.get("visible", True)) and not bool(option.get("hidden"))
            condition = option.get("condition", option.get("visible_if"))
            if condition is not None:
                shown = shown and condition_eval(condition, env)
            if not _is_unlocked(option) and str(option["id"]) not in unlocked:
                shown = False
            if shown:
                visible.append(option)
        merged["options"] = visible

    merged["authoritative"] = "server"
    merged["resync"] = resync
    merged["conflicts"] = conflicts
    merged["client_only_progress"] = client_only
    merged["client_revision"] = local_rev
    merged["server_revision"] = server_rev
    return merged


def _eval_env(state: Mapping[str, Any]) -> dict[str, Any]:
    """条件求值环境：``variables`` 中的键优先，其余字段同名可用。"""

    env = dict(state["variables"]) if isinstance(state.get("variables"), Mapping) else {}
    env.update({key: value for key, value in state.items() if key != "variables"})
    return env


# ---------------------------------------------------------------------------
# 选项状态：返回恢复 / 已选标记 / 顺序
# ---------------------------------------------------------------------------


def restore_options(node: Any, history: Any = None) -> dict[str, Any]:
    """返回上一层时恢复选项及其解锁、已选状态。

    ``node`` 是对话节点映射（含 ``options``）或选项序列；``history`` 是历史
    栈、已选 id 列表或带 ``history`` 的状态。选项保持作者顺序，并补上
    ``unlocked`` / ``seen`` / ``available``。
    """

    node_map = _as_mapping(node)
    node_chosen, node_unlocked = _history_state(node_map.get("history"))
    history_chosen, history_unlocked = _history_state(history)

    chosen = _unique(_flatten_ids(node_map.get("selected")) + node_chosen + history_chosen)
    unlocked = _unique(_flatten_ids(node_map.get("unlocked")) + node_unlocked + history_unlocked)
    chosen_set = {str(item) for item in chosen}
    unlocked_set = {str(item) for item in unlocked}
    env = _eval_env(node_map)

    options = _options_of(node)
    for option in options:
        oid = str(option["id"])
        option["seen"] = bool(option.get("seen")) or oid in chosen_set
        option["unlocked"] = oid in unlocked_set or _is_unlocked(option)
        condition = option.get("condition", option.get("visible_if"))
        shown = bool(option.get("visible", True)) and not bool(option.get("hidden"))
        if condition is not None:
            shown = shown and condition_eval(condition, env)
        option["available"] = bool(option["unlocked"] and shown)

    return {
        "node": node_map.get("node", node_map.get("id", node_map.get("node_id"))),
        "options": options,
        "restored": True,
        # 回退后已解锁、已选状态仍然保留；历史里选过但当前节点没有的 id
        # 也一并带出，UI 可显示完整记录。
        "selected": _unique(chosen + [option["id"] for option in options if option["seen"]]),
        "unlocked": _unique(unlocked + [option["id"] for option in options if option["unlocked"]]),
        "available": [option["id"] for option in options if option["available"]],
    }


def history_stack(frame: Any, choice: Any = None) -> list[dict[str, Any]]:
    """把当前节点连同完整选项快照压入历史栈。

    用法：

    * ``history_stack(node, "a")`` —— 压入节点及本次选择；
    * ``history_stack(stack, node)`` —— 把新节点追加到已有栈；
    * ``history_stack(stack, "a")`` —— 只在栈顶补记选择。

    返回新栈，不修改入参。每帧保留该层选项与解锁状态，因此逐层返回时
    已解锁的选项不会丢失。
    """

    stack: list[dict[str, Any]] = []
    node_map: dict[str, Any] = {}
    if isinstance(frame, Mapping) and isinstance(frame.get("history"), list):
        stack = copy.deepcopy(frame["history"])
        node_map = copy.deepcopy(dict(frame))
    elif isinstance(frame, list) and all(isinstance(item, Mapping) and "chosen" in item for item in frame):
        stack = copy.deepcopy(frame)
        if isinstance(choice, Mapping) and ("options" in choice or "node" in choice):
            node_map = copy.deepcopy(dict(choice))
            choice = choice.get("chosen")
    else:
        node_map = _as_mapping(frame)

    if not node_map and choice is None:
        return stack  # 仅回传现有历史，不压入空帧

    options = _options_of(node_map)

    chosen_id: Any = None
    chosen_index: Any = None
    if isinstance(choice, Mapping):
        chosen_id = next((choice[key] for key in ("id", "option_id", "option", "value") if key in choice), None)
        chosen_index = choice.get("index", choice.get("line_index"))
    elif isinstance(choice, int) and not isinstance(choice, bool) and 0 <= choice < len(options):
        chosen_index = choice
        chosen_id = options[choice]["id"]
    else:
        chosen_id = choice

    if chosen_index is not None:
        try:
            chosen_index = int(chosen_index)
        except (TypeError, ValueError):
            chosen_index = None

    unlocked = [option["id"] for option in options if _is_unlocked(option)]
    for option in options:
        if chosen_id is not None and str(option["id"]) == str(chosen_id):
            unlocked.extend(_flatten_ids(option.get("unlocks")))
            option["seen"] = True

    stack.append(
        {
            "node": node_map.get("node", node_map.get("id", node_map.get("node_id"))),
            "options": copy.deepcopy(options),
            "chosen": chosen_id,
            "chosen_index": chosen_index,
            "unlocked": _unique(unlocked),
            "seen": _unique([option["id"] for option in options if option.get("seen")]),
        }
    )
    return stack


def seen_marker(node: Any, selected: Any = None) -> list[dict[str, Any]]:
    """给选过的选项附加持久化的已选标记，供 UI 显示“已选过”。

    ``selected`` 可以是本次选择、已选 id 列表，或包含 ``seen``/``selected``/
    ``history`` 的状态；节点自带的 ``seen`` 字段同样生效。
    """

    node_map = _as_mapping(node)
    seen_ids = _flatten_ids(selected)
    for key in ("seen", "selected", "chosen", "history"):
        if key in node_map:
            seen_ids.extend(_flatten_ids(node_map[key]))
    seen_set = {str(item) for item in seen_ids}

    options = _options_of(node)
    for option in options:
        option["seen"] = bool(option.get("seen")) or str(option["id"]) in seen_set
        option["marker"] = "seen" if option["seen"] else None
    return options


def option_order(options: Any, locale: Any = None) -> list[dict[str, Any]]:
    """按作者定义顺序排列选项；译文长度和换行不参与排序。

    排序只依据 ``order``（缺省为输入下标），``locale`` 仅为调用方兼容保留，
    因此本地化后 A/B 的顺序和点击目标不会互换。
    """

    items = _options_of(options)

    def sort_key(pair: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, option = pair
        try:
            return float(option.get("order", index)), index
        except (TypeError, ValueError):
            return float(index), index

    ordered = [option for _, option in sorted(enumerate(items), key=sort_key)]
    for position, option in enumerate(ordered):
        option["position"] = position
        option.setdefault("locale", locale)
    return ordered


# ---------------------------------------------------------------------------
# 存档
# ---------------------------------------------------------------------------


def _write_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """先写临时文件再替换，避免存档写到一半断电留下损坏文件。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def mid_save(state: Any, path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """保存对话中途进度：节点、行号、选项下标、历史栈与任务变量。

    传入路径时原子写入 JSON；返回的存档同时可用于内存读档。读档落到当前
    台词与选项，而不是对话第一句。
    """

    data = _as_mapping(state)
    cursor = data.get("cursor") if isinstance(data.get("cursor"), Mapping) else {}

    node = data.get("node", data.get("node_id", cursor.get("node", cursor.get("node_id"))))
    line_index = data.get("line_index", cursor.get("line_index", data.get("line", 0)))
    option_index = data.get("option_index", cursor.get("option_index", data.get("option", 0)))
    if not isinstance(line_index, int) or isinstance(line_index, bool):
        line_index = 0
    if not isinstance(option_index, int) or isinstance(option_index, bool):
        option_index = 0

    snapshot: dict[str, Any] = {
        "version": SAVE_VERSION,
        "kind": "dialogue_mid_save",
        "node": node,
        "line_index": line_index,
        "option_index": option_index,
        "cursor": {"node": node, "line_index": line_index, "option_index": option_index},
        "history": copy.deepcopy(data.get("history", cursor.get("history", []))),
        "selected": copy.deepcopy(data.get("selected", [])),
        "seen": copy.deepcopy(data.get("seen", [])),
        "unlocked": copy.deepcopy(data.get("unlocked", [])),
        "variables": copy.deepcopy(data.get("variables", {})),
        "rng_state": copy.deepcopy(data.get("rng_state", data.get("rng"))),
        "state": copy.deepcopy(data),
    }

    if isinstance(path, (str, os.PathLike)):
        _write_atomic(path, snapshot)
        snapshot["path"] = os.fspath(path)
    return snapshot


def load_save(source: Any) -> dict[str, Any]:
    """读取 ``mid_save`` 生成的存档（映射或文件路径），恢复中途游标。"""

    if isinstance(source, Mapping):
        snapshot = copy.deepcopy(dict(source))
    elif isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    else:
        raise TypeError("source must be a save mapping or a file path")

    version = snapshot.get("version", SAVE_VERSION)
    if isinstance(version, int) and version > SAVE_VERSION:
        raise ValueError(f"save version {version} is newer than supported {SAVE_VERSION}")

    state = snapshot.get("state")
    restored = copy.deepcopy(dict(state)) if isinstance(state, Mapping) else {}
    restored.update(
        {
            "node": snapshot.get("node"),
            "line_index": snapshot.get("line_index", 0),
            "option_index": snapshot.get("option_index", 0),
            "history": copy.deepcopy(snapshot.get("history", [])),
            "cursor": copy.deepcopy(snapshot.get("cursor", {})),
            "resumed": True,
            "save_version": snapshot.get("version"),
        }
    )
    return restored


# ---------------------------------------------------------------------------
# 语音 / 字幕 / 自动播放
# ---------------------------------------------------------------------------


def _seconds(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return max(0.0, _number(value))
    if isinstance(value, Mapping):
        for key in ("duration", "seconds", "time", "length"):
            if key in value:
                return _seconds(value[key])
    return 0.0


def _reading_time(text: str, chars_per_second: float = 12.0) -> float:
    """字幕阅读时长；长句按字符数延长，避免字幕提前消失。"""

    if not text:
        return 0.0
    return max(1.0, len(str(text)) / (chars_per_second if chars_per_second > 0 else 12.0))


def voice_wait(voice_duration: Any, subtitle_duration: Any = 0.0, margin: Any = 0.0) -> float:
    """自动播放的等待时长：语音与字幕都结束后才推进，并留安全边距。"""

    if isinstance(voice_duration, Mapping):
        config = voice_duration
        voice = _seconds(
            config.get("voice_duration", config.get("voice", config.get("audio_duration", 0.0)))
        )
        subtitle = _seconds(config.get("subtitle_duration", config.get("subtitle", subtitle_duration)))
    else:
        voice = _seconds(voice_duration)
        subtitle = _seconds(subtitle_duration)
    return round(max(voice, subtitle) + max(0.0, _seconds(margin)), 3)


def timeline_unify(voice_duration: Any, subtitle: Any = 0.0, chars_per_second: float = 12.0) -> dict[str, Any]:
    """把语音轨与字幕轨合并到同一条时间轴。

    字幕不早于语音结束；长句字幕按阅读速度延长。返回的 ``duration`` 是两者
    中较晚结束的时间，UI 与自动播放都读这一条时间轴，不会各算一套。
    """

    config = voice_duration if isinstance(voice_duration, Mapping) else {}
    voice = _seconds(
        config.get("voice_duration", config.get("voice", config.get("audio_duration", voice_duration)))
    )

    raw_subtitle = config.get("subtitle", subtitle)
    if isinstance(raw_subtitle, str):
        subtitle_duration = _reading_time(raw_subtitle, chars_per_second)
        source = "reading_time"
    else:
        subtitle_duration = _seconds(raw_subtitle)
        source = "explicit"

    duration = max(voice, subtitle_duration)
    return {
        "clock": "shared",
        "voice_duration": round(voice, 3),
        "subtitle_duration": round(subtitle_duration, 3),
        "voice_end": round(voice, 3),
        "subtitle_end": round(duration, 3),
        "duration": round(duration, 3),
        "subtitle_source": source,
        # 字幕短于语音时被延长到语音结束，UI 不再提前收起字幕。
        "subtitle_extended": subtitle_duration < voice,
    }


def auto_play(voice_duration: Any, subtitle_duration: Any = 0.0, margin: float = 0.1) -> dict[str, Any]:
    """自动播放推进计划：统一时间轴结束后才允许进入下一句。"""

    timeline = timeline_unify(voice_duration, subtitle_duration)
    return {
        "advance_after": round(timeline["duration"] + max(0.0, _number(margin)), 3),
        "wait_for_voice": True,
        "voice_end": timeline["voice_end"],
        "subtitle_end": timeline["subtitle_end"],
        "duration": timeline["duration"],
        "interruptible": True,
    }


# ---------------------------------------------------------------------------
# 布局 / 本地化
# ---------------------------------------------------------------------------


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _wrap_text(text: str, width: int) -> list[str]:
    """按显示宽度折行，保留显式换行；CJK 字符占 2 格。"""

    width = max(1, int(width))
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        current: list[str] = []
        used = 0
        for char in paragraph:
            char_width = _char_width(char)
            if current and used + char_width > width:
                lines.append("".join(current))
                current, used = [], 0
            current.append(char)
            used += char_width
        lines.append("".join(current))
    return lines or [""]


def layout_stable(options: Any, max_width: Any = 40, slot_lines: int | None = None) -> dict[str, Any]:
    """固定槽位布局：选项顺序、行高和点击区域不随译文长度变化。

    每个选项占同样的 ``slot_height``，``top`` 只由顺序决定。给出
    ``slot_lines`` 时槽位高度固定，超长译文标记 ``clipped`` 交给 UI 滚动，
    后续选项位置完全不变。
    """

    if isinstance(max_width, Mapping):
        config = max_width
        width = config.get("max_width", config.get("width", 40))
        if slot_lines is None:
            slot_lines = config.get("slot_lines", config.get("lines_per_slot"))
    else:
        width = max_width
    try:
        width = max(1, int(width))
    except (TypeError, ValueError):
        width = 40

    fixed_slot = slot_lines is not None
    try:
        slot_lines = max(1, int(slot_lines)) if fixed_slot else 1
    except (TypeError, ValueError):
        slot_lines = 1

    wrapped: list[tuple[dict[str, Any], list[str]]] = []
    for option in option_order(options):
        lines = _wrap_text(str(option.get("text", "")), width)
        wrapped.append((option, lines))
        if not fixed_slot:
            slot_lines = max(slot_lines, len(lines))

    slot_height = slot_lines
    rows = []
    for position, (option, lines) in enumerate(wrapped):
        top = position * slot_height
        rows.append(
            {
                "id": option["id"],
                "text": option.get("text", ""),
                "lines": lines,
                "top": top,
                "height": slot_height,
                "action": option.get("target", option.get("action", option["id"])),
                "stable_index": position,
                "clipped": len(lines) > slot_lines,
                "overflow_lines": max(0, len(lines) - slot_lines),
                "hitbox": {"x": 0, "y": top, "width": width, "height": slot_height},
            }
        )

    return {
        "width": width,
        "slot_lines": slot_lines,
        "slot_height": slot_height,
        "fixed_slot": fixed_slot,
        "rows": rows,
        "order": [row["id"] for row in rows],
        "stable": True,
    }


def localization_width(text_or_base: Any, limit_or_localized: Any = 40) -> dict[str, Any]:
    """测量本地化文本宽度，或比较基准/译文两种语言的宽度。

    文本输入返回折行结果；两个数值输入（基准宽度、译文宽度）返回放大比例
    与建议槽位宽度，供布局层预留空间。
    """

    if isinstance(text_or_base, Mapping):
        mapping = text_or_base
        if "text" in mapping:
            return localization_width(
                mapping["text"], mapping.get("max_width", mapping.get("width", limit_or_localized))
            )
        return localization_width(
            mapping.get("base_width", 0), mapping.get("localized_width", limit_or_localized)
        )

    if isinstance(text_or_base, str):
        try:
            max_width = max(1, int(limit_or_localized))
        except (TypeError, ValueError):
            max_width = 40
        lines = _wrap_text(text_or_base, max_width)
        natural = max((_display_width(line) for line in lines), default=0)
        return {
            "text": text_or_base,
            "display_width": _display_width(text_or_base),
            "max_width": max_width,
            "lines": lines,
            "line_count": len(lines),
            "natural_line_width": natural,
            "fits": natural <= max_width,
        }

    base = max(0.0, _number(text_or_base))
    localized = max(0.0, _number(limit_or_localized, base))
    return {
        "base_width": base,
        "localized_width": localized,
        "scale": round(localized / base, 3) if base else 1.0,
        "recommended_width": math.ceil(max(base, localized)),
        "extra_cells": max(0.0, localized - base),
    }


# ---------------------------------------------------------------------------
# 发言人
# ---------------------------------------------------------------------------


def speaker_lock(speakers: Any, candidate: Any = None) -> dict[str, Any]:
    """同一时刻只允许一个发言人，抢占者进入先进先出的等待队列。

    ``speakers`` 可以是锁状态（含 ``holder``/``queue``）或当前持锁者，
    ``candidate`` 是请求发言的角色。返回新状态：``granted`` 表示是否拿到
    发言权，``holder`` 始终是唯一正在说话的人。
    """

    if isinstance(speakers, Mapping):
        state = copy.deepcopy(dict(speakers))
    else:
        state = {"holder": copy.deepcopy(speakers)}

    holder = state.get("holder", state.get("speaker"))
    raw_queue = state.get("queue", [])
    if isinstance(raw_queue, Mapping):
        raw_queue = list(raw_queue.values())
    elif not _is_list(raw_queue):
        raw_queue = [] if raw_queue is None else [raw_queue]
    queue = [copy.deepcopy(item) for item in raw_queue]

    requester = candidate
    if isinstance(candidate, Mapping):
        requester = candidate.get("speaker", candidate.get("id", candidate.get("name")))

    granted = False
    if requester is not None:
        if holder is None or holder == requester:
            holder, granted = requester, True
        elif requester not in queue:
            queue.append(requester)

    state.update({"holder": holder, "speaker": holder, "locked": holder is not None})
    state["queue"] = queue
    return {
        **state,
        "holder": holder,
        "speaker": holder,
        "locked": holder is not None,
        "granted": granted,
        "queue": queue,
        "waiting": list(queue),
    }


# ---------------------------------------------------------------------------
# 权重随机
# ---------------------------------------------------------------------------


def weighted_random(seed: Any, weights: Any = None) -> int:
    """按数据中的权重返回选项下标。

    权重不再写死在代码里：``weights`` 可以是权重列表、选项数量（等权），
    或带 ``weight`` 的选项列表；同一 ``seed`` 结果可复现，换种子（例如每次
    进入分支时用存档计数器）就会得到不同结果。
    """

    source = weights
    if isinstance(seed, Mapping) and weights is None:
        source = seed.get("weights", seed.get("options"))
        seed = seed.get("seed")

    if isinstance(source, Mapping):
        raw = source.get("weights", source.get("options"))
        if isinstance(raw, Mapping):
            raw_items = list(raw.values())
        elif _is_list(raw):
            raw_items = list(raw)
        else:
            raw_items = [source.get("weight", 1.0)]
    elif isinstance(source, int) and not isinstance(source, bool):
        raw_items = [1.0] * max(0, source)
    elif _is_list(source):
        raw_items = list(source)
    elif source is None:
        raw_items = []
    else:
        raw_items = [source]

    parsed: list[float] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            value = item.get("weight", item.get("probability", 1.0))
        elif isinstance(item, (list, tuple)) and item:
            value = item[-1]
        else:
            value = item
        weight = _number(value, 0.0)
        parsed.append(float("inf") if math.isinf(weight) else max(0.0, weight))

    if not parsed:
        return 0

    rng = random.Random(seed) if seed is not None else random.Random()
    infinite = [index for index, weight in enumerate(parsed) if not math.isfinite(weight)]
    if infinite:  # 无限权重始终优先
        return infinite[rng.randrange(len(infinite))]

    total = sum(parsed)
    if total <= 0:
        return rng.randrange(len(parsed))

    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(parsed):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(parsed) - 1


# ---------------------------------------------------------------------------
# 跳过补偿
# ---------------------------------------------------------------------------


def skip_compensate(skipped: Any, quest_state: Any = None) -> dict[str, Any]:
    """跳过对话时补执行被跳过节点上的事件与任务条件。

    返回新的任务状态、已补执行的事件与已满足的条件。条件在该节点的事件
    生效后求值，与正常播放到该节点的结果一致，因此跳过不会卡住后续流程。
    """

    state = _as_mapping(quest_state)
    if isinstance(skipped, Mapping):
        entries = skipped.get("skipped", skipped.get("nodes", skipped))
    else:
        entries = skipped
    entries = list(entries) if _is_list(entries) else [entries]

    applied: list[dict[str, Any]] = []
    conditions: list[Any] = []
    met: list[Any] = []

    for entry in entries:
        if not isinstance(entry, Mapping):
            _apply_event(entry, state, applied)
            continue

        events = entry.get("events", entry.get("on_skip", entry.get("effects")))
        if isinstance(events, Mapping):
            events = [events]
        elif events is None:
            events = []
        elif not _is_list(events):
            events = [events]
        events = list(events)

        has_effect = any(
            key in entry
            for key in ("set", "add", "append", "grant", "item", "quest", "complete_quest", "unlock", "flag")
        )
        if not events and (has_effect or any(key in entry for key in ("type", "kind", "op", "event"))):
            events = [entry]
        elif has_effect:
            events.append(entry)

        for event in events:
            _apply_event(event, state, applied)

        entry_conditions: list[Any] = []
        for key in ("conditions", "condition", "requires"):
            if key in entry:
                entry_conditions.extend(_as_conditions(entry[key]))
        conditions.extend(entry_conditions)
        for condition in entry_conditions:
            if condition_eval(condition, _eval_env(state)):
                met.append(condition)

    if met:
        existing = state.get("conditions_met", [])
        state["conditions_met"] = _unique([*(existing if _is_list(existing) else [existing]), *met])

    return {
        "state": state,
        "applied": applied,
        "conditions": conditions,
        "conditions_met": met,
        "skipped_count": len(entries),
        "compensated": True,
    }


def _state_set(state: dict[str, Any], path: Any, value: Any) -> None:
    """按点分路径写入任务状态，例如 ``quest.kills``。"""

    if not isinstance(path, str):
        state[path] = copy.deepcopy(value)
        return
    current = state
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _state_get(state: Mapping[str, Any], path: Any, default: Any = None) -> Any:
    if not isinstance(path, str):
        return state.get(path, default)
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _apply_event(event: Any, state: dict[str, Any], applied: list[dict[str, Any]]) -> None:
    """把一条补偿事件写入任务状态，并记录到 ``applied`` 供存档回放。"""

    if event is None:
        return
    if isinstance(event, str):
        events = state.setdefault("events", [])
        if isinstance(events, list) and event not in events:
            events.append(event)
        applied.append({"event": event})
        return
    if _is_list(event):
        for item in event:
            _apply_event(item, state, applied)
        return
    if not isinstance(event, Mapping):
        applied.append({"event": event})
        return

    payload = copy.deepcopy(dict(event))

    # 简写：{"set": {...}} / {"add": {...}} / {"append": {...}}
    if isinstance(payload.get("set"), Mapping):
        for key, value in payload["set"].items():
            _state_set(state, key, value)
            applied.append({"type": "set", "key": key, "value": value})
    if isinstance(payload.get("add"), Mapping):
        for key, delta in payload["add"].items():
            _state_set(state, key, _number(_state_get(state, key, 0)) + _number(delta))
            applied.append({"type": "add", "key": key, "delta": delta})
    if isinstance(payload.get("append"), Mapping):
        for key, value in payload["append"].items():
            current = _state_get(state, key, [])
            _state_set(state, key, [*(current if isinstance(current, list) else [current]), value])
            applied.append({"type": "append", "key": key, "value": value})

    kind = str(payload.get("type", payload.get("kind", payload.get("op", "")))).lower()
    key = payload.get("key", payload.get("name", payload.get("var")))
    value = payload.get("value", payload.get("amount", payload.get("delta", True)))

    def append_to(name: str, item: Any) -> None:
        bucket = state.setdefault(name, [])
        if isinstance(bucket, list) and item not in bucket:
            bucket.append(item)

    if kind in {"set", "assign"} and key is not None:
        _state_set(state, key, value)
        applied.append({"type": "set", "key": key, "value": value})
    elif kind in {"add", "increment"} and key is not None:
        _state_set(state, key, _number(_state_get(state, key, 0)) + _number(value))
        applied.append({"type": "add", "key": key, "delta": _number(value)})
    elif kind in {"append", "push"} and key is not None:
        current = _state_get(state, key, [])
        _state_set(state, key, [*(current if isinstance(current, list) else [current]), value])
        applied.append({"type": "append", "key": key, "value": value})
    elif kind in {"grant", "item", "give"} and key is not None:
        items = state.setdefault("items", {})
        if isinstance(items, dict):
            items[key] = _number(items.get(key, 0)) + max(1.0, _number(value, 1.0))
        applied.append({"type": "grant", "key": key, "value": value})
    elif kind in {"complete", "complete_quest"} and key is not None:
        quests = state.setdefault("quests", {})
        if isinstance(quests, dict):
            quests[key] = payload.get("status", "complete")
        applied.append({"type": "quest_complete", "key": key})
    elif kind in {"unlock", "discover"} and key is not None:
        append_to("unlocked", key)
        applied.append({"type": "unlock", "key": key})
    elif kind in {"flag", "toggle"} and key is not None:
        flags = state.setdefault("flags", {})
        if isinstance(flags, dict):
            flags[key] = value
        applied.append({"type": "flag", "key": key, "value": value})
    elif kind in {"event", "trigger"} and key is not None:
        append_to("events", key)
        applied.append({"type": "event", "key": key})
    elif not kind:
        # 无类型的简写对象按“键 -> 值”补写；已处理的字段不重复写入。
        handled = {
            "set", "add", "append", "type", "kind", "op", "key", "name", "var",
            "value", "amount", "delta", "condition", "conditions", "requires",
        }
        for item_key, item_value in payload.items():
            if item_key not in handled:
                _state_set(state, item_key, item_value)
                applied.append({"type": "set", "key": item_key, "value": item_value})


__all__ = [
    "restore_options",
    "server_authority",
    "seen_marker",
    "mid_save",
    "load_save",
    "voice_wait",
    "layout_stable",
    "speaker_lock",
    "timeline_unify",
    "weighted_random",
    "skip_compensate",
    "condition_eval",
    "option_order",
    "history_stack",
    "auto_play",
    "localization_width",
]
