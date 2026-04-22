# MOSx

基于 Python 的本地桌面工具，用于从宽表 CSV 数据中提取 MOS 晶体管参数并输出结果表。

## 当前支持

- 通过多个列联合区分每个晶体管
- 映射曲线类型列、Drain Bias 条件列、Gate Voltage 列、Drain Current 列
- 为每个晶体管设置 `Width / Length (nm)`，支持首行一键应用到全部
- 计算 `Idoff`、`Ids`、`Idl`、`Vts`、`Vtl`
- 支持输入单位切换与 Gate Length 归一化
- 支持配置 `Vdd`、低漏压目标值、阈值常数 `nA * W / L`
- 支持切换 `NMOS / PMOS` 极性处理
- 结果表导出 CSV、复制全部
- 绘制曲线并标出 `Idoff / Ids / Idl / Vts / Vtl` 取点位置

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行

```powershell
python app.py
```

## 计算约定

- `Idoff`: 高漏压条件下，取最接近 `Vg = 0V` 的点，输出单位 `pA`
- `Ids`: 高漏压条件下，取最接近配置 `Vdd` 的 `Vg` 点，输出单位 `uA`
- `Idl`: 低漏压条件下，取最接近配置 `Vdd` 的 `Vg` 点，输出单位 `uA`
- `Vts / Vtl`: 在对应 `IdVg` 曲线上，以 `阈值常数 * W / L` 为目标电流，使用相邻点线性插值求 `Vg`
- `PMOS / NMOS`: 计算和绘图会按所选极性处理电流方向，避免 PMOS 负电流导致阈值插值方向错误

## 备注

- 如果 Drain Bias 区分列是数值列，软件会按 `Vdd` 和低漏压目标值自动挑选最接近的高/低偏压值
- 如果 Drain Bias 区分列是文本列，仍可手动通过下拉框指定哪些值对应高漏压和低漏压
