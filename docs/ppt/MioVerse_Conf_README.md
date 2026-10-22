# Lifelong MAPD Hybrid Stack — Conference PPT（重做）

## 交付（中英各 26 页）

| 文件 | 说明 |
|------|------|
| `Lifelong_MAPD_Hybrid_ZH.pptx` | 中文（推荐） |
| `Lifelong_MAPD_Hybrid_EN.pptx` | English |
| `MioVerse_Conf_ZH.pptx` / `_EN.pptx` | 同步写出的副本（文件名兼容） |
| `build_conf_ppt.py` | 生成脚本 |
| `generateimage/mioverse-conf-ppt/{zh,en}/` | 论文风配图（`*-fig-*.jpg`） |

## 叙述顺序（已按要求重排）

1. **研究问题与价值**（无竞赛/品牌叙事）+ 规则与 VALID  
2. **历史方法**（CBS/ECBS、PP、PIBT、MAPD、RHCR + 缺口）  
3. **我们的方法与创新点** + 总览图  
4. **细节**  
   - 4.1 Meeting 生命周期图 + 触发/I/O + 内部过程  
   - 4.2 协议工具栈（故障→工具）  
   - 4.3 StageNet / FACN 结构图 + **紧随消融**（**无 RuleNet**）  
5. **结果对比**（PIBT/M0/PP/ECBS/Hier/Ours：full100·VALID·done·sim·wall）+ 为何最好  

## 重新生成

```powershell
C:\Users\zkb17\miniconda3\python.exe docs\ppt\build_conf_ppt.py
```
