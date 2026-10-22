# Academic Paper Drafts → Word（扩充版 ~16 页）

## 最终交付

| 文件 | 说明 |
|------|------|
| [`word/MeetingRH_Paper_EN.docx`](word/MeetingRH_Paper_EN.docx) | **英文正式稿**（推荐，约 16 页） |
| [`word/MeetingRH_Paper_ZH.docx`](word/MeetingRH_Paper_ZH.docx) | **中文正式稿**（推荐，约 16 页） |

## 本版相对旧稿的扩充

- **Method 大幅加细**：会商触发/流水线、时空 BFS、StageContract、StageNet 8 维特征与 ASA、FACN 22 维特征与 AFA/TBC、全部协议阈值（竞争目的地 / 垫区出口 / FG-CEBI / E4 / 进度脉冲 / PGFA / DLO / 区域准入等）
- **参考文献 60 篇**（CBS 族、优先/窗口、PIBT/LaCAM、终身 MAPD、学习、仓储等）
- 图 1–4、主结果表、协议对照表、算法伪代码、附录常数速查

## 重新生成

```bash
cd docs/paper
python build_word_expanded.py
```

依赖：`python-docx`（`pip install python-docx`）。

## 源文件

| 路径 | 说明 |
|------|------|
| `build_word_expanded.py` | 扩充版 Word 生成器 |
| `word_fmt.py` / `refs60.py` | 格式工具与 60 篇文献列表 |
| `refs_extended.bib` | BibTeX（~60） |
| `figures/` | 插图 |

说明：页数随 Word 字体/打印机略有浮动；目标会议双栏模板下页数会再压缩。录用前请按 AAAI/IJCAI/AAMAS 官方模板微调。
