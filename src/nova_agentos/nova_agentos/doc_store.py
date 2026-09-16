"""工具使用说明(doc)存储:docs/<tool>.md,doc 名与工具名相同。

doc 是按需加载的工具手册,不进 skill 索引;模型从工具列表得知工具名后,
用 load_doc 读取对应说明。
"""
from __future__ import annotations
from pathlib import Path


class DocStore:
    """扫描 docs 目录,按工具名提供使用说明正文。"""

    def __init__(self, docs_dir: str | Path) -> None:
        self.root = Path(docs_dir)

    def names(self) -> list[str]:
        """返回所有可用 doc 名(文件名去掉 .md),按名排序。"""
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.md"))

    def load(self, names: list[str]) -> dict[str, str]:
        """按名加载 doc 正文;不存在的名字不返回。"""
        contents: dict[str, str] = {}
        for name in names:
            path = self.root / f"{name}.md"
            if path.is_file():
                contents[name] = path.read_text(encoding="utf-8")
        return contents
