#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re


ROOT = Path("/home/markwcl/MPC_ML")
OLD_PATH = ROOT / "paper_old.txt"
NEW_PATH = ROOT / "paper_3_16_v1.txt"
OUT_PATH = ROOT / "comparison.txt"

HELPERS = [
    r"\documentclass[conference]{IEEEtran}",
    r"\IEEEoverridecommandlockouts",
    r"% diff helpers for comparison draft",
    r"\RequirePackage{xcolor}",
    r"\RequirePackage[normalem]{ulem}",
    r"\newcommand{\DIFaddbegin}{\begingroup\color{blue}}",
    r"\newcommand{\DIFaddend}{\endgroup}",
    r"\newcommand{\DIFdelbegin}{\begingroup\color{red}}",
    r"\newcommand{\DIFdelend}{\endgroup}",
    r"\newcommand{\DIFadd}[1]{{\color{blue}#1}}",
    r"\newcommand{\DIFdel}[1]{{\color{red}\sout{#1}}}",
]

BLOCKED_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "bmatrix",
    "pmatrix",
    "vmatrix",
    "Vmatrix",
    "cases",
    "split",
    "figure",
    "figure*",
    "table",
    "table*",
    "tabular",
    "tabular*",
    "algorithm",
    "algorithmic",
    "IEEEeqnarray",
}

INLINE_BRACED_COMMANDS = [
    r"\caption",
    r"\section",
    r"\subsection",
    r"\subsubsection",
    r"\paragraph",
    r"\textbf",
]


@dataclass(frozen=True)
class LineInfo:
    blocked_depth: int


@dataclass(frozen=True)
class Unit:
    kind: str
    text: str


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


TEXTLIKE_PREFIXES = (
    r"\item ",
    r"\lettrine",
    r"\caption{",
    r"\section{",
    r"\subsection{",
    r"\subsubsection{",
    r"\paragraph{",
    r"\textbf{",
)

SINGLE_LINE_TEXTLIKE_PREFIXES = (
    r"\item ",
    r"\section{",
    r"\subsection{",
    r"\subsubsection{",
    r"\paragraph{",
    r"\caption{",
)

RAW_PREFIXES = (
    r"\documentclass",
    r"\IEEEoverridecommandlockouts",
    r"\usepackage",
    r"\RequirePackage",
    r"\def\BibTeX",
    r"\setcounter",
    r"\renewcommand",
    r"\allowdisplaybreaks",
    r"\title{",
    r"\author{",
    r"\IEEEauthorblock",
    r"\maketitle",
    r"\centering",
    r"\includegraphics",
    r"\label{",
)


def compute_line_info(lines: list[str]) -> list[LineInfo]:
    infos: list[LineInfo] = []
    blocked_depth = 0
    pattern = re.compile(r"\\(begin|end)\{([^}]+)\}")

    for line in lines:
        for kind, env in pattern.findall(line):
            if kind == "end" and env in BLOCKED_ENVS and blocked_depth > 0:
                blocked_depth -= 1

        infos.append(LineInfo(blocked_depth=blocked_depth))

        for kind, env in pattern.findall(line):
            if kind == "begin" and env in BLOCKED_ENVS:
                blocked_depth += 1

    return infos


def emit_block(kind: str, lines: list[str], out: list[str]) -> None:
    if not lines:
        return
    begin = r"\DIFaddbegin" if kind == "add" else r"\DIFdelbegin"
    end = r"\DIFaddend" if kind == "add" else r"\DIFdelend"
    out.append(begin)
    out.extend(lines)
    out.append(end)


def emit_units_block(kind: str, units: list[Unit], out: list[str]) -> None:
    emit_block(kind, [u.text for u in units], out)


def is_balanced(line: str) -> bool:
    brace = 0
    bracket = 0
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace < 0:
                return False
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
            if bracket < 0:
                return False
        i += 1
    return brace == 0 and bracket == 0


def strip_outer_command(line: str) -> tuple[str, str, str] | None:
    for cmd in INLINE_BRACED_COMMANDS:
        prefix = f"{cmd}{{"
        if not line.startswith(prefix) or not line.endswith("}"):
            continue
        inner = line[len(prefix):-1]
        if is_balanced(inner):
            return prefix, inner, "}"
    return None


def split_for_inline(line: str) -> tuple[str, str, str]:
    if line.startswith(r"\item "):
        return r"\item ", line[len(r"\item "):], ""

    outer = strip_outer_command(line)
    if outer is not None:
        return outer

    return "", line, ""


def parse_balanced(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    punct = set("{}[](),.;:!?~$%^&|<>/=+-")

    while i < len(text):
        ch = text[i]
        if ch.isspace():
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        if ch == "\\":
            start = i
            i += 1
            if i < len(text) and text[i].isalpha():
                while i < len(text) and (text[i].isalpha() or text[i] == "@"):
                    i += 1
                if i < len(text) and text[i] == "*":
                    i += 1
            elif i < len(text):
                i += 1

            while i < len(text) and text[i] in "[{":
                if text[i] == "[":
                    i = parse_balanced(text, i, "[", "]")
                else:
                    i = parse_balanced(text, i, "{", "}")
            tokens.append(text[start:i])
            continue

        if ch in punct:
            tokens.append(ch)
            i += 1
            continue

        j = i + 1
        while j < len(text):
            nxt = text[j]
            if nxt.isspace() or nxt == "\\" or nxt in punct:
                break
            j += 1
        tokens.append(text[i:j])
        i = j

    return tokens


def wrap_inline(kind: str, content: str) -> str:
    if not content:
        return ""
    macro = r"\DIFadd" if kind == "add" else r"\DIFdel"
    return f"{macro}{{{content}}}"


def inline_diff_text(old_text: str, new_text: str) -> str:
    old_tokens = tokenize(old_text)
    new_tokens = tokenize(new_text)
    sm = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    chunks: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            chunks.extend(old_tokens[i1:i2])
        elif tag == "delete":
            chunks.append(wrap_inline("del", "".join(old_tokens[i1:i2])))
        elif tag == "insert":
            chunks.append(wrap_inline("add", "".join(new_tokens[j1:j2])))
        else:
            chunks.append(wrap_inline("del", "".join(old_tokens[i1:i2])))
            chunks.append(wrap_inline("add", "".join(new_tokens[j1:j2])))

    return "".join(chunks)


def can_inline(old_line: str, new_line: str, old_info: LineInfo, new_info: LineInfo) -> bool:
    old_stripped = old_line.strip()
    new_stripped = new_line.strip()

    if old_info.blocked_depth > 0 or new_info.blocked_depth > 0:
        return False
    if not old_stripped or not new_stripped:
        return False
    if old_stripped.startswith("%") or new_stripped.startswith("%"):
        return False
    if r"\begin{" in old_line or r"\end{" in old_line or r"\begin{" in new_line or r"\end{" in new_line:
        return False
    if "&" in old_line or "&" in new_line:
        return False
    if old_stripped.endswith(r"\\") or new_stripped.endswith(r"\\"):
        return False
    if not is_balanced(old_line) or not is_balanced(new_line):
        return False
    return True


def inline_diff_line(old_line: str, new_line: str, old_info: LineInfo, new_info: LineInfo) -> str | None:
    if not can_inline(old_line, new_line, old_info, new_info):
        return None

    old_prefix, old_body, old_suffix = split_for_inline(old_line)
    new_prefix, new_body, new_suffix = split_for_inline(new_line)
    if old_prefix != new_prefix or old_suffix != new_suffix:
        return inline_diff_text(old_line, new_line)

    return old_prefix + inline_diff_text(old_body, new_body) + old_suffix


def is_textlike(line: str, info: LineInfo) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if info.blocked_depth > 0:
        return False
    if stripped.startswith("%"):
        return False
    if r"\begin{" in stripped or r"\end{" in stripped:
        return False
    if any(stripped.startswith(prefix) for prefix in RAW_PREFIXES):
        return False
    if any(stripped.startswith(prefix) for prefix in TEXTLIKE_PREFIXES):
        return True
    if stripped.startswith("\\"):
        return False
    return True


def split_units(lines: list[str], infos: list[LineInfo]) -> list[Unit]:
    units: list[Unit] = []
    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        if para_lines:
            units.append(Unit("paragraph", " ".join(s.strip() for s in para_lines)))
            para_lines = []

    for line, info in zip(lines, infos):
        stripped = line.strip()
        if stripped == "":
            flush_para()
            continue

        if is_textlike(line, info):
            if any(stripped.startswith(prefix) for prefix in SINGLE_LINE_TEXTLIKE_PREFIXES):
                flush_para()
                units.append(Unit("paragraph", stripped))
                continue
            para_lines.append(line)
            continue

        flush_para()
        units.append(Unit("raw", line))

    flush_para()
    return units


def diff_blocks(
    old_units: list[Unit],
    new_units: list[Unit],
) -> list[str]:
    out: list[str] = []
    sm = SequenceMatcher(
        a=[(u.kind, u.text) for u in old_units],
        b=[(u.kind, u.text) for u in new_units],
        autojunk=False,
    )

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(u.text for u in old_units[i1:i2])
            continue

        if tag == "delete":
            emit_units_block("del", old_units[i1:i2], out)
            continue

        if tag == "insert":
            emit_units_block("add", new_units[j1:j2], out)
            continue

        old_block = old_units[i1:i2]
        new_block = new_units[j1:j2]

        if len(old_block) == len(new_block):
            for old_unit, new_unit in zip(old_block, new_block):
                if old_unit == new_unit:
                    out.append(old_unit.text)
                    continue

                if old_unit.kind == new_unit.kind == "paragraph":
                    out.append(inline_diff_text(old_unit.text, new_unit.text))
                    continue

                if old_unit.kind == new_unit.kind == "raw":
                    annotated = inline_diff_line(
                        old_unit.text,
                        new_unit.text,
                        LineInfo(0),
                        LineInfo(0),
                    )
                    if annotated is None:
                        emit_units_block("del", [old_unit], out)
                        emit_units_block("add", [new_unit], out)
                    else:
                        out.append(annotated)
                else:
                    emit_units_block("del", [old_unit], out)
                    emit_units_block("add", [new_unit], out)
            continue

        if len(old_block) == len(new_block) == 1:
            old_unit = old_block[0]
            new_unit = new_block[0]
            if old_unit.kind == new_unit.kind == "paragraph":
                out.append(inline_diff_text(old_unit.text, new_unit.text))
                continue

        emit_units_block("del", old_block, out)
        emit_units_block("add", new_block, out)

    return out


def main() -> None:
    old_lines = read_lines(OLD_PATH)
    new_lines = read_lines(NEW_PATH)

    old_body = old_lines[2:]
    new_body = new_lines[2:]
    old_infos = compute_line_info(old_body)
    new_infos = compute_line_info(new_body)
    old_units = split_units(old_body, old_infos)
    new_units = split_units(new_body, new_infos)

    out_lines = HELPERS + diff_blocks(old_units, new_units)
    OUT_PATH.write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
