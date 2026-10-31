"""Convert the paper draft (GitHub-flavoured Markdown) into a standalone LaTeX body.

Deliberately narrow: it handles exactly the constructs `paper/DRAFT.md` uses --
headings, pipe tables, images, bullet lists, fenced code, `[@key]` citations,
inline bold/italic/code -- and refuses to guess at anything else. Text is
escaped before inline formatting is re-applied, so a literal `_` or `%` in prose
cannot break the build.

Emits only the body; the preamble lives in `paper/latex/main.tex`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

# XeTeX with the Times package has no glyphs for these; spell them in LaTeX.
UNICODE = {"\u2014": "---", "\u2013": "--", "\u2212": "$-$", "\u2248": "$\\approx$",
           "\u2264": "$\\leq$", "\u2265": "$\\geq$", "\u00b1": "$\\pm$",
           "\u00b2": "$^2$", "\u00b3": "$^3$", "\u00d7": "$\\times$",
           "\u00b7": "$\\cdot$", "\u00b0": "$^\\circ$", "\u2192": "$\\to$",
           "\u0394": "$\\Delta$", "\u03b1": "$\\alpha$", "\u03b2": "$\\beta$",
           "\u03b3": "$\\gamma$", "\u03bb": "$\\lambda$", "\u03bc": "$\\mu$",
           "\u03c1": "$\\rho$", "\u03c3": "$\\sigma$", "\u03c4": "$\\tau$",
           "\u03c7": "$\\chi$", "\u03b7": "$\\eta$", "\u00b5": "$\\mu$"}


def escape(text):
    return "".join(SPECIAL.get(ch, ch) for ch in text)


def inline(text):
    """Escape LaTeX specials, then restore the constructs we do support."""
    held = []

    def hold(value):
        held.append(value)
        return f"\x00{len(held) - 1}\x00"

    for char, latex in UNICODE.items():
        text = text.replace(char, hold(latex))
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + escape(m.group(1)) + "}"), text)
    text = re.sub(r"\[@([\w:,\- ]+)\]",
                  lambda m: hold(r"\cite{" + ",".join(k.strip() for k in m.group(1).split(",")) + "}"),
                  text)
    text = re.sub(r"\$([^$]+)\$", lambda m: hold("$" + m.group(1) + "$"), text)
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\\emph{\1}", text)
    return restore_placeholders(text, held)


def restore_placeholders(text, held):
    """Repeat until stable: a held value may itself contain a placeholder."""
    for _ in range(5):
        new = re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)
        if new == text:
            break
        text = new
    return text


def table(lines):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    align_row = rows[1]
    spec = "".join("r" if set(c) <= set("-:") and c.endswith(":") else
                   ("l" if set(c) <= set("-:") else "c") for c in align_row)
    body = [r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{" + spec + "}", r"\toprule",
            " & ".join(inline(c) for c in rows[0]) + r" \\", r"\midrule"]
    for row in rows[2:]:
        body.append(" & ".join(inline(c) for c in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}}"]
    return body


def convert(markdown):
    out, i = [], 0
    lines = markdown.splitlines()
    in_abstract = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            out += [r"\begin{verbatim}"]
            while i < len(lines) and not lines[i].strip().startswith("```"):
                out.append(lines[i])
                i += 1
            out += [r"\end{verbatim}"]
            i += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            caption = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            caption = re.sub(r"^\*(.+)\*$", r"\1", caption)
            caption = re.sub(r"^Figure\s+\d+\.\s*", "", caption)   # LaTeX numbers figures
            out += [r"\begin{figure}[t]", r"\centering",
                    r"\includegraphics[width=0.92\linewidth]{" + str(Path(match.group(2)).with_suffix("")) + "}",
                    r"\caption{" + inline(caption) + "}", r"\end{figure}"]
            i += 1
            while i < len(lines) and (not lines[i].strip() or
                                      re.match(r"^\*[^*].*\*$", lines[i].strip())):
                i += 1
            continue

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out += table(block)
            continue

        if stripped == "---":
            i += 1
            continue

        match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if match:
            level, title = len(match.group(1)), match.group(2).strip()
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)          # section numbers come from LaTeX
            if level == 1:
                i += 1                                       # title lives in main.tex
                continue
            if in_abstract and not (level == 2 and title.lower() == "abstract"):
                out.append(r"\end{abstract}")
                in_abstract = False
            if level == 2 and title.lower() == "abstract":
                out.append(r"\begin{abstract}")
                in_abstract = True
                i += 1
                continue
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            out.append("\\" + cmd + "{" + inline(title) + "}")
            i += 1
            continue

        if stripped.startswith("- "):
            out.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append(r"\item " + inline(lines[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            continue

        if re.match(r"^\d+\.\s", stripped):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                out.append(r"\item " + inline(re.sub(r"^\d+\.\s", "", lines[i].strip())))
                i += 1
            out.append(r"\end{enumerate}")
            continue

        if not stripped:
            i += 1
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|[-|]|\d+\.\s|```|!\[)", lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        out.append(inline(" ".join(paragraph)))
        out.append("")

    if in_abstract:
        out.append(r"\end{abstract}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper/DRAFT.md")
    ap.add_argument("--body", default="paper/latex/body.tex")
    ap.add_argument("--supplement", default="paper/latex/supplement_body.tex")
    args = ap.parse_args()
    text = Path(args.input).read_text()
    # Appendices go to the supplementary document: CVPR counts pages up to the
    # references, and the appendices are reproduction detail rather than claims.
    split = text.find("\n## Appendix")
    if split != -1:
        Path(args.supplement).write_text(convert(text[split:]) + "\n")
        text = text[:split]
    body = convert(text)
    # The abstract heading is consumed above, so close it before the first section.
    body = body.replace(r"\end{abstract}" + "\n" + r"\section{", r"\end{abstract}" + "\n\n" + r"\section{")
    Path(args.body).write_text(body + "\n")
    print(f"wrote {args.body} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
