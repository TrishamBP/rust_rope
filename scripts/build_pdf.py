"""Builds ROPE_Engineering.pdf from docs/ROPE_Engineering.md (reproducible).

Pipeline:
  1. scripts/make_diagrams.py writes docs/diagrams/*.svg.
  2. Markdown -> HTML (python-markdown + Pygments). Figures, tables and
     listings get numbers; SVG diagrams are inlined (vector output).
  3. Headless Chrome prints the HTML to PDF (CSS page-margin boxes give the
     header, footer and page numbers).
  4. PyMuPDF finds the page of every heading, figure and table; the script
     writes these numbers into the table of contents and the lists, and
     prints again until the numbers are stable.
  5. PDF bookmarks (outline) and metadata are added.
  6. Quality checks (see check()).

Run:  scripts/.venv/Scripts/python scripts/build_pdf.py
Needs: markdown, pygments, pymupdf (scripts/requirements.txt) and Chrome.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import markdown
import pymupdf
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "ROPE_Engineering.md"
HTML_OUT = ROOT / "target" / "ROPE_Engineering.html"
PDF_OUT = ROOT / "ROPE_Engineering.pdf"
DIAGRAMS = ROOT / "docs" / "diagrams"
VERSION = "1.0"
BODY_MARKER = "BODYSTARTMARKER"
BUILD_DATE = date.today().isoformat()
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CSS = r"""
@page {
  size: A4;
  margin: 20mm 17mm 18mm 17mm;
  @top-left { content: "Rust RoPE Engineering Guide"; font: 8.5pt "Segoe UI", sans-serif; color: #6b7280; }
  @top-right { content: "Version VERSION · BUILD_DATE"; font: 8.5pt "Segoe UI", sans-serif; color: #6b7280; }
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font: 8.5pt "Segoe UI", sans-serif; color: #374151; }
}
@page :first { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }
html { font-family: "Cambria", "Georgia", serif; font-size: 10.3pt; line-height: 1.42; color: #111827; }
body { margin: 0; }
h1, h2, h3, h4 { font-family: "Segoe UI", "Arial", sans-serif; line-height: 1.2; break-after: avoid; }
h1.part { break-before: page; font-size: 25pt; color: #1e3a8a; margin-top: 35mm; border-bottom: 3px solid #1e3a8a; padding-bottom: 6pt; }
h2 { break-before: auto; font-size: 18pt; color: #1e3a8a; margin: 22pt 0 10pt 0; border-bottom: 1.5px solid #93c5fd; padding-bottom: 4pt; }
h2.nobreak { break-before: auto; }
h3 { font-size: 13pt; color: #1f2937; margin: 16pt 0 5pt 0; }
h4 { font-size: 11pt; color: #374151; margin: 12pt 0 4pt 0; }
p { margin: 5pt 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 4pt 0 6pt 0; padding-left: 20pt; }
li { margin: 1.5pt 0; }
code { font-family: "Cascadia Mono", "Consolas", monospace; font-size: 8.9pt; background: #f3f4f6; padding: 0 2px; border-radius: 2px; }
pre { font-family: "Cascadia Mono", "Consolas", "Segoe UI Symbol", monospace; font-size: 8.5pt; line-height: 1.32;
      white-space: pre-wrap; word-break: break-word; margin: 0; }
.codehilite { background: #f8fafc; border: 1px solid #cbd5e1; border-left: 3px solid #2563eb; border-radius: 4px;
              padding: 6pt 8pt; margin: 6pt 0 8pt 0; break-inside: avoid; }
.codehilite pre code, .eq pre { background: none; padding: 0; font-size: 8.5pt; }
.eq { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 4px; padding: 6pt 10pt; margin: 6pt 0 8pt 0; break-inside: avoid; }
.eq pre { font-family: "Cascadia Mono", "Consolas", "Cambria Math", "Segoe UI Symbol", monospace; font-size: 9.6pt; line-height: 1.45; }
.listing { break-inside: avoid; margin: 8pt 0; }
.listing-cap { font-family: "Segoe UI", sans-serif; font-size: 8.8pt; color: #1e3a8a; font-weight: 600; margin-bottom: -3pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0; font-size: 8.9pt; break-inside: avoid; }
caption { caption-side: top; text-align: left; font-family: "Segoe UI", sans-serif; font-size: 8.8pt; font-weight: 600; color: #1e3a8a; padding-bottom: 3pt; }
th { background: #e0e7ff; text-align: left; }
th, td { border: 1px solid #c7d2fe; padding: 2.5pt 4pt; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
figure { margin: 10pt 0; text-align: center; break-inside: avoid; }
figure svg { max-width: 100%; height: auto; }
figcaption { font-family: "Segoe UI", sans-serif; font-size: 8.8pt; color: #374151; margin-top: 3pt; text-align: center; }
figcaption b { color: #1e3a8a; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 4px solid #f59e0b; background: #fffbeb; break-inside: avoid; }
.ref { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 4px; padding: 4pt 8pt; margin: 6pt 0; font-size: 9.4pt; break-inside: avoid; }
.title { text-align: center; padding-top: 55mm; break-after: page; }
.title h1 { font-size: 34pt; color: #1e3a8a; margin: 0; border: none; }
.title .sub { font-size: 15pt; color: #374151; margin-top: 10pt; font-family: "Segoe UI", sans-serif; }
.title .meta { margin-top: 60mm; font-size: 11pt; color: #374151; font-family: "Segoe UI", sans-serif; line-height: 1.7; }
.front h2 { break-before: page; }
.toc { font-family: "Segoe UI", sans-serif; font-size: 9.6pt; }
.toc a { color: #111827; text-decoration: none; }
.toc .row { display: flex; align-items: baseline; margin: 1.2pt 0; }
.toc .row .t { flex: 0 1 auto; }
.toc .row .dots { flex: 1 1 auto; border-bottom: 1px dotted #9ca3af; margin: 0 4pt; transform: translateY(-3pt); }
.toc .row .p { flex: 0 0 auto; min-width: 18pt; text-align: right; }
.toc .l1 { font-weight: 700; color: #1e3a8a; margin-top: 7pt; font-size: 10.4pt; }
.toc .l2 { font-weight: 600; padding-left: 10pt; margin-top: 3pt; }
.toc .l3 { padding-left: 24pt; color: #374151; font-size: 9pt; }
a { color: #1d4ed8; }
.partintro { font-family: "Segoe UI", sans-serif; margin-top: 14pt; font-size: 11pt; break-after: page; }
.partintro p { text-align: left; color: #374151; }
.partintro ol { list-style: none; padding-left: 0; }
.partintro li { margin: 5pt 0; padding: 5pt 10pt; border-left: 4px solid #93c5fd; background: #f8fafc; }
.partintro a { color: #1e3a8a; text-decoration: none; }
.partintro .sec { color: #6b7280; font-size: 9pt; }
"""


def slug(text: str, used: dict) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s).strip("-") or "section"
    n = used.get(s, 0)
    used[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def preprocess(md: str) -> str:
    """```math blocks -> raw HTML equation boxes."""
    def eq(m):
        return f'\n<div class="eq"><pre>{html.escape(m.group(1))}</pre></div>\n'
    return re.sub(r"^```math\n(.*?)^```\s*$", eq, md, flags=re.S | re.M)


def inline_svg(path: str) -> str:
    data = (ROOT / "docs" / path).read_text(encoding="utf-8")
    return data


def convert(md_text: str):
    body = markdown.markdown(
        preprocess(md_text),
        extensions=["tables", "fenced_code", "codehilite", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"}},
    )
    used: dict = {}
    headings = []  # (level, text, id)

    def head(m):
        level, attrs, inner = int(m.group(1)), m.group(2) or "", m.group(3)
        text = re.sub(r"<[^>]+>", "", inner)
        text = html.unescape(text).strip()
        hid = slug(text, used)
        cls = ""
        if level == 1:
            cls = ' class="part"'
        headings.append((level, text, hid))
        return f'<h{level} id="{hid}"{cls}{attrs}>{inner}</h{level}>'

    body = re.sub(r"<h([1-3])([^>]*)>(.*?)</h\1>", head, body, flags=re.S)

    # A short overview of the chapters (and their sections) on each part page.
    overview, current = {}, None
    for level, text, hid in headings:
        if level == 1:
            current = hid
            overview[current] = []
        elif level == 2 and current:
            overview[current].append([text, hid, []])
        elif level == 3 and current and overview[current]:
            overview[current][-1][2].append(text)
    for part_id, chapters in overview.items():
        items = "".join(
            f'<li><a href="#{cid}"><b>{html.escape(t)}</b></a><br><span class="sec">'
            f'{html.escape(" · ".join(secs))}</span></li>' for t, cid, secs in chapters)
        intro = f'<div class="partintro"><p>This part contains:</p><ol>{items}</ol></div>'
        pattern = r'(<h1 id="' + re.escape(part_id) + r'"[^>]*>.*?</h1>)'
        body = re.sub(pattern, lambda m: m.group(1) + intro, body, count=1, flags=re.S)

    figures, tables, listings = [], [], []

    def fig(m):
        alt, src, title = html.unescape(m.group(1)), m.group(2), html.unescape(m.group(3) or "")
        n = len(figures) + 1
        figures.append((n, alt, f"fig-{n}"))
        cap = f"<b>Figure {n}.</b> {html.escape(alt)}." + (f" {html.escape(title)}" if title else "")
        return f'<figure id="fig-{n}">{inline_svg(src)}<figcaption>{cap}</figcaption></figure>'

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.svg)"(?: title="([^"]*)")?\s*/?></p>', fig, body)

    def tab(m):
        cap = m.group(1).strip()
        n = len(tables) + 1
        text = re.sub(r"<[^>]+>", "", html.unescape(cap))
        tables.append((n, text, f"tab-{n}"))
        return f'<table id="tab-{n}"><caption>Table {n}. {cap}</caption>'

    body = re.sub(r"<p>Table: (.*?)</p>\s*<table>", tab, body, flags=re.S)

    def lst(m):
        cap = m.group(1).strip()
        n = len(listings) + 1
        listings.append((n, re.sub(r"<[^>]+>", "", html.unescape(cap))))
        return f'<div class="listing"><div class="listing-cap">Listing {n}. {cap}</div>{m.group(2)}</div>'

    body = re.sub(r'<p>Listing: (.*?)</p>\s*(<div class="codehilite">.*?</div>)', lst, body, flags=re.S)
    return body, headings, figures, tables, listings


def front_matter(headings, figures, tables, pages: dict) -> str:
    def row(cls, text, target, page):
        return (f'<div class="row {cls}"><a class="t" href="#{target}">{html.escape(text)}</a>'
                f'<span class="dots"></span><span class="p">{page}</span></div>')

    toc = ['<div class="toc">']
    for level, text, hid in headings:
        if text in ("Contents", "List of Figures", "List of Tables"):
            continue
        toc.append(row(f"l{level}", text, hid, pages.get(hid, "—")))
    toc.append("</div>")
    lof = ['<div class="toc">'] + [row("l3", f"Figure {n}. {t}", fid, pages.get(fid, "—")) for n, t, fid in figures] + ["</div>"]
    lot = ['<div class="toc">'] + [row("l3", f"Table {n}. {t}", tid, pages.get(tid, "—")) for n, t, tid in tables] + ["</div>"]
    title = f"""
<div class="title">
  <h1 style="break-before:auto;margin-top:0">Rust RoPE Engineering Guide</h1>
  <div class="sub">Rotary Position Embeddings from first principles:<br>mathematics, Rust implementation, validation and scaling</div>
  <div class="sub" style="font-size:12pt;margin-top:18pt">An engineering textbook for the <code>rust_rope</code> project</div>
  <div class="meta">Version {VERSION} · generated {BUILD_DATE}<br>
  Source: <code>docs/ROPE_Engineering.md</code> · Generator: <code>scripts/build_pdf.py</code><br>
  Project state: Stages 1–8 implemented · Stage 9 (CPU performance) planned</div>
</div>"""
    return (title + '<div class="front"><h2 id="contents" class="nobreak">Contents</h2>' + "".join(toc)
            + '<h2 id="list-of-figures">List of Figures</h2>' + "".join(lof)
            + '<h2 id="list-of-tables">List of Tables</h2>' + "".join(lot) + "</div>")


def render(md_text: str, pages: dict):
    body, headings, figures, tables, listings = convert(md_text)
    css = CSS.replace("VERSION", VERSION).replace("BUILD_DATE", BUILD_DATE)
    style = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Rust RoPE Engineering Guide</title>'
           f"<style>{css}\n{style}\n.codehilite {{ background: #f8fafc; }}</style></head><body>"
           f"{front_matter(headings, figures, tables, pages)}"
           f'{body}</body></html>')
    # The body marker goes directly after the first part heading, so that it
    # is on the first body page (not on the last front-matter page).
    marker = f'<div style="font-size:1pt;line-height:1pt;color:#fff">{BODY_MARKER}</div>'
    end = doc.find("</h1>", doc.find('class="part"'))
    doc = doc[:end + 5] + marker + doc[end + 5:]
    return doc, headings, figures, tables, listings


def chrome() -> Path:
    for c in CHROME_CANDIDATES:
        if c.exists():
            return c
    sys.exit("Chrome or Edge not found")


def print_pdf(html_text: str, out: Path):
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html_text, encoding="utf-8")
    with tempfile.TemporaryDirectory() as profile:
        cmd = [str(chrome()), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", "--generate-pdf-document-outline",
               f"--print-to-pdf={out}", HTML_OUT.as_uri()]
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)


def find_pages(pdf: Path, headings, figures, tables) -> dict:
    doc = pymupdf.open(pdf)
    texts = [" ".join(p.get_text().split()) for p in doc]
    # The body starts at the page with the (invisible) body marker, so that
    # the entries of the table of contents are never matched.
    start = next((i for i, t in enumerate(texts) if BODY_MARKER in t), 0)
    pages = {}
    # Headings: use the outline that Chrome generates from h1–h3, in document
    # order. (A text search would also match the chapter lists on the part
    # pages.)
    norm = lambda t: " ".join(t.split())
    outline = [(norm(title), page) for _, title, page in doc.get_toc(simple=True)]
    pos, last_page = 0, start + 1
    for _, text, hid in headings:
        for k in range(pos, len(outline)):
            if outline[k][0] == norm(text):
                pages[hid] = outline[k][1]
                pos = k + 1
                break
        else:
            # Fallback (Chrome sometimes omits an outline entry): text search
            # from the page of the previous heading, in document order.
            for i in range(last_page - 1, len(texts)):
                if norm(text) in texts[i]:
                    pages[hid] = i + 1
                    break
        last_page = pages.get(hid, last_page)
    targets = [(f, f"Figure {n}.") for n, _, f in figures] + [(t, f"Table {n}.") for n, _, t in tables]
    for target, needle in targets:
        needle = " ".join(needle.split())
        for i in range(start, len(texts)):
            if needle in texts[i]:
                pages[target] = i + 1
                break
    doc.close()
    return pages


def add_outline(pdf: Path, headings, pages):
    doc = pymupdf.open(pdf)
    toc = [[1, "Contents", 2]]
    for level, text, hid in headings:
        if hid in pages:
            toc.append([level, text, pages[hid]])
    doc.set_toc(toc)
    doc.set_metadata({"title": "Rust RoPE Engineering Guide", "subject": "Rotary Position Embeddings in Rust",
                      "keywords": "RoPE, Rust, Transformer, rotary position embedding",
                      "creator": "scripts/build_pdf.py (python-markdown, headless Chrome, PyMuPDF)",
                      "producer": "PyMuPDF"})
    tmp = pdf.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    shutil.move(tmp, pdf)


# --------------------------------------------------------------------------- checks
def check(md_text: str, headings, figures, tables, listings, pages) -> list[str]:
    problems = []
    # 1. File references point to real files.
    for ref in sorted(set(re.findall(r"`((?:src|tests|examples|benches|reference|scripts|docs)/[\w./-]+\.(?:rs|py|md|txt|toml))`", md_text))):
        if not (ROOT / ref).exists() and not ref.startswith("reference/data/"):
            problems.append(f"missing file reference: {ref}")
    # 2. Listings: the file exists, the named functions exist, and every code line exists in the file.
    for m in re.finditer(r"^Listing: `([^`]+)` — (.*?)\n\n```rust\n(.*?)^```", md_text, flags=re.S | re.M):
        path, funcs, code = m.group(1), m.group(2), m.group(3)
        f = ROOT / path
        if not f.exists():
            problems.append(f"listing file missing: {path}")
            continue
        source = f.read_text(encoding="utf-8")
        norm_source = {" ".join(l.split()) for l in source.splitlines()}
        for name in re.findall(r"`([A-Za-z_][\w:]*)`", funcs):
            short = name.split("::")[-1]
            if not re.search(rf"\b(fn|struct|enum|const|impl)\s+{re.escape(short)}\b", source) and short not in source:
                problems.append(f"{path}: name {name} not found")
        for line in code.splitlines():
            n = " ".join(line.split())
            if not n or n in ("// ...", "...") or n.startswith("// …"):
                continue
            if n not in norm_source:
                problems.append(f"{path}: listing line not in source: {line.strip()[:70]}")
    # 3. Every heading, figure and table has a page.
    for _, text, hid in headings:
        if hid not in pages and text not in ("Contents", "List of Figures", "List of Tables"):
            problems.append(f"heading without page: {text}")
    for n, _, fid in figures:
        if fid not in pages:
            problems.append(f"figure {n} without page")
    for n, _, tid in tables:
        if tid not in pages:
            problems.append(f"table {n} without page")
    # 4. Text extraction: replacement characters.
    doc = pymupdf.open(PDF_OUT)
    full = "".join(p.get_text() for p in doc)
    if "\ufffd" in full:
        problems.append("replacement character U+FFFD in PDF text")
    doc.close()
    return problems


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_diagrams.py")], check=True)
    md_text = SRC.read_text(encoding="utf-8")
    pages: dict = {}
    for attempt in range(4):
        html_text, headings, figures, tables, listings = render(md_text, pages)
        print_pdf(html_text, PDF_OUT)
        new_pages = find_pages(PDF_OUT, headings, figures, tables)
        print(f"pass {attempt + 1}: {len(new_pages)} targets located")
        if new_pages == pages:
            break
        pages = new_pages
    add_outline(PDF_OUT, headings, pages)
    doc = pymupdf.open(PDF_OUT)
    n_pages = doc.page_count
    doc.close()
    problems = check(md_text, headings, figures, tables, listings, pages)
    chapters = sum(1 for lvl, t, _ in headings if lvl == 2 and t.startswith("Chapter"))
    print(f"PDF: {PDF_OUT} ({PDF_OUT.stat().st_size // 1024} KiB, {n_pages} pages)")
    print(f"parts: {sum(1 for l, _, _ in headings if l == 1)}, chapters: {chapters}, "
          f"figures: {len(figures)}, tables: {len(tables)}, listings: {len(listings)}")
    if problems:
        print(f"{len(problems)} PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
