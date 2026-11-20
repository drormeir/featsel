"""Build the final report as one PDF.

Order: title pages, contents, abstract, chapters 1 to 10, references,
appendices A to E. The style is in detailed_report/report.css.

The title page holds personal details, so it is read from
references/local/title.md and the PDF is written there too. Git ignores
that folder. In title.md, a line `* * *` starts the next page.

    .venv/bin/python scripts/build_report_pdf.py
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'detailed_report'
LOCAL = ROOT / 'references' / 'local'
REPO_URL = 'https://github.com/drormeir/featsel/blob/main/'
TITLE = 'Choosing Few Genes from Many: Feature Selection for High-Dimensional Data'

FILES = [
    'abstract.md',
    '01_the_problem.md',
    '02_the_data.md',
    '03_how_to_measure.md',
    '04_the_methods.md',
    '05_the_software.md',
    '06_making_it_fast.md',
    '07_results_scanb.md',
    '08_results_images.md',
    '09_conclusions.md',
    '10_open_work.md',
    'references.md',
    'appendix_a_reproduce.md',
    'appendix_b_data_details.md',
    'appendix_c_comparison_tables.md',
    'appendix_d_engineering.md',
    'appendix_e_glossary.md',
]

LINK = re.compile(r'\]\(([^)\s]+)\)')


def chapter_id(name: str) -> str:
    return 'chap-' + Path(name).stem


def link_target(target: str) -> str:
    """Point links inside the PDF, and links to the repo at GitHub."""
    if target.startswith(('http', '#')) or target.endswith('.png'):
        return target
    if target.startswith('../'):
        return REPO_URL + target[3:]
    name, _, anchor = target.partition('#')
    if name in FILES:
        return '#' + (anchor or chapter_id(name))
    return REPO_URL + 'detailed_report/' + target


def chapter(name: str) -> str:
    lines = []
    titled = False
    for line in (REPORT / name).read_text().splitlines():
        if line.startswith('[Back to contents]('):
            continue
        if not titled and line.startswith('# '):
            line += f' {{#{chapter_id(name)}}}'
            titled = True
        lines.append(line)
    text = '\n'.join(lines)
    return LINK.sub(lambda m: f']({link_target(m.group(1))})', text)


def pandoc(args: list[str], text: str) -> str:
    return subprocess.run(
        ['pandoc', '-f', 'markdown+gfm_auto_identifiers', *args],
        input=text, stdout=subprocess.PIPE, text=True, check=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--title', type=Path, default=LOCAL / 'title.md',
                        help='title and supervisor pages (default: %(default)s)')
    parser.add_argument('--out', type=Path, default=LOCAL / 'report.pdf',
                        help='the PDF (default: %(default)s)')
    args = parser.parse_args()
    if not args.title.is_file():
        sys.exit(f'Title page not found: {args.title}\n'
                 'Write it there. Keep it out of git: references/local/ is ignored.')

    title_html = pandoc(['-t', 'html5'], args.title.read_text())
    body = '\n\n'.join(chapter(name) for name in FILES)
    with tempfile.TemporaryDirectory() as tmp:
        before = Path(tmp) / 'title.html'
        before.write_text(f'<div class="title-page">\n{title_html}</div>\n')
        html = pandoc([
            # A wide --columns stops pandoc fixing the table column widths.
            '-t', 'html5', '--standalone', '--section-divs', '--columns=10000',
            '--toc', '--toc-depth=2',
            '--metadata', 'toc-title=Contents',
            '--metadata', f'pagetitle={TITLE}',
            '--variable', 'document-css=false',
            '--include-before-body', str(before),
        ], body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(REPORT)).write_pdf(
        args.out, stylesheets=[CSS(filename=REPORT / 'report.css')])
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
