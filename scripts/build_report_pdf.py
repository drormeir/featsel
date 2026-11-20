"""Build the final or the detailed report as one PDF.

The final report is one file, final_report/README.md. The detailed report
is, in order: abstract, chapters 1 to 10, references, appendices A to E.
Title pages and contents come first. Both use detailed_report/report.css.

The title page holds personal details, so it is read from
references/local/title.md and the PDF is written there too. Git ignores
that folder. In title.md, a line `* * *` starts the next page.

    .venv/bin/python scripts/build_report_pdf.py                     # the final report
    .venv/bin/python scripts/build_report_pdf.py --report detailed
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / 'references' / 'local'
CSS_FILE = ROOT / 'detailed_report' / 'report.css'
REPO_URL = 'https://github.com/drormeir/featsel/blob/main/'
TITLE = 'Choosing Few Genes from Many: Feature Selection for High-Dimensional Data'

DETAILED = [
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
# Each report: its folder and its files, in order.
REPORTS = {
    'final': ('final_report', ['README.md']),
    'detailed': ('detailed_report', DETAILED),
}

LINK = re.compile(r'\]\(([^)\s]+)\)')


def chapter_id(name: str) -> str:
    return 'chap-' + Path(name).stem


def link_target(target: str, folder: str, files: list[str]) -> str:
    """Point links inside the PDF, and links to the repo at GitHub."""
    if target.startswith(('http', '#')) or target.endswith('.png'):
        return target
    if target.startswith('../'):
        return REPO_URL + target[3:]
    name, _, anchor = target.partition('#')
    if name in files:
        return '#' + (anchor or chapter_id(name))
    return f'{REPO_URL}{folder}/{target}'


def chapter(folder: str, name: str, files: list[str]) -> str:
    lines = []
    titled = False
    for line in (ROOT / folder / name).read_text().splitlines():
        if line.startswith('[Back to contents]('):
            continue
        if not titled and line.startswith('# '):
            line += f' {{#{chapter_id(name)}}}'
            titled = True
        lines.append(line)
    text = '\n'.join(lines)
    return LINK.sub(lambda m: f']({link_target(m.group(1), folder, files)})', text)


def pandoc(args: list[str], text: str) -> str:
    return subprocess.run(
        ['pandoc', '-f', 'markdown+gfm_auto_identifiers-implicit_figures', *args],
        input=text, stdout=subprocess.PIPE, text=True, check=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report', choices=REPORTS, default='final',
                        help='which report (default: %(default)s)')
    parser.add_argument('--title', type=Path, default=LOCAL / 'title.md',
                        help='title and supervisor pages (default: %(default)s)')
    parser.add_argument('--out', type=Path,
                        help='the PDF (default: references/local/<report>_report.pdf)')
    args = parser.parse_args()
    if not args.title.is_file():
        sys.exit(f'Title page not found: {args.title}\n'
                 'Write it there. Keep it out of git: references/local/ is ignored.')
    folder, files = REPORTS[args.report]
    out = args.out or LOCAL / f'{args.report}_report.pdf'

    title_html = pandoc(['-t', 'html5'], args.title.read_text())
    body = '\n\n'.join(chapter(folder, name, files) for name in files)
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

    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT / folder)).write_pdf(
        out, stylesheets=[CSS(filename=CSS_FILE)])
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
