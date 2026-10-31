"""
Fill the generated tables of the report from the runner's results.

docs/report/00_report_outline.md: every number in the report comes from a
results file, never retyped by hand. A chapter marks each generated table
with a pair of lines:

    <!-- generated: scanb_macro_f1 -->
    <!-- end generated -->

and this script replaces what lies between them. Table names are
<dataset>_<table>, with tables macro_f1, gain and own_count, built by
scripts/summarize_comparison.py from results/comparison_<dataset>.csv.

Usage:
    python scripts/report_tables.py docs/report/06_results.md
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from summarize_comparison import gain_table, macro_f1_table, own_count_table

ROOT = Path(__file__).resolve().parent.parent
TABLES = {'macro_f1': macro_f1_table, 'gain': gain_table, 'own_count': own_count_table}
BLOCK = re.compile(r'(<!-- generated: (\w+?)_(macro_f1|gain|own_count) -->\n).*?'
                   r'(<!-- end generated -->)', re.DOTALL)


def fill(text):
    """The text with every generated block rebuilt from its results file."""
    results = {}

    def table(match):
        dataset, name = match.group(2), match.group(3)
        if dataset not in results:
            results[dataset] = pd.read_csv(ROOT / 'results' / f'comparison_{dataset}.csv')
        return f'{match.group(1)}{TABLES[name](results[dataset])}\n{match.group(4)}'

    return BLOCK.sub(table, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapters', nargs='+', help='Markdown files to update in place')
    args = parser.parse_args()
    for chapter in map(Path, args.chapters):
        chapter.write_text(fill(chapter.read_text()))


if __name__ == '__main__':
    main()
