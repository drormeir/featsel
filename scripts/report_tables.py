"""
Fill the generated tables of the report from the results files.

docs/report/00_report_outline.md: every number in the report comes from a
results file, never retyped by hand. A chapter marks each generated table
with a pair of lines:

    <!-- generated: scanb_macro_f1 -->
    <!-- end generated -->

and this script replaces what lies between them with the table named in
TABLES.

Usage:
    python scripts/report_tables.py docs/report/05_implementation.md docs/report/06_results.md
"""

import argparse
import re
from functools import cache
from pathlib import Path

import pandas as pd
from summarize_comparison import _markdown, gain_table, macro_f1_table, own_count_table

ROOT = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r'(<!-- generated: (\w+) -->\n).*?(<!-- end generated -->)', re.DOTALL)


@cache
def _results(name):
    return pd.read_csv(ROOT / 'results' / f'{name}.csv')


def _speed():
    """The speed benchmark: its setting, then seconds and speed-up of each run."""
    speed = _results('speed_benchmark')
    return f'{speed.setting[0]}\n\n' + _markdown(speed.drop(columns='setting').set_index('run'))


COMPARISON = {'macro_f1': macro_f1_table, 'gain': gain_table, 'own_count': own_count_table}


def build(name):
    """
    The table called name: 'speed', or '<dataset>_<table>' with a table of
    COMPARISON built from results/comparison_<dataset>.csv.
    """
    if name == 'speed':
        return _speed()
    dataset, table = name.split('_', 1)
    return COMPARISON[table](_results(f'comparison_{dataset}'))


def fill(text):
    """The text with every generated block rebuilt from its results file."""
    return BLOCK.sub(lambda m: f'{m.group(1)}{build(m.group(2))}\n{m.group(3)}', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapters', nargs='+', help='Markdown files to update in place')
    args = parser.parse_args()
    for chapter in map(Path, args.chapters):
        chapter.write_text(fill(chapter.read_text()))


if __name__ == '__main__':
    main()
