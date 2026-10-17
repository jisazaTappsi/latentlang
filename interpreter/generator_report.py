"""Measure the training-data generator on REPORT_PROGRAMS programs (util.py): how many come out
valid, and how much of the language in basic.py their samples exercise, as line and branch
coverage and as AST features. Run it after changing the interpreter: new code the samples never
run shows up as not covered.

Run: python -m interpreter.generator_report
"""
import contextlib
import io
import json
import os
import random
import tempfile
import time

import coverage

from . import basic
from . import data_generator as dg
from . import tokens
from .util import MAX_INTERPRETER_STEPS, REPORT_PROGRAMS

# The classes a training program can exercise; the model and REPL entry points are left out.
LANGUAGE_CLASSES = ['Lexer', 'Parser', 'AstTextReader', 'Interpreter', 'Number', 'String', 'List',
                    'BaseFunction', 'Function', 'BuiltInFunction']
# Built-ins no generated program may call: they print, read input, clear the screen or load files.
NOT_GENERATED = {f'BuiltInFunction.execute_{name}' for name in ['print', 'input', 'input_int', 'clear', 'run']}


def parse(text):
    token_list, _ = basic.Lexer('<report>', text).make_tokens()
    return basic.Parser(token_list).parse().node


def replay(rows):
    """Run each kept program and rebuild each of its AST targets, as training and inference do."""
    for row in rows:
        context = dg.new_context()
        basic.Interpreter(max_steps=MAX_INTERPRETER_STEPS).visit(parse(row['text']), context)
        for line in row['ast_text'].split('\n'):
            if line:
                basic.AstTextReader.get_tree_from_string(line.removeprefix(tokens.SOF).removesuffix(tokens.EOF))


def coverage_by_function(rows):
    """Coverage of basic.py while replaying `rows`: function name -> coverage.py summary."""
    dg._get_compiled_func_templates()  # Compile templates now, so their parsing isn't counted.
    cov = coverage.Coverage(branch=True, include=[basic.__file__], data_file=None)
    cov.start()
    try:
        replay(rows)
    finally:
        cov.stop()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        path = os.path.join(directory, 'coverage.json')
        cov.json_report(outfile=path)
        with open(path) as file:
            report = json.load(file)
    return {name: function['summary'] for name, function in next(iter(report['files'].values()))['functions'].items()}


def measure():
    random.seed(0)
    dg.STATS.clear()
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # build_samples prints progress.
        rows, _ = dg.build_samples(REPORT_PROGRAMS)
    seconds = time.perf_counter() - start
    return {
        'rows': rows,
        'seconds': seconds,
        'statements': sum(len(row['x_in']) for row in rows),
        'functions': coverage_by_function(rows),
        'features': set().union(*(dg.ast_features(parse(row['text'])) for row in rows)),
    }


def language_functions(functions, cls):
    return {name: summary for name, summary in functions.items()
            if name.startswith(f'{cls}.') and name not in NOT_GENERATED}


def percent(covered, total):
    return f'{covered / total * 100:5.1f}%' if total else '    - '


def coverage_cell(summaries):
    lines = percent(sum(s['covered_lines'] for s in summaries), sum(s['num_statements'] for s in summaries))
    branches = percent(sum(s['covered_branches'] for s in summaries), sum(s['num_branches'] for s in summaries))
    return f'{lines} / {branches}'


def print_report(result):
    rows = result['rows']
    print(f'\nValid programs: {len(rows)} of {REPORT_PROGRAMS} ({len(rows) / REPORT_PROGRAMS * 100:.1f}%), '
          f'{result["statements"]} statements, {result["seconds"]:.1f}s')
    dg.print_stats()

    print('\nCoverage of the language in basic.py, lines / branches. Error-reporting branches stay')
    print('uncovered by design: every sample is a valid program.\n')
    for cls in [*LANGUAGE_CLASSES, 'total']:
        classes = LANGUAGE_CLASSES if cls == 'total' else [cls]
        summaries = [s for c in classes for s in language_functions(result['functions'], c).values()]
        print(f'  {cls:20}{coverage_cell(summaries)}')

    missing = sorted(dg.LANGUAGE_FEATURES - result['features'])
    print(f'\nAST features never generated ({len(missing)} of {len(dg.LANGUAGE_FEATURES)}): '
          f'{", ".join(missing) or "none"}')

    print('\nFunctions the samples never run in full (lines run / total, branches run / total):')
    for cls in LANGUAGE_CLASSES:
        for name, s in language_functions(result['functions'], cls).items():
            if s['missing_lines'] or s['missing_branches']:
                print(f'  {name:42} {s["covered_lines"]:>3}/{s["num_statements"]:<3} {s["covered_branches"]:>3}/{s["num_branches"]}')


def main():
    print_report(measure())


if __name__ == '__main__':
    main()
