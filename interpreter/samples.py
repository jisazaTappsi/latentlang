import statistics
import pandas as pd

from . import data
from . import basic
from .util import *
from .basic import AstTextReader
from .data_generator import print_program
from .train_module import CrossAttentionTransformer

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


def get_sample_val_data(num):
    df = pd.read_pickle(DATASET_NAME)

    # Choose samples of validation data
    val_df = data.get_val_data(df)
    random_val_ids = torch.randint(len(val_df), (num,))
    random_val_df = val_df.iloc[random_val_ids]

    return random_val_df


def sample_decode(my_data, merges):
    return CrossAttentionTransformer.fix_unmatched_parenthesis(data.decode(my_data, merges))


def symbol_tables_equal(pred_symbols, target_symbols):
    """Compare two symbol-table dicts (name -> basic.Value): same names, each holding an equal
    value (see `values_equal`)."""
    return pred_symbols.keys() == target_symbols.keys() and \
        all(values_equal(pred_symbols[name], target_symbols[name]) for name in pred_symbols)


def values_equal(p, t):
    """Whether two runs left the same value. They build distinct objects, so this compares what
    they hold: numbers within tolerance, lists element by element, user functions by name,
    parameters, body and return form, anything else (strings, built-ins) by its printed form.

    A function's printed form is only `<function f>`: compared by that, a predicted `f` with a
    wrong body would count as correct."""
    if type(p) != type(t):
        return False
    if isinstance(p, basic.Number):
        try:
            return bool(torch.allclose(torch.tensor(float(p.value)), torch.tensor(float(t.value))))
        except (TypeError, ValueError, OverflowError):
            return p.value == t.value
    if isinstance(p, basic.List):
        return len(p.elements) == len(t.elements) and \
            all(values_equal(a, b) for a, b in zip(p.elements, t.elements))
    if isinstance(p, basic.Function):
        return (p.name, p.arg_names, p.should_auto_return, repr(p.body_node)) == \
            (t.name, t.arg_names, t.should_auto_return, repr(t.body_node))
    return repr(p) == repr(t)


def run(num_samples=100):

    # Load data and merges
    val_samples = get_sample_val_data(num=num_samples)
    lex_merges, ast_merges = data.get_merges()

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()

    tree_scores_per_sample = []
    computation_scores_per_sample = []
    for idx, sample in val_samples.iterrows():
        predicted_ctx = basic.Context('<program>')
        predicted_ctx.symbol_table = basic.get_symbol_table()
        target_ctx = basic.Context('<program>')
        target_ctx.symbol_table = basic.get_symbol_table()
        program_statements = sample.text.splitlines()
        sample_run_ok = True
        statement_ast_matches = []

        print(f'{idx}:')
        print_program(program_statements)
        for x_in, x_out in zip(sample.x_in, sample.x_out):
            predicted_ast_text = model.inference(x_in, ast_merges)
            print(f'P: {predicted_ast_text}')
            target_ast_text = sample_decode(x_out, ast_merges)
            print(f"T: {target_ast_text}\n")

            statement_ast_matches.append(predicted_ast_text == target_ast_text)

            try:
                predicted_ast = AstTextReader.get_tree_from_string(predicted_ast_text)
                predicted_res = basic.Interpreter().visit(predicted_ast, predicted_ctx)
                if predicted_res.error:
                    sample_run_ok = False
                    break
            except Exception as e:
                print(f'building/executing predicted AST gets: {e}, continuing...\n')
                sample_run_ok = False
                break

            try:
                target_ast = AstTextReader.get_tree_from_string(target_ast_text)
                target_res = basic.Interpreter().visit(target_ast, target_ctx)
                if target_res.error:
                    sample_run_ok = False
                    break
            except Exception as e:
                print(f'building/executing target AST gets: {e}, continuing...\n')
                sample_run_ok = False
                break

        # Per-sample tree: 1 iff every statement had correct AST AND the sample ran
        # (this way, any parse/interpret error forces tree=0, and if tree=1 we know
        # both predicted and target programs were actually executable).
        tree_ok = sample_run_ok and all(statement_ast_matches)
        tree_scores_per_sample.append(int(tree_ok))

        if sample_run_ok:
            tables_match = symbol_tables_equal(predicted_ctx.symbol_table.symbols, target_ctx.symbol_table.symbols)
            computation_scores_per_sample.append(int(tables_match))
        else:
            tables_match = False
            computation_scores_per_sample.append(0)

        print(f'Trees are equal: {tree_ok} | predicted: {predicted_ctx.symbol_table.symbols} target: {target_ctx.symbol_table.symbols} | computation is equal: {tables_match}\n')

    n_samples = len(computation_scores_per_sample)
    tree_pct = statistics.mean(tree_scores_per_sample) * 100 if tree_scores_per_sample else 0
    computation_pct = statistics.mean(computation_scores_per_sample) * 100 if computation_scores_per_sample else 0
    print(f'Per-sample (n={n_samples}): Tree {round(tree_pct, 3)}%  |  Computation {round(computation_pct, 3)}%')
    return computation_pct


if __name__ == '__main__':
    run(num_samples=100)
