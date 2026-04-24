import re
import math
import copy
import random
import pandas as pd
from typing import Optional


import data
import basic
import tokens
from util import *


def log_scale_int(low: int, high: int, base: float = math.e) -> int:
    """Sample an integer in [low, high] with log scale so smaller numbers are overrepresented.
    low must be >= 1 (use separate logic to include 0)."""
    if low >= high:
        return low
    # u uniform in [0,1] -> value log-uniform in [low, high], then round
    u = random.random()
    log_val = math.log(low, base) + u * (math.log(high, base) - math.log(low, base))
    return min(high, max(low, int(round(math.exp(log_val)))))


def int_part():
    return 0 if random.random() < 0.1 else log_scale_int(1, 2**31, base=6)


def generate_number():
    """Generate a random integer or float (log scale: smaller numbers more likely)."""
    # Include 0 with small probability; otherwise log-scale in [1, 9999]
    if random.random() < 0.3:  # 30% chance of float
        integer_part = int_part()
        decimal_part = random.randint(0, 99)  # 2 digits id drawn from uniform distribution
        return f"{integer_part}.{decimal_part:02d}"
    else:
        return str(int_part())


def generate_factor(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate a factor according to grammar:
    factor : INT|FLOAT|IDENTIFIER
          : (PLUS|MINUS) factor
          : LPAREN expr RPAREN
    """
    if depth >= max_depth:
        # At max depth, only generate numbers (or variable if allowed) to avoid infinite recursion
        if allowed_vars and random.random() < 0.3:
            return random.choice(allowed_vars)
        return generate_number()
    
    choice = random.random()
    
    if allowed_vars and choice < 0.25:  # 25% chance: use a declared variable
        return random.choice(allowed_vars)
    elif choice < 0.5:  # number (or variable if not using vars)
        if allowed_vars and random.random() < 0.4:
            return random.choice(allowed_vars)
        return generate_number()
    elif choice < 0.75:  # unary operator
        op = random.choice(['+', '-'])
        factor = generate_factor(depth + 1, max_depth, allowed_vars)
        return f"{op}{factor}"
    else:  # parentheses
        expr = generate_expr(depth + 1, max_depth, allowed_vars)
        return f"({expr})"


def generate_call(depth, max_depth, allowed_vars):
    """call       : factor (LPAREN (expr (COMMA expr)*)? RPAREN)?
    Functions: name(args). Learned *operators*: infix between calls, e.g. 8 times 9 (template-defined; not builtins)."""
    # Keep recursion bounded: at max depth emit only a simple factor.
    if depth >= max_depth:
        return generate_factor(depth, max_depth, allowed_vars)

    choice = random.random()
    if choice < 0.33:
        return generate_factor(depth, max_depth, allowed_vars)
    elif choice < 0.66:
        name, params, _ = random.choice(FUNC_TEMPLATES)
        args = [generate_expr(depth + 1, max_depth, allowed_vars) for _ in params]
        return f"{name}({','.join(args)})"
    else:
        name, params, _ = random.choice([f for f in FUNC_TEMPLATES if len(f[1]) == 2])
        left = generate_call(depth + 1, max_depth, allowed_vars)
        right = generate_call(depth + 1, max_depth, allowed_vars)
        if name == "over":
            while True:
                try:
                    test = right.replace(tokens.NULL, '0')
                    if int(eval(test)) != 0:
                        break
                except Exception:
                    break
                right = generate_call(depth + 1, max_depth, allowed_vars)
        return f"{left} {name} {right}"


def generate_term(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate a term according to grammar:
    term : factor ((MUL|DIV) factor)*
    """
    if depth >= max_depth:
        return generate_call(depth, max_depth, allowed_vars)

    result = generate_call(depth, max_depth, allowed_vars)
    
    # Randomly add more factors with MUL or DIV operators (often 0 to keep equations small)
    num_ops = 1 if random.random() < 0.3 else 0
    for _ in range(num_ops):

        # Desperate attempt at trying to decrease the division by zeroes...
        op = random.choices(['*', '/'], weights=[0.95, 0.05], k=1)[0]

        while True:  # Don't allow division by zero
            factor = generate_call(depth, max_depth, allowed_vars)
            try:
                factor = factor.replace(tokens.NULL, '0')  # in our interpreter None is a 0
                mini_expr = int(eval(factor))
                if op != '/' or mini_expr != 0:
                    break
            except:
                break

        result = f"{result}{op}{factor}"
    
    return result


def generate_expr(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate an expression according to grammar:
    expr : (term(PLUS|MINUS) term)*
    """
    if allowed_vars is None:
        allowed_vars = []
    if depth >= max_depth:  # At max depth, generate a simple term
        return generate_term(depth, max_depth, allowed_vars)
    
    result = generate_term(depth, max_depth, allowed_vars)
    
    # Randomly add more terms with PLUS or MINUS operators (often 0 to keep equations small)
    num_ops = 1 if random.random() < 0.25 else 0
    for _ in range(num_ops):
        op = random.choice(['+', '-'])
        term = generate_term(depth, max_depth, allowed_vars)
        result = f"{result}{op}{term}"
    
    return result


def gen_arith_expr(allowed_vars):
    """
    Generate a valid arithmetic expression that can be parsed.
    Optionally limit the length of the generated expression.
    """
    max_length = BLOCK_SIZE

    # Start with a reasonable max_depth based on BLOCK_SIZE. Deeper expressions tend to be longer.
    max_depth = 4
    expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=allowed_vars)

    # If expression is too long, regenerate with lower max_depth (stricter cap leaves headroom for comparisons/logic encoding)
    while len(expr) > max_length//5:
        max_depth = max(1, max_depth - 1)
        expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=allowed_vars)

    return expr


# Variable names: letters only, no digits. Exclude names that are keywords.
_VAR_NAME_LETTERS = 'abcdefghijklmnopqrstuvwxyz'  # single letters (no 'v' to avoid 'var')


def _new_var_name(declared: list) -> str:
    """Return a variable name not in declared, not a keyword, and not a template name. Letters only."""
    forbidden = (
        set(declared)
        | set(tokens.KEYWORDS)
        | {n for n, _, _ in FUNC_TEMPLATES}
    )
    # Single-letter names first
    available = [c for c in _VAR_NAME_LETTERS if c not in forbidden]
    if available:
        return random.choice(available)
    # Fallback: two-letter names (letters only)
    for c1 in _VAR_NAME_LETTERS:
        for c2 in _VAR_NAME_LETTERS:
            name = c1 + c2
            if name not in forbidden:
                return name
    # Last resort: longer names (should not happen in practice)
    for length in range(3, 10):
        for _ in range(100):
            name = ''.join(random.choice(_VAR_NAME_LETTERS) for _ in range(length))
            if name not in forbidden:
                return name
    return 'x'


BOOLEAN_LITERALS = [tokens.NULL, tokens.TRUE, tokens.FALSE]


def generate_program_expression(allowed_vars) -> str:
    """
    Generate an expression compatible with the full grammar, including
    arithmetic, comparison operators, logical AND/OR, and the identifiers
    True, False and null.
    """

    # Start from either a boolean-like identifier or a plain arithmetic expression
    if random.random() < 0.3:
        expr = random.choice(BOOLEAN_LITERALS)
    else:
        expr = gen_arith_expr(allowed_vars)

    # Optionally add one or more comparison operations (slightly lower so more fit in BLOCK_SIZE)
    if random.random() < 0.20:
        num_comparisons = random.randint(0, 1)
        for _ in range(num_comparisons):
            op = random.choice(['==', '!=', '<', '>', '<=', '>='])
            right = gen_arith_expr(allowed_vars)
            expr = f"{expr}{op}{right}"

    # Optional leading NOT (low prob to keep equations small)
    if random.random() < 0.15:
        expr = f"{tokens.NOT} {expr}"

    # Optionally chain with AND/OR (often 0 to keep equations small)
    num_logic_ops = 1 if random.random() < 0.2 else 0
    for _ in range(num_logic_ops):
        op = random.choice([tokens.AND, tokens.OR])
        right = gen_arith_expr(allowed_vars)
        # Sometimes turn the right-hand side into a comparison expression as well
        if random.random() < 0.7:
            num_cmp = random.randint(0, 1)
            for _ in range(num_cmp):
                cmp_op = random.choice(['==', '!=', '<', '>', '<=', '>='])
                right = f"{right}{cmp_op}{gen_arith_expr(allowed_vars)}"
        if random.random() < 0.3:
            right = f"{tokens.NOT} {right}"
        expr = f"{expr} {op} {right}"

    return expr


def generate_program_statements(texts) -> list:
    """Generates a short program with valid statements. Each statement is either a variable declaration
    ('var x = expr') or a standalone expression. Expressions may use previously declared variables."""
    declared = copy.copy(BOOLEAN_LITERALS)
    statements = []
    num_statements = random.randint(2, 5)

    for _ in range(num_statements):
        while True:
            if not declared or random.random() < 0.8:
                # Variable declaration: var name = expr
                name = _new_var_name(declared)
                expr = generate_program_expression(declared)
                text = f"{tokens.VAR} {name} = {expr}"
                if text not in texts:
                    declared.append(name)
                    break
            else:
                # Standalone expression (can use declared variables)
                text = generate_program_expression(declared)
                if text not in texts:
                    break

        texts.add(text)
        statements.append(text)

    return statements


# ---------------------------------------------------------------------------
# Template names: function call syntax name(args) and infix syntax left name right (arity-2 only).
# Sorted by name; used for compilation, symbol injection, and AST inlining.
# ---------------------------------------------------------------------------
FUNC_TEMPLATES = [
    ("add", ["x", "y"], "x+y"),
    ("add3", ["a", "b", "c"], "a+b+c"),
    ("add4", ["a", "b", "c", "d"], "a+b+c+d"),
    ("antip", ["a", "b"], "0-a*b"),
    ("apart", ["a", "b"], "a-b"),
    ("avg", ["a", "b"], "(a+b)/2"),
    ("balance", ["a", "b"], "(a+b)/2"),
    ("blend", ["a", "b"], "(a+b)/2"),
    ("boost", ["n"], "n+n"),
    ("brim", ["x"], "x+x"),
    ("bump2", ["n"], "n+2"),
    ("burst", ["n"], "n+n+n+n"),
    ("chime", ["n"], "n+1"),
    ("combo", ["a", "b"], "a*b+a+b"),
    ("crisp", ["x"], "x*x"),
    ("cube", ["x"], "x*x*x"),
    ("cubeln", ["x"], "x*x*x+x"),
    ("dec", ["n"], "n-1"),
    ("dec2", ["n"], "n-2"),
    ("dice", ["a", "b"], "a*b"),
    ("diff", ["x", "y"], "x-y"),
    ("dist", ["a", "b"], "a-b"),
    ("dotpair", ["a", "b"], "a*b"),
    ("double", ["x"], "x+x"),
    ("double2", ["x"], "x*2"),
    ("duplex", ["x"], "x+x"),
    ("edge", ["a", "b"], "a*a+b*b"),
    ("epic", ["a", "b", "c"], "(a+b+c)/3"),
    ("fold4", ["a", "b", "c", "d"], "a+b+c+d"),
    ("fourth", ["x"], "x*x*x*x"),
    ("fuzzy", ["a", "b"], "(a+b)/2"),
    ("geom3", ["a", "b", "c"], "a*b*c"),
    ("glide", ["x"], "x/2"),
    ("grip", ["a", "b"], "a-b"),
    ("halve", ["x"], "x/2"),
    ("halve3", ["x"], "x/3"),
    ("hinge", ["a", "b"], "a*b+a+b"),
    ("inc", ["n"], "n+1"),
    ("invmix", ["a", "b"], "a*(0-b)"),
    ("jolt", ["n"], "n+3"),
    ("jump", ["n"], "n+2"),
    ("layer", ["x"], "x*x+x"),
    ("lift2", ["x"], "x+2"),
    ("merge", ["a", "b"], "a+b"),
    ("mid3", ["a", "b", "c"], "(a+b+c)/3"),
    ("mix2", ["a", "b"], "(a+b)/2"),
    ("mul", ["a", "b"], "a*b"),
    ("mul3", ["a", "b", "c"], "a*b*c"),
    ("neg8", ["x"], "0-x"),
    ("negate", ["x"], "not x"),
    ("nook", ["x"], "0-x-x"),
    ("over", ["a", "b"], "a/b"),
    ("plus", ["a", "b"], "a+b"),
    ("poly2", ["x"], "x*x+x"),
    ("prod", ["x", "y"], "x*y"),
    ("quad", ["x"], "x*x*x*x"),
    ("ramp", ["n"], "n+n+n+n"),
    ("reach", ["x"], "1/x"),
    ("ring", ["a", "b", "c"], "a+b+c"),
    ("scale", ["x", "f"], "x*f"),
    ("skew", ["a", "b", "c"], "a+b*2+c"),
    ("slip", ["a", "b"], "a-b"),
    ("spark", ["x"], "x*3"),
    ("square", ["x"], "x*x"),
    ("sub", ["a", "b"], "a-b"),
    ("sum", ["a", "b"], "a+b"),
    ("sumsq", ["a", "b"], "a*a+b*b"),
    ("surge", ["n"], "n+n+n"),
    ("swath", ["a", "b"], "a*a+b*b"),
    ("sweep", ["a", "b"], "a/b"),
    ("tap", ["x"], "x+x"),
    ("thrice", ["n"], "n+n+n"),
    ("thrice2", ["x"], "x*3"),
    ("tilt", ["a", "b"], "a-b"),
    ("times", ["a", "b"], "a*b"),
    ("times8", ["x"], "x*8"),
    ("triple", ["n"], "n+n+n"),
    ("twin", ["x"], "x+x"),
    ("twist", ["a", "b"], "a-b"),
    ("wedge", ["a", "b"], "a*b-a"),
]


_COMPILED_FUNC_TEMPLATES = None
_COMPILED_OP_TEMPLATES = None


def _get_compiled_func_templates():
    """Compile FUNC_TEMPLATES body expressions into AST nodes once."""
    global _COMPILED_FUNC_TEMPLATES
    if _COMPILED_FUNC_TEMPLATES is not None:
        return _COMPILED_FUNC_TEMPLATES

    compiled = []
    for name, params, body in FUNC_TEMPLATES:
        lexer = basic.Lexer('<func_template>', body)
        body_tokens, error = lexer.make_tokens()
        if error:
            continue
        parser = basic.Parser(body_tokens)
        ast = parser.parse()
        if ast.error:
            continue
        compiled.append((name, params, ast.node))

    _COMPILED_FUNC_TEMPLATES = compiled
    return _COMPILED_FUNC_TEMPLATES


def _get_compiled_op_templates():
    """Compile FUNC_TEMPLATES body expressions into AST nodes (parallel to FUNC_TEMPLATES)."""
    global _COMPILED_OP_TEMPLATES
    if _COMPILED_OP_TEMPLATES is not None:
        return _COMPILED_OP_TEMPLATES

    compiled = []
    for name, params, body in FUNC_TEMPLATES:
        lexer = basic.Lexer('<op_template>', body)
        body_tokens, error = lexer.make_tokens()
        if error:
            continue
        parser = basic.Parser(body_tokens)
        ast = parser.parse()
        if ast.error:
            continue
        compiled.append((name, params, ast.node))

    _COMPILED_OP_TEMPLATES = compiled
    return _COMPILED_OP_TEMPLATES


def _load_template_functions_into_context(context):
    """Inject template bodies for FUNC names and for infix OP names (used at runtime via visit_BinOp + Function)."""
    for name, params, body_node in _get_compiled_func_templates():
        func_value = basic.Function(name, body_node, params).set_context(context).set_pos()
        context.symbol_table.set(name, func_value)
    for name, params, body_node in _get_compiled_op_templates():
        func_value = basic.Function(name, body_node, params).set_context(context).set_pos()
        context.symbol_table.set(name, func_value)


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(text)):
        if text[idx] == '(':
            depth += 1
        elif text[idx] == ')':
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_top_level_args(args_text: str) -> list[str]:
    """Split CallNode arg string on top-level spaces."""
    args = []
    current = []
    depth = 0

    for ch in args_text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ' ' and depth == 0:
            if current:
                args.append(''.join(current).strip())
                current = []
        else:
            current.append(ch)

    if current:
        args.append(''.join(current).strip())

    return [a for a in args if a]


def _replace_template_calls_in_expr(expr_text: str, templates_map: dict) -> str:
    """
    Rewrite template calls in AST string form:
    (IDENTIFIER:sum(INT:9 INT:8)) -> (INT:9 PLUS INT:8)
    """
    out = []
    i = 0
    ident_prefix = '(IDENTIFIER:'

    while i < len(expr_text):
        if expr_text.startswith(ident_prefix, i):
            name_start = i + len(ident_prefix)
            name_end = name_start
            while name_end < len(expr_text) and (expr_text[name_end].isalnum() or expr_text[name_end] == '_'):
                name_end += 1

            func_name = expr_text[name_start:name_end]
            if name_end < len(expr_text) and expr_text[name_end] == '(' and func_name in templates_map:
                args_open = name_end
                args_close = _find_matching_paren(expr_text, args_open)
                if args_close != -1 and args_close + 1 < len(expr_text) and expr_text[args_close + 1] == ')':
                    raw_args = expr_text[args_open + 1:args_close]
                    arg_exprs = _split_top_level_args(raw_args) if raw_args.strip() else []
                    params, body_ast = templates_map[func_name]

                    if len(arg_exprs) == len(params):
                        expanded_args = [
                            _replace_template_calls_in_expr(arg, templates_map)
                            for arg in arg_exprs
                        ]
                        replaced = body_ast
                        for param, arg in zip(params, expanded_args):
                            replaced = re.sub(
                                rf'IDENTIFIER:{re.escape(param)}\b',
                                lambda _m, a=arg: a,
                                replaced
                            )
                        replaced = _replace_template_calls_in_expr(replaced, templates_map)
                        out.append(replaced)
                        i = args_close + 2  # consume " )) "
                        continue

        out.append(expr_text[i])
        i += 1

    return ''.join(out)


def _parse_one_ast_node(s: str, start: int):
    """Parse one AST subtree string at start; returns (node_str, end_index) or (None, start)."""
    while start < len(s) and s[start].isspace():
        start += 1
    if start >= len(s):
        return None, start
    if s[start] == '(':
        close = _find_matching_paren(s, start)
        if close == -1:
            return None, start
        return s[start : close + 1], close + 1
    m = re.match(r'[A-Z_]+:[^\s()]+', s[start:])
    if m:
        return m.group(0), start + len(m.group(0))
    return None, start


def _try_expand_infix_op(segment: str, op_map: dict) -> Optional[str]:
    """If segment is '(left IDENTIFIER:op right)' for a template op, return inlined body AST string."""
    if len(segment) < 2 or segment[0] != '(' or segment[-1] != ')':
        return None
    inner = segment[1:-1]
    for op_name, (params, body_ast) in op_map.items():
        needle = f' IDENTIFIER:{op_name} '
        pos = inner.find(needle)
        while pos != -1:
            left_s, le = _parse_one_ast_node(inner, 0)
            r_start = pos + len(needle)
            right_s, rend = _parse_one_ast_node(inner, r_start)
            if (
                left_s is not None
                and right_s is not None
                and le == pos
                and rend == len(inner)
                and len(params) == 2
            ):
                replaced = body_ast
                for param, arg in zip(params, (left_s, right_s)):
                    replaced = re.sub(
                        rf'IDENTIFIER:{re.escape(param)}\b',
                        lambda _m, a=arg: a,
                        replaced,
                    )
                return replaced
            pos = inner.find(needle, pos + 1)
    return None


def _replace_infix_identifier_ops_in_expr(expr_text: str, op_map: dict) -> str:
    """Inline (left IDENTIFIER:op right) using FUNC_TEMPLATES bodies."""
    if not op_map:
        return expr_text
    out = expr_text
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            if out[i] == '(':
                close = _find_matching_paren(out, i)
                if close != -1:
                    seg = out[i : close + 1]
                    exp = _try_expand_infix_op(seg, op_map)
                    if exp is not None:
                        out = out[:i] + exp + out[close + 1 :]
                        changed = True
                        break
            i += 1
    return out


def get_replaced_methods(ast_text: str) -> str:
    """Inline function calls and learned infix ops into primitive AST."""
    func_map = {
        name: (params, str(body_node))
        for name, params, body_node in _get_compiled_func_templates()
    }
    op_map = {
        name: (params, str(body_node))
        for name, params, body_node in _get_compiled_op_templates()
    }
    prev = None
    t = ast_text
    while t != prev:
        prev = t
        t = _replace_template_calls_in_expr(t, func_map)
        t = _replace_infix_identifier_ops_in_expr(t, op_map)
    return t


def _substitute_params(body, params, args):
    """Replace parameter names in body with concrete argument values (whole-word)."""
    result = body
    for param, arg in zip(params, args):
        result = re.sub(rf'\b{re.escape(param)}\b', arg, result)
    return result


def generate_func_call_sample(idx):
    """Generate a single (call_lex, body_ast) pair with random args.
    Returns (lex_text, ast_text, x_in, x_out) or Nones on failure."""
    name, params, body = random.choice(FUNC_TEMPLATES)
    args = [generate_number() for _ in params]

    call_text = f"{name}({', '.join(args)})"
    body_text = _substitute_params(body, params, args)

    # Lex the call (model input)
    lexer_call = basic.Lexer('<stdin>', call_text)
    call_tokens, error = lexer_call.make_tokens()
    if error:
        return None
    lex_text = ' '.join(t.__repr__() for t in call_tokens)

    # Parse the substituted body (model target)
    lexer_body = basic.Lexer('<stdin>', body_text)
    body_tokens, error = lexer_body.make_tokens()
    if error:
        return None

    parser = basic.Parser(body_tokens)
    ast = parser.parse()
    if ast.error:
        return None
    ast_text = f'{tokens.SOF} {ast.node} {tokens.EOF}'

    lex_encoded = data.encode(lex_text, {})
    ast_encoded = data.encode(ast_text, {})
    if len(lex_encoded) > BLOCK_SIZE or len(ast_encoded) > BLOCK_SIZE:
        return None

    x_in = data.add_pad_tokens_and_trim(lex_encoded, BLOCK_SIZE)
    x_out = data.add_pad_tokens_and_trim(ast_encoded, BLOCK_SIZE)
    return {
        'lexer_text': f'\n{lex_text}',
        'ast_text': f'\n{ast_text}',
        'text': call_text,
        'x_in': [x_in],
        'x_out': [x_out],
        'symbols': {'_output_list': []},
        'id': idx,
    }


class Sample:
    lexer_text: str
    ast_text: str
    text: str
    x_in: list
    x_out: list
    symbols: dict
    id: Optional[int]

    def __init__(self, statements, idx):
        self.lexer_text = ''
        self.ast_text = ''
        self.text = '\n'.join(statements)
        self.x_in = []
        self.x_out = []
        self.symbols = {'_output_list': []}
        self.id = idx


def print_program(statements):
    print('\nProgram sample------------------')
    print(f'\n'.join(statements))
    print('----------------------------------\n')


def generate():
    invalid_count = 0
    rows = []
    texts = set()

    for idx in range(NUM_TRAINING_SAMPLES):
        is_valid = True
        statements = generate_program_statements(texts)
        symbol_table = basic.get_symbol_table()
        sample = Sample(statements, idx)

        if idx % 1_000 == 0:
            print_program(statements)
            print(f"loaded: {(idx / NUM_TRAINING_SAMPLES) * 100:.2f}%")

        for text in statements:
            lexer = basic.Lexer('<stdin>', text)
            try:
                token_list, error = lexer.make_tokens()
                if error:
                    print(f'Lexing is invalid!: {error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break
                lexer_text = ' '.join(t.__repr__() for t in token_list)
                sample.lexer_text += f"\n{lexer_text}"

                if random.random() < 0.5:
                    text_error = text.replace('var ', '')
                    lexer_error = basic.Lexer('<stdin>', text_error)
                    token_list_error, error = lexer_error.make_tokens()
                    if error:
                        print(f'Lexing is invalid: {error.as_string()}')
                        invalid_count += 1
                        is_valid = False
                        break
                    lexer_text = ' '.join(t.__repr__() for t in token_list_error)

                # Try to parse
                parser = basic.Parser(token_list)
                ast = parser.parse()
                if ast.error:
                    print(f'Parsing is invalid: {ast.error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break
                ast_text = f'{tokens.SOF} {ast.node} {tokens.EOF}'
                ast_text = get_replaced_methods(ast_text)
                sample.ast_text += f'\n{ast_text}'

                interpreter = basic.Interpreter()
                context = basic.Context('<program>')
                context.symbol_table = symbol_table
                _load_template_functions_into_context(context)
                res = interpreter.visit(ast.node, context)
                symbol_table = context.symbol_table
                if res.error:
                    if 'zero' not in res.error.as_string():
                        print(f'Interpretation is invalid!: {res.error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break

                lex_encoded = data.encode(lexer_text, {})
                ast_encoded = data.encode(ast_text, {})
                if len(lex_encoded) <= BLOCK_SIZE and len(ast_encoded) <= BLOCK_SIZE:
                    sample.x_in.append(data.add_pad_tokens_and_trim(lex_encoded, BLOCK_SIZE))
                    sample.x_out.append(data.add_pad_tokens_and_trim(ast_encoded, BLOCK_SIZE))
                    sample.symbols |= symbol_table.symbols
                    if res.value:
                        sample.symbols['_output_list'].append(res.value)
                else:
                    #print('Encodings are too long...')
                    invalid_count += 1
                    is_valid = False
                    break
            except Exception:
                invalid_count += 1
                is_valid = False
                break

        if is_valid:
            rows.append(sample.__dict__)

    samples_df = pd.DataFrame(rows)
    # Random shuffle with random seed
    samples_df = samples_df.sample(frac=1, random_state=random.randint(0, 2**31 - 1)).reset_index(drop=True)
    samples_df.to_pickle(DATASET_NAME)  # Save dataset as a Pandas DataFrame (pickled)

    valid_count = len(samples_df)
    print(f"\nSample generation: Valid: {valid_count}, Invalid: {invalid_count}, Success rate: {valid_count / NUM_TRAINING_SAMPLES * 100:.1f}%")


if __name__ == '__main__':
    generate()
