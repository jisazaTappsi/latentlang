"""Training data: random valid programs over the whole language, as (lexer text, AST text) samples.

Programs are built per value type from the operations the interpreter accepts for that type
(the `Value` classes in basic.py), so type errors, undefined names, wrong call arities and a
`break`/`continue`/`return` outside its loop or function are never generated. What types can't
rule out (division by zero, list indexes, loops that don't end) is handled by running each
candidate statement for real, after the statements before it, under the MAX_INTERPRETER_STEPS
budget: it is kept only if it runs cleanly and fits BLOCK_SIZE, otherwise another is drawn.

The generator emits source text, and a sample's AST comes from parsing that text, so the two
sides of a sample can never disagree. Template calls are inlined in the AST, so the model learns
to emit only primitive operations.

Write the dataset with: python -m interpreter.data_generator
"""
import copy
import functools
import math
import random
from collections import Counter, namedtuple
from typing import Optional

import pandas as pd

from . import data
from . import basic
from . import tokens
from .templates import FUNC_TEMPLATES
from .util import BLOCK_SIZE, DATASET_NAME, MAX_INTERPRETER_STEPS, NUM_TRAINING_SAMPLES


########################
# TEMPLATES
########################

def _unwrap_single_statement(node):
    """Strip the BlockNode that `Parser.parse` wraps around a lone expression.

    Template bodies are single expressions, and a `Function` returns whatever its body
    node evaluates to. Left wrapped, every template call would return List([value])
    instead of value, so `7 + (8 times 8)` becomes an illegal operation.
    """
    if isinstance(node, basic.BlockNode) and len(node.statement_nodes) == 1:
        return node.statement_nodes[0]
    return node


@functools.cache
def _get_compiled_func_templates():
    """Compile FUNC_TEMPLATES body expressions into AST nodes once."""
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
        compiled.append((name, params, _unwrap_single_statement(ast.node)))
    return compiled


@functools.cache
def _templates_by_name():
    return {name: (params, body_node) for name, params, body_node in _get_compiled_func_templates()}


def _load_template_functions_into_context(context):
    """
    Inject template bodies as functions, callable as `name(a, b)` and as infix `a name b`
    (visit_BinOpNode runs the same Function). Template bodies are bare expressions with no
    explicit `return`, so they need should_auto_return=True; block-bodied functions would use
    False and return explicitly.
    """
    for name, params, body_node in _get_compiled_func_templates():
        func_value = basic.Function(name, body_node, params, should_auto_return=True).set_context(context).set_pos()
        context.symbol_table.set(name, func_value)


def _map_child_nodes(node, fn):
    """Shallow copy of `node` with `fn` applied to each child node, however it is held
    (IfNode keeps children in tuples, CallNode, ListNode and BlockNode in lists)."""
    def map_value(value):
        if isinstance(value, list):
            return [map_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(map_value(v) for v in value)
        if type(value).__name__.endswith('Node'):
            return fn(value)
        return value

    new_node = copy.copy(node)
    for key, value in vars(node).items():
        setattr(new_node, key, map_value(value))
    return new_node


def _substitute_params(body_node, params, args):
    """`body_node` with every parameter reference replaced by its argument node. All at once,
    so an argument that mentions another parameter's name (`sum(b, 1)` for `a+b`) is left alone."""
    bindings = dict(zip(params, args))

    def substitute(node):
        if isinstance(node, basic.VarAccessNode) and node.var_name_tok.value in bindings:
            return bindings[node.var_name_tok.value]
        return _map_child_nodes(node, substitute)

    return substitute(body_node)


def inline_templates(node):
    """Replace template calls `sum(a, b)` and infix uses `a sum b` with the template's body,
    so the AST the model learns to emit only uses primitive operations."""
    templates = _templates_by_name()

    def inline(node):
        node = _map_child_nodes(node, inline)  # Arguments first.
        if isinstance(node, basic.CallNode) and isinstance(node.node_to_call, basic.VarAccessNode):
            name, args = node.node_to_call.var_name_tok.value, node.arg_nodes
        elif isinstance(node, basic.BinOpNode) and node.op_tok.type == tokens.IDENTIFIER:
            name, args = node.op_tok.value, [node.left_node, node.right_node]
        else:
            return node
        if name not in templates or len(templates[name][0]) != len(args):
            return node
        params, body_node = templates[name]
        return inline(_substitute_params(body_node, params, args))  # The body may use templates too.

    return inline(node)


########################
# TYPES
########################

NUM, STR = 'num', 'str'


def list_of(elem):
    return ('list', elem)


def fun_of(params, ret):
    return ('fun', tuple(params), ret)


def is_list(t):
    return isinstance(t, tuple) and t[0] == 'list'


def matches(actual, wanted):
    """An empty list, typed `list_of(None)`, can stand in for a list of anything."""
    return actual == wanted or (is_list(actual) and is_list(wanted) and actual[1] is None)


def random_type():
    r = random.random()
    if r < 0.5:
        return NUM
    if r < 0.7:
        return STR
    if r < 0.95:
        return list_of(random.choice([NUM, NUM, STR]))
    return list_of(list_of(NUM))


def value_type(value):
    """Type of a runtime value, or None for values the generator doesn't use (mixed lists)."""
    if isinstance(value, basic.Number):
        return NUM if isinstance(value.value, (int, float)) else None
    if isinstance(value, basic.String):
        return STR
    if isinstance(value, basic.List):
        types = {value_type(element) for element in value.elements}
        if not types:
            return list_of(None)
        if len(types) == 1 and None not in types:
            return list_of(types.pop())
    return None


########################
# NAMES
########################

CONSTANTS = [tokens.NULL, tokens.TRUE, tokens.FALSE, 'math_pi']
BUILT_INS = {name for name, value in basic.global_symbol_table.symbols.items() if isinstance(value, basic.BuiltInFunction)}
# Never assigned: keywords, built-ins, constants, and templates (inlining rewrites any call by name).
RESERVED = set(tokens.KEYWORDS) | BUILT_INS | set(CONSTANTS) | {name for name, _, _ in FUNC_TEMPLATES}
LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def new_context():
    """A program context with only the built-ins, constants and templates. Not a copy of the
    global table, which a REPL session or a test may have filled with variables (and lists the
    copy would share)."""
    table = basic.get_symbol_table()
    table.symbols = {name: value for name, value in table.symbols.items() if name in BUILT_INS or name in CONSTANTS}
    context = basic.Context('<program>')
    context.symbol_table = table
    _load_template_functions_into_context(context)
    return context


def _template_signatures():
    """name -> (fun type, indexes of the params the body divides by), read off the template bodies."""
    signatures = {}
    for name, params, body in _get_compiled_func_templates():
        divisors = set()

        def find_divisors(node):
            if isinstance(node, basic.BinOpNode) and node.op_tok.type == tokens.DIV \
                    and isinstance(node.right_node, basic.VarAccessNode) and node.right_node.var_name_tok.value in params:
                divisors.add(params.index(node.right_node.var_name_tok.value))
            return _map_child_nodes(node, find_divisors)

        find_divisors(body)
        signatures[name] = (fun_of([NUM] * len(params), NUM), divisors)
    return signatures


TEMPLATES = _template_signatures()


########################
# PRINTING
########################

# How tightly a printed expression binds, following grammar.txt. An operand whose level is
# above what its position allows gets parenthesized. GREEDY forms (inline if/for/while/def,
# `var x = ...`) run to the end of the line, so they're parenthesized whenever anything follows.
ATOM, POWER, UNARY, TERM, ARITH, COMP, NOT, LOGIC, GREEDY = range(9)

Expr = namedtuple('Expr', 'src type level')


def operand(expr, max_level):
    if expr.level > max_level or (expr.level > ATOM and random.random() < 0.05):
        return f'({expr.src})'
    return expr.src


def binary(left, op, right, type_, level, left_max, right_max):
    space = ' ' if op[0].isalpha() or random.random() < 0.7 else ''  # Word operators need their spaces.
    return Expr(f'{operand(left, left_max)}{space}{op}{space}{operand(right, right_max)}', type_, level)


def int_literal(value):
    return Expr(str(value), NUM, ATOM) if value >= 0 else Expr(f'-{-value}', NUM, UNARY)


def log_scale_int(low, high):
    """An integer in [low, high], log-uniform so smaller numbers are more likely. `low` must be >= 1."""
    if low >= high:
        return low
    value = math.exp(random.uniform(math.log(low), math.log(high)))
    return min(high, max(low, round(value)))


def num_literal(nonzero=False):
    r = random.random()
    if r < 0.1 and not nonzero:
        return '0'
    if r < 0.75:
        return str(log_scale_int(1, 1000))
    if r < 0.85:
        return str(log_scale_int(1, 10**6))
    return f'{log_scale_int(1, 1000)}.{random.randint(1, 99):02d}'


COMMENTS = ['# note', '# todo: check this', '#x = 1 (not code)']
STRING_WORDS = ['', 'a', 'hi', 'latent', 'lang', 'x y', 'hello world', 'ok!', 'n=1']
STRING_CHARS = 'abcxyz  ()[]:,."\'\\\n\t'


def str_literal():
    if random.random() < 0.7:
        value = random.choice(STRING_WORDS)
    else:
        value = ''.join(random.choice(STRING_CHARS) for _ in range(random.randint(1, 5)))
    quote = random.choice('"\'')
    escaped = value.replace('\\', '\\\\').replace(quote, f'\\{quote}').replace('\n', '\\n').replace('\t', '\\t')
    return f'{quote}{escaped}{quote}'


def call_text(callee, args):
    return f'{callee}({("," if random.random() < 0.3 else ", ").join(args)})'


########################
# SCOPE
########################

class Scope:
    """What a statement may use. The top-level scope is read from the real symbol table, so it also
    knows current values; a body's scope is tracked while generating it and only knows types."""

    def __init__(self, variables, functions, values=None, in_loop=False, return_type=None, fixed=()):
        self.variables = variables  # name -> type of the data it holds
        self.functions = functions  # name -> fun type of the user function it holds
        self.values = values or {}  # name -> current basic.Value, top level only
        self.in_loop = in_loop
        self.return_type = return_type  # set inside a function body
        self.fixed = set(fixed)  # names the body must not reassign, like its while counter

    def child(self, **changes):
        """A body's scope: its own copies, and no known values (a loop body runs many times)."""
        scope = Scope(dict(self.variables), dict(self.functions), None, self.in_loop, self.return_type, self.fixed)
        for key, value in changes.items():
            setattr(scope, key, value)
        return scope

    def of_type(self, wanted):
        return [name for name, t in self.variables.items() if matches(t, wanted)]

    def assignable(self, wanted=None):
        return [name for name, t in self.variables.items()
                if name not in CONSTANTS and name not in self.fixed and (wanted is None or t == wanted)]

    def known_length(self, name):
        value = self.values.get(name)
        return len(value.elements) if isinstance(value, basic.List) else None


def symbol_reprs(symbols):
    return {name: repr(value) for name, value in symbols.items()}


def choose(options):
    """Call a weighted random option; options returning None don't apply, so try another."""
    options = [(weight, make) for weight, make in options if weight > 0]
    while options:
        index = random.choices(range(len(options)), weights=[weight for weight, _ in options])[0]
        result = options.pop(index)[1]()
        if result is not None:
            return result
    return None


########################
# GENERATOR
########################

STATS = Counter()  # Statements drawn, kept and rejected (with why), across all programs.


class ProgramGenerator:
    MAX_ATTEMPTS = 40
    INDENT = '    '

    def __init__(self, texts):
        self.texts = texts  # Every statement already used, so none repeats across the dataset.
        self.accepted = []  # (text, AST node) per kept statement
        self.signatures = {}  # name -> fun type, for user functions in kept statements
        self.next_kind = None  # Statement kind a setup statement prepared for.
        self.symbols, _ = self.run([])

    @staticmethod
    def run(nodes):
        """Run `nodes` as one program: (symbols, None), or (None, why it failed)."""
        context = new_context()
        interpreter = basic.Interpreter(max_steps=MAX_INTERPRETER_STEPS)
        for node in nodes:
            try:
                res = interpreter.visit(node, context)
            except RecursionError:
                return None, 'recursion limit'
            except Exception as e:  # Operations the interpreter doesn't guard, e.g. float overflow.
                return None, f'python {type(e).__name__}'
            if res.error:
                return None, 'step limit' if isinstance(res.error, basic.StepLimitError) else 'runtime error'
        return context.symbol_table.symbols, None

    # Retries of one statement kind shrink it, so kinds that tend to run past BLOCK_SIZE (blocks)
    # get smaller instead of being replaced by kinds that fit (assignments).
    DEPTHS = [2, 2, 2, 1, 1, 1, 1, 0, 0, 0]

    def add_statement(self):
        """Draw statements until one is valid after the kept ones; False if none was."""
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt % len(self.DEPTHS) == 0:
                kind = self.next_kind or random.choices(list(self.KINDS), weights=list(self.KINDS.values()))[0]
                self.next_kind = None
            self.pending = {}  # Function signatures this candidate introduces.
            self.used_names = set()
            self.sep = '\n' if random.random() < 0.8 else '; '
            lines = self.top_statement(kind, self.DEPTHS[attempt % len(self.DEPTHS)])
            label = kind
            if lines is None:
                # The kind needs a variable that isn't there yet (a while counter, a list for a
                # built-in): assign one now, and draw that kind again next. A rejected setup is
                # retried as the same kind, so it gets another setup.
                self.next_kind, label = kind, 'setup'
                lines = self.setup_statement(kind)
            reason = self.try_accept(self.with_comment(lines[0]))
            STATS[f'drawn {label}'] += 1
            STATS[f'kept {label}' if reason is None else f'rejected: {reason}'] += 1
            if reason is None:
                return True
        return False

    def try_accept(self, text):
        if text in self.texts:
            return 'duplicate'
        token_list, error = basic.Lexer('<stdin>', text).make_tokens()
        if error:
            return 'lex error'
        ast = basic.Parser(token_list).parse()
        if ast.error:
            return 'syntax error'
        lex_text = ' '.join(repr(token) for token in token_list)
        inlined = inline_templates(ast.node)
        ast_text = f'{tokens.SOF} {inlined} {tokens.EOF}'
        if max(len(data.encode(lex_text, {})), len(data.encode(ast_text, {}))) > BLOCK_SIZE:
            return 'too long'
        prefix = [node for _, node in self.accepted]
        symbols, reason = self.run(prefix + [ast.node])
        if reason:
            return reason
        if repr(inlined) != repr(ast.node):
            # The model learns the inlined form, which repeats an argument for each use of its
            # parameter: an argument with side effects, like `square(pop(l, 0))`, would then run twice.
            inlined_symbols, reason = self.run(prefix + [inlined])
            if reason or symbol_reprs(inlined_symbols) != symbol_reprs(symbols):
                return 'inlining changes the result'
        self.texts.add(text)
        self.accepted.append((text, ast.node))
        self.signatures.update(self.pending)
        self.symbols = symbols
        return None

    def top_scope(self):
        variables = {name: NUM for name in CONSTANTS}
        functions, values = {}, {}
        for name, value in self.symbols.items():
            if name in RESERVED:
                continue
            if isinstance(value, basic.Function):
                signature = self.signatures.get(name)
                if signature and len(signature[1]) == len(value.arg_names):
                    functions[name] = signature
                continue
            type_ = value_type(value)
            if type_ is not None:
                variables[name] = type_
                values[name] = value
        return Scope(variables, functions, values)

    def fresh_name(self, scope):
        taken = set(scope.variables) | set(scope.functions) | set(self.symbols) | self.used_names | RESERVED
        free = [c for c in LETTERS if c not in taken]
        name = random.choice(free) if free else next(a + b for a in LETTERS for b in LETTERS if a + b not in taken)
        self.used_names.add(name)
        return name

    # Layout

    def block_text(self, head, statements):
        """`head`, then the statements as an indented block (or `;`-separated on one line)."""
        if self.sep == '\n':
            body = '\n'.join(self.INDENT + line for statement in statements for line in statement.split('\n'))
            blank = '\n' if random.random() < 0.1 else ''  # Blank lines before a block's first statement.
            return f'{self.with_comment(head)}\n{blank}{body}'
        return f'{head}; ' + '; '.join(statements)

    def with_comment(self, line):
        """Sometimes end `line` with a comment. Only in newline layout: a comment runs to the end of the line."""
        if self.sep == '\n' and random.random() < 0.05:
            return f'{line}  {random.choice(COMMENTS)}'
        return line

    def then_line(self, text, piece, new_line):
        """Continue `text` with `piece`: on the same line after an inline branch, else on the next."""
        return f'{text}{self.sep}{piece}' if new_line else f'{text} {piece}'

    def if_text(self, cases, else_case):
        """`cases`: (condition, body) pairs; a body is ('inline', statement) or ('block', statements)."""
        text, after_block = None, False
        branches = [(f'{tokens.IF if k == 0 else tokens.ELIF} {condition} then', body) for k, (condition, body) in enumerate(cases)]
        if else_case:
            branches.append((tokens.ELSE, else_case))
        for head, (form, body) in branches:
            piece = f'{head} {body}' if form == 'inline' else self.block_text(head, body)
            text = piece if text is None else self.then_line(text, piece, after_block)
            after_block = form == 'block'
        return self.then_line(text, tokens.END, True) if after_block else text

    # Statements: each returns a list of statement texts, or None when it doesn't apply.

    KINDS = {'assign': 4, 'expression': 2, 'def': 2, 'if': 2.5, 'for': 2, 'while': 1.5, 'builtin': 1}

    def top_statement(self, kind, depth):
        scope = self.top_scope()
        nest = 2 if depth > 1 else 1
        if kind == 'assign':
            return self.assign(scope, depth)
        if kind == 'expression':
            return self.expression_statement(scope, depth)
        if kind == 'def':
            return self.def_statement(scope, depth, nest)
        if kind == 'if':
            return self.if_statement(scope, depth, nest)
        if kind == 'for':
            return self.for_statement(scope, depth, nest)
        if kind == 'while':
            return self.while_statement(scope, depth, nest, top_level=True)
        if kind == 'builtin':
            return self.mutation_statement(scope, depth)
        raise ValueError(f'Unknown statement kind: {kind}')

    def setup_statement(self, kind):
        """`var n = 2` for a while loop's counter, `var l = [...]` for a list built-in."""
        scope = self.top_scope()
        name = self.fresh_name(scope)
        if kind == 'while':
            # Wide range so the setup line stays unique across a large dataset; the loop bound is
            # taken relative to it, so the loop stays short.
            start = random.randint(-5, 10) if random.random() < 0.5 else log_scale_int(1, 10**4)
            return [f'{tokens.VAR} {name} = {int_literal(start).src}']
        return [f'{tokens.VAR} {name} = {self.list_literal(random.choice([NUM, STR]), scope, 1, random.randint(1, 3)).src}']

    def block(self, scope, depth, nest):
        """1-3 statements for a block body (fewer when shrinking), generated in `scope`, which they
        extend as they go."""
        statements = []
        for _ in range(random.choice([1, 1, 2, 2, 3] if depth > 0 else [1, 1, 2])):
            statements += self.body_statement(scope, depth, nest)
        return statements

    def body_statement(self, scope, depth, nest):
        options = [
            (4, lambda: self.assign(scope, depth)),
            (2, lambda: self.expression_statement(scope, depth)),
            (2, lambda: self.mutation_statement(scope, depth)),
            (3 if scope.in_loop or scope.return_type else 0, lambda: self.control_statement(scope, depth)),
        ]
        if nest > 0:
            options += [
                (2, lambda: self.if_statement(scope, depth, nest - 1)),
                (1, lambda: self.for_statement(scope, depth, nest - 1)),
                (1, lambda: self.while_statement(scope, depth, nest - 1)),
            ]
        return choose(options)

    def inline_statement(self, scope, depth):
        """The single statement of an inline body. Its value is kept below GREEDY, so a following
        `elif`/`else` can't be swallowed by it."""
        options = [
            (3, lambda: self.assign(scope, depth, inline=True)),
            (2, lambda: self.expression_statement(scope, depth, inline=True)),
            (2 if scope.in_loop or scope.return_type else 0, lambda: self.control_statement(scope, depth, inline=True)),
        ]
        return choose(options)[0]

    def assign(self, scope, depth, inline=False):
        if random.random() < 0.08:
            return self.assign_function(scope, depth, inline)
        existing = scope.assignable()
        if existing and random.random() < 0.4:
            name = random.choice(existing)
            # Only the top level (the scope that knows values) may change a variable's type: a body
            # runs later or repeatedly, and the rest of it relies on the type it started with.
            in_body = not scope.values
            type_ = scope.variables[name] if in_body or random.random() < 0.7 else random_type()
            if type_ == list_of(None):
                type_ = list_of(NUM)
        else:
            name, type_ = self.fresh_name(scope), random_type()
        value = self.expr(type_, scope, depth)
        scope.variables[name] = type_
        scope.values.pop(name, None)
        equals = ' = ' if random.random() < 0.8 else '='
        return [f'{tokens.VAR} {name}{equals}{operand(value, LOGIC if inline else GREEDY)}']

    def assign_function(self, scope, depth, inline):
        """`var g = f` or `var g = def (a) -> ...`: g becomes callable with the same signature."""
        name = self.fresh_name(scope)
        if scope.functions and random.random() < 0.4:
            source = random.choice(list(scope.functions))
            value, signature = Expr(source, scope.functions[source], ATOM), scope.functions[source]
        else:
            value = self.anonymous_function(scope, depth)
            signature = value.type
        scope.functions[name] = signature
        self.pending[name] = signature
        return [f'{tokens.VAR} {name} = {operand(value, LOGIC if inline else GREEDY)}']

    def expression_statement(self, scope, depth, inline=False):
        value = self.expr(random_type(), scope, depth)
        return [operand(value, LOGIC) if inline else value.src]

    def control_statement(self, scope, depth, inline=False):
        options = []
        if scope.in_loop:
            for keyword in (tokens.BREAK, tokens.CONTINUE):
                options.append((1, lambda keyword=keyword: keyword))
        if scope.return_type:
            options.append((2, lambda: self.return_text(scope, depth)))
        statement = choose(options)
        if inline or random.random() < 0.3:
            return [statement]
        # Usually guarded, so the statements after it stay reachable.
        return [f'{tokens.IF} {operand(self.condition(scope, depth), LOGIC)} then {statement}']

    def return_text(self, scope, depth):
        if scope.return_type == NUM and random.random() < 0.15:
            return tokens.RETURN  # Returns null.
        return f'{tokens.RETURN} {operand(self.expr(scope.return_type, scope, depth), LOGIC)}'

    def mutation_statement(self, scope, depth):
        """append(l, x), extend(l, m) or pop(l, i) on a list variable: in-place built-ins."""
        lists = [name for name in scope.assignable() if is_list(scope.variables[name])]
        if not lists:
            return None
        name = random.choice(lists)
        elem = scope.variables[name][1] or random.choice([NUM, STR])
        length = scope.known_length(name)
        options = [
            (2, lambda: call_text('append', [name, operand(self.expr(elem, scope, depth), GREEDY)])),
            (1, lambda: call_text('extend', [name, operand(self.expr(list_of(elem), scope, depth), GREEDY)])),
            (1 if length else 0, lambda: call_text('pop', [name, int_literal(random.randint(-length, length - 1)).src])),
        ]
        statement = choose(options)
        scope.variables[name] = list_of(elem)
        scope.values.pop(name, None)
        return [statement]

    def if_statement(self, scope, depth, nest):
        cases = []
        for _ in range(random.choice([1, 1, 2, 3] if depth > 1 else [1, 1, 2])):
            cases.append((operand(self.condition(scope, depth), LOGIC), self.branch(scope, depth, nest)))
        else_case = self.branch(scope, depth, nest) if random.random() < (0.5 if depth > 0 else 0.3) else None
        return [self.if_text(cases, else_case)]

    def branch(self, scope, depth, nest):
        body_scope = scope.child()
        if random.random() < 0.5:
            return 'inline', self.inline_statement(body_scope, depth - 1 if depth > 0 else 0)
        return 'block', self.block(body_scope, max(depth - 1, 0), nest)

    def for_statement(self, scope, depth, nest):
        variable = self.fresh_name(scope)
        body_scope = scope.child(in_loop=True)
        body_scope.variables[variable] = NUM
        head = f'{tokens.FOR} {variable} = {self.loop_bounds(scope)} then'
        if random.random() < 0.5:
            return [f'{head} {self.inline_statement(body_scope, max(depth - 1, 0))}']
        return [self.then_line(self.block_text(head, self.block(body_scope, max(depth - 1, 0), nest)), tokens.END, True)]

    def loop_bounds(self, scope):
        """`start to end [step s]` running 0-5 times."""
        count = random.randint(0, 5)
        lists = [name for name in scope.of_type(list_of(NUM)) + scope.of_type(list_of(STR))
                 if scope.known_length(name) is not None and scope.known_length(name) <= 6]
        if lists and random.random() < 0.15:
            return f'0 {tokens.TO} len({random.choice(lists)})'
        if random.random() < 0.2:
            start = random.randint(3, 8)
            return f'{start} {tokens.TO} {start - count} {tokens.STEP} -{random.choice([1, 1, 2])}'
        start = random.randint(0, 3)
        step = random.choice(['', '', f' {tokens.STEP} 1', f' {tokens.STEP} 2'])
        return f'{start} {tokens.TO} {start + count}{step}'

    def while_statement(self, scope, depth, nest, top_level=False, inline=None):
        """A while loop that ends: a counter is moved first thing in the body (so a `continue`
        can't skip it) towards a bound a few steps away."""
        setup = []
        if top_level:
            counters = [name for name in scope.assignable(NUM)
                        if isinstance(scope.values.get(name), basic.Number) and abs(scope.values[name].value) < 10**6]
            if not counters:
                return None
            counter = random.choice(counters)
            start = int(scope.values[counter].value)
        else:
            counter, start = self.fresh_name(scope), random.randint(0, 3)
            setup = [f'{tokens.VAR} {counter} = {start}']
            scope.variables[counter] = NUM

        up = random.random() < 0.7
        bound = int_literal(start + random.randint(1, 4) if up else start - random.randint(1, 4)).src
        step = f'{tokens.VAR} {counter} = {counter} {"+" if up else "-"} 1'
        test = f'{counter} {"<" if up else ">"} {bound}'
        body_scope = scope.child(in_loop=True, fixed=scope.fixed | {counter})

        form = 0 if inline else random.random()
        if form < 0.3:
            return setup + [f'{tokens.WHILE} {test} then {step}']
        if form < 0.7:
            head, body = f'{tokens.WHILE} {test} then', [step]
        else:
            head = f'{tokens.WHILE} {random.choice(["1", tokens.TRUE])} then'
            body = [step, f'{tokens.IF} {counter} {">=" if up else "<="} {bound} then {tokens.BREAK}']
        body += self.block(body_scope, max(depth - 1, 0), nest) if random.random() < 0.7 else []
        return setup + [self.then_line(self.block_text(head, body), tokens.END, True)]

    def def_statement(self, scope, depth, nest):
        name = self.fresh_name(scope)
        params = [self.fresh_name(scope) for _ in range(random.choice([0, 1, 1, 2, 2, 3]))]
        param_types = [random.choice([NUM, NUM, NUM, STR, list_of(NUM)]) for _ in params]
        ret = random_type()
        # Function bodies see only their params and other functions: they run when called, by
        # which time a global they read may hold another type.
        body_scope = Scope({**{c: NUM for c in CONSTANTS}, **dict(zip(params, param_types))},
                           dict(scope.functions), return_type=ret)
        head = f'{tokens.FUN} {name}({", ".join(params)})'
        signature = fun_of(param_types, ret)
        scope.functions[name] = signature
        self.pending[name] = signature

        if random.random() < 0.5:
            return [f'{head} -> {self.expr(ret, body_scope, max(depth - 1, 0)).src}']
        body = self.block(body_scope, max(depth - 1, 0), nest - 1) if random.random() < 0.6 else []
        if ret != NUM or random.random() < 0.7:  # A function that falls off its end returns null.
            body.append(self.return_text(body_scope, max(depth - 1, 0)))
        if not body:
            body = [self.return_text(body_scope, 0)]
        return [self.then_line(self.block_text(head, body), tokens.END, True)]

    # Expressions

    def expr(self, type_, scope, depth):
        if type_ is None:
            type_ = NUM
        if type_ == NUM:
            options = self.num_options(scope, depth)
        elif type_ == STR:
            options = self.str_options(scope, depth)
        else:
            options = self.list_options(type_, scope, depth)
        if depth > 0:
            d = depth - 1
            options += [
                (2, lambda: self.call(type_, scope, d)),
                (1, lambda: self.index(type_, scope, d)),
                (0.7, lambda: self.inline_if(type_, scope, d)),
            ]
        return choose(options)

    def variable(self, type_, scope):
        names = scope.of_type(type_)
        own = [name for name in names if name not in CONSTANTS]
        if own and random.random() < 0.85:  # Constants are always in scope; don't let them crowd out the rest.
            names = own
        return Expr(random.choice(names), type_, ATOM) if names else None

    def num_options(self, scope, depth):
        options = [
            (3, lambda: Expr(num_literal(), NUM, ATOM)),
            (4, lambda: self.variable(NUM, scope)),
        ]
        if depth > 0:
            d = depth - 1
            num = lambda: self.expr(NUM, scope, d)
            options += [
                (4, lambda: self.arithmetic(scope, d)),
                (1, lambda: Expr(f'{random.choice(["-", "-", "+"])}{operand(num(), UNARY)}', NUM, UNARY)),
                (1, lambda: binary(num(), '^', Expr(str(random.randint(0, 3)), NUM, ATOM), NUM, POWER, ATOM, UNARY)),
                (2, lambda: binary(num(), random.choice(['==', '!=', '<', '>', '<=', '>=']), num(), NUM, COMP, COMP, ARITH)),
                (1, lambda: binary(num(), random.choice([tokens.AND, tokens.OR]), num(), NUM, LOGIC, LOGIC, NOT)),
                (1, lambda: Expr(f'{tokens.NOT} {operand(num(), NOT)}', NUM, NOT)),
            ]
        return options

    def arithmetic(self, scope, depth):
        op = random.choice(['+', '-', '*', '/'])
        left = self.expr(NUM, scope, depth)
        if op in '+-':
            return binary(left, op, self.expr(NUM, scope, depth), NUM, ARITH, ARITH, TERM)
        right = self.nonzero(scope, depth) if op == '/' else self.expr(NUM, scope, depth)
        return binary(left, op, right, NUM, TERM, TERM, UNARY)

    def nonzero(self, scope, depth):
        known = [name for name in scope.of_type(NUM)
                 if isinstance(scope.values.get(name), basic.Number) and scope.values[name].value != 0]
        return choose([
            (4, lambda: Expr(num_literal(nonzero=True), NUM, ATOM)),
            (2 if known else 0, lambda: Expr(random.choice(known), NUM, ATOM)),
            (1 if depth > 0 else 0, lambda: self.expr(NUM, scope, depth - 1)),
        ])

    def str_options(self, scope, depth):
        options = [
            (3, lambda: Expr(str_literal(), STR, ATOM)),
            (4, lambda: self.variable(STR, scope)),
        ]
        if depth > 0:
            d = depth - 1
            text = lambda: self.expr(STR, scope, d)
            options += [
                (3, lambda: binary(text(), '+', text(), STR, ARITH, ARITH, TERM)),
                (1, lambda: binary(text(), '*', Expr(str(random.randint(0, 3)), NUM, ATOM), STR, TERM, TERM, UNARY)),
                (0.3, lambda: Expr(f'-{operand(text(), UNARY)}', STR, UNARY)),  # Evaluates to "".
            ]
        return options

    def list_options(self, type_, scope, depth):
        elem = type_[1]
        options = [
            (3, lambda: self.list_literal(elem, scope, depth, random.randint(0, 3))),
            (4, lambda: self.variable(type_, scope)),
        ]
        if depth > 0:
            d = depth - 1
            options += [
                (2, lambda: binary(self.expr(type_, scope, d), '+', self.expr(elem, scope, d), type_, ARITH, ARITH, TERM)),
                (1, lambda: binary(self.expr(type_, scope, d), '*', self.expr(type_, scope, d), type_, TERM, TERM, UNARY)),
                (1, lambda: self.pop_by_minus(type_, scope, d)),
                (2, lambda: self.for_expression(elem, scope, d)),
                (1 if elem == NUM and scope.values else 0, lambda: self.while_expression(scope)),
            ]
        return options

    def list_literal(self, elem, scope, depth, length):
        elements = [self.expr(elem, scope, max(depth - 1, 0)).src for _ in range(length)]
        return Expr(f'[{("," if random.random() < 0.3 else ", ").join(elements)}]', list_of(elem), ATOM)

    def indexable(self, list_type, scope, depth):
        """A list of `list_type` whose length is known, with that length: a top-level variable or a literal."""
        known = [name for name in scope.of_type(list_type) if scope.known_length(name)]
        if known and random.random() < 0.7:
            name = random.choice(known)
            return Expr(name, list_type, ATOM), scope.known_length(name)
        length = random.randint(1, 3)
        return self.list_literal(list_type[1], scope, depth, length), length

    def index(self, elem, scope, depth):
        """`l / i`: the element at index i."""
        source, length = self.indexable(list_of(elem), scope, depth)
        return binary(source, '/', int_literal(random.randint(-length, length - 1)), elem, TERM, TERM, UNARY)

    def pop_by_minus(self, type_, scope, depth):
        """`l - i`: the list without its element at index i."""
        source, length = self.indexable(type_, scope, depth)
        return binary(source, '-', int_literal(random.randint(-length, length - 1)), type_, ARITH, ARITH, TERM)

    def for_expression(self, elem, scope, depth):
        """`for i = a to b then x`: the list of each iteration's x."""
        variable = self.fresh_name(scope)
        body_scope = scope.child(in_loop=True)
        body_scope.variables[variable] = NUM
        body = self.expr(elem, body_scope, depth)
        return Expr(f'{tokens.FOR} {variable} = {self.loop_bounds(scope)} then {operand(body, LOGIC)}',
                    list_of(elem), GREEDY)

    def while_expression(self, scope):
        """`while n < k then var n = n + 1`: the list of n's values. Top level only, where n's value is known."""
        lines = self.while_statement(scope, 0, 0, top_level=True, inline=True)
        return Expr(lines[0], list_of(NUM), GREEDY) if lines else None

    def inline_if(self, type_, scope, depth):
        branches = [f'{tokens.IF} {operand(self.condition(scope, depth), LOGIC)} then {operand(self.expr(type_, scope, depth), LOGIC)}']
        if random.random() < 0.3:
            branches.append(f'{tokens.ELIF} {operand(self.condition(scope, depth), LOGIC)} then {operand(self.expr(type_, scope, depth), LOGIC)}')
        # Without an else an inline if can evaluate to null, which is only a NUM.
        if type_ != NUM or random.random() < 0.7:
            branches.append(f'{tokens.ELSE} {operand(self.expr(type_, scope, depth), LOGIC)}')
        return Expr(' '.join(branches), type_, GREEDY)

    def condition(self, scope, depth):
        """Something `is_true` is defined for: a number (usually a comparison) or a string."""
        if random.random() < 0.15:
            return self.expr(STR, scope, depth)
        if depth > 0 and random.random() < 0.6:
            num = lambda: self.expr(NUM, scope, depth - 1)
            return binary(num(), random.choice(['==', '!=', '<', '>', '<=', '>=']), num(), NUM, COMP, COMP, ARITH)
        return self.expr(NUM, scope, depth)

    def anonymous_function(self, scope, depth, ret=None):
        params = [self.fresh_name(scope) for _ in range(random.choice([0, 1, 1, 2]))]
        param_types = [random.choice([NUM, NUM, STR]) for _ in params]
        ret = ret or random_type()
        body_scope = Scope({**{c: NUM for c in CONSTANTS}, **dict(zip(params, param_types))},
                           dict(scope.functions), return_type=ret)
        body = self.expr(ret, body_scope, max(depth - 1, 0))
        return Expr(f'{tokens.FUN} ({", ".join(params)}) -> {body.src}', fun_of(param_types, ret), GREEDY)

    def call(self, type_, scope, depth):
        """A call returning `type_`: user functions, templates, built-ins or `(def ...)(args)`."""
        def args_for(params, divisors=()):
            return [(self.nonzero(scope, depth) if i in divisors else self.expr(p, scope, depth)) for i, p in enumerate(params)]

        def named(name, params, divisors=()):
            args = args_for(params, divisors)
            if len(args) == 2 and random.random() < 0.4:  # Infix: `a name b`.
                return binary(args[0], name, args[1], type_, TERM, TERM, UNARY)
            return Expr(call_text(name, [operand(a, GREEDY) for a in args]), type_, ATOM)

        options = []
        for name, (_, params, ret) in scope.functions.items():
            if matches(ret, type_):
                options.append((3, lambda name=name, params=params: named(name, params)))
        if type_ == NUM:
            options.append((4, lambda: named(*self.template())))
            options.append((1, lambda: Expr(call_text('len', [operand(self.expr(list_of(random.choice([NUM, STR])), scope, depth), GREEDY)]), NUM, ATOM)))
            options.append((1, lambda: Expr(call_text(random.choice(['is_num', 'is_str', 'is_list', 'is_fun']), [self.any_value(scope, depth)]), NUM, ATOM)))
        if type_ == STR:
            options.append((1, lambda: Expr(call_text('print_ret', [self.any_value(scope, depth)]), STR, ATOM)))
        options.append((0.5, lambda: self.immediate_call(type_, scope, depth)))
        lists = [name for name in scope.assignable(list_of(type_)) if scope.known_length(name)]
        if lists:
            name = random.choice(lists)
            length = scope.known_length(name)
            options.append((0.5, lambda: Expr(call_text('pop', [name, int_literal(random.randint(-length, length - 1)).src]), type_, ATOM)))
        return choose(options)

    def template(self):
        name = random.choice(list(TEMPLATES))
        (_, params, _), divisors = TEMPLATES[name]
        return name, params, divisors

    def immediate_call(self, type_, scope, depth):
        """`(def (a) -> ...)(x)`."""
        function = self.anonymous_function(scope, depth, ret=type_)
        args = [operand(self.expr(p, scope, depth), GREEDY) for p in function.type[1]]
        return Expr(call_text(f'({function.src})', args), type_, ATOM)

    def any_value(self, scope, depth):
        if scope.functions and random.random() < 0.1:
            return random.choice(list(scope.functions))
        return operand(self.expr(random_type(), scope, depth), GREEDY)


def generate_program(texts):
    """2-5 valid statements. `texts` holds every statement used so far, so none repeats."""
    generator = ProgramGenerator(texts)
    for _ in range(random.randint(2, 5)):
        generator.add_statement()
    return [text for text, _ in generator.accepted]


########################
# FEATURES
########################

def ast_features(node):
    """Node types, operators and block/inline forms present in an AST."""
    features = {type(node).__name__}
    if isinstance(node, (basic.BinOpNode, basic.UnaryOpNode)):
        op = node.op_tok
        features.add(f'{type(node).__name__} {op.type}:{op.value}' if op.type == tokens.KEYWORD else f'{type(node).__name__} {op.type}')
    if isinstance(node, basic.ListNode):
        features.add('list literal')
    if isinstance(node, basic.IfNode):
        features |= {f'if {"block" if block else "inline"}' for _, _, block in node.cases}
        features |= {'elif'} if len(node.cases) > 1 else set()
        features |= {'else'} if node.else_case else set()
    if isinstance(node, (basic.ForNode, basic.WhileNode)):
        features.add(f'{type(node).__name__} {"block" if node.is_block else "inline"}')
    if isinstance(node, basic.ForNode) and node.step_value_node:
        features.add('for step')
    if isinstance(node, basic.FuncDefNode):
        features.add(f'def {"block" if node.is_block else "inline"}')
        features.add('def named' if node.var_name_tok else 'def anonymous')
    if isinstance(node, basic.ReturnNode):
        features.add('return value' if node.node_to_return else 'return bare')

    _map_child_nodes(node, lambda child: features.update(ast_features(child)) or child)
    return features


# Every feature `ast_features` can report; a generator covering the language produces them all.
LANGUAGE_FEATURES = {
    'NumberNode', 'StringNode', 'list literal', 'VarAccessNode', 'VarAssignNode', 'CallNode',
    'ContinueNode', 'BreakNode', 'return value', 'return bare',
    *(f'BinOpNode {op}' for op in ['PLUS', 'MINUS', 'MUL', 'DIV', 'POW', 'EE', 'NE', 'LT', 'GT', 'LTE', 'GTE', 'IDENTIFIER']),
    'BinOpNode KEYWORD:and', 'BinOpNode KEYWORD:or', 'UnaryOpNode MINUS', 'UnaryOpNode PLUS', 'UnaryOpNode KEYWORD:not',
    'if inline', 'if block', 'elif', 'else', 'ForNode inline', 'ForNode block', 'for step',
    'WhileNode inline', 'WhileNode block', 'def inline', 'def block', 'def named', 'def anonymous',
}


########################
# SAMPLES
########################

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


def build_sample(statements, idx):
    """The Sample for one program, or None if a statement doesn't lex, parse, run or fit
    BLOCK_SIZE. The generator already checks all of that, so None means a bug."""
    sample = Sample(statements, idx)
    context = new_context()

    for text in statements:
        token_list, error = basic.Lexer('<stdin>', text).make_tokens()
        if error:
            return None
        ast = basic.Parser(token_list).parse()
        if ast.error:
            return None
        res = basic.Interpreter(max_steps=MAX_INTERPRETER_STEPS).visit(ast.node, context)
        if res.error:
            return None

        lexer_text = ' '.join(repr(t) for t in token_list)
        sample.lexer_text += f'\n{lexer_text}'
        if random.random() < 0.5:  # Half the time the model reads it without `var`, so it learns to restore it.
            token_list, error = basic.Lexer('<stdin>', text.replace('var ', '')).make_tokens()
            if error:
                return None
            lexer_text = ' '.join(repr(t) for t in token_list)
        ast_text = f'{tokens.SOF} {inline_templates(ast.node)} {tokens.EOF}'
        sample.ast_text += f'\n{ast_text}'

        lex_encoded, ast_encoded = data.encode(lexer_text, {}), data.encode(ast_text, {})
        if len(lex_encoded) > BLOCK_SIZE or len(ast_encoded) > BLOCK_SIZE:
            return None
        # One token more than BLOCK_SIZE, so get_batch can shift it into the input and the target.
        sample.x_in.append(data.add_pad_tokens_and_trim(lex_encoded, BLOCK_SIZE + 1))
        sample.x_out.append(data.add_pad_tokens_and_trim(ast_encoded, BLOCK_SIZE + 1))
        sample.symbols |= context.symbol_table.symbols
        if res.value:
            sample.symbols['_output_list'].append(res.value)
    return sample


def build_samples(count):
    """`count` generated programs as sample rows, and how many failed to build (should be 0)."""
    rows, failed, texts = [], 0, set()
    for idx in range(count):
        statements = generate_program(texts)
        if idx % 1_000 == 0:
            print_program(statements)
            print(f'loaded: {idx / count * 100:.2f}%')
        sample = build_sample(statements, idx)
        if sample is None:
            failed += 1
        else:
            rows.append(sample.__dict__)
    return rows, failed


def print_stats():
    drawn = sum(n for key, n in STATS.items() if key.startswith('drawn'))
    kept = sum(n for key, n in STATS.items() if key.startswith('kept'))
    print(f'\nStatements drawn: {drawn}, kept: {kept} ({kept / max(drawn, 1) * 100:.1f}%)')
    for kind in [*ProgramGenerator.KINDS, 'setup']:
        print(f'  {kind}: kept {STATS[f"kept {kind}"]} of {STATS[f"drawn {kind}"]}')
    for key, n in sorted(STATS.items(), key=lambda kv: -kv[1]):
        if key.startswith('rejected'):
            print(f'  {key}: {n}')


def generate():
    """Build NUM_TRAINING_SAMPLES samples, shuffled, and pickle them to DATASET_NAME."""
    rows, failed = build_samples(NUM_TRAINING_SAMPLES)
    samples_df = pd.DataFrame(rows)
    samples_df = samples_df.sample(frac=1, random_state=random.randint(0, 2**31 - 1)).reset_index(drop=True)
    samples_df.to_pickle(DATASET_NAME)
    print(f'\nSample generation: Valid: {len(samples_df)}, Failed: {failed}, '
          f'Success rate: {len(samples_df) / NUM_TRAINING_SAMPLES * 100:.1f}%')
    print_stats()


if __name__ == '__main__':
    generate()
