from . import basic
from . import data
from . import tokens
from .basic import EOF


def run_statements(text):
    """Run `text` and return (per-statement values, error).

    Since `statements` became the parser entry point, `basic.run` always returns a
    `List` wrapping one value per top-level statement.
    """
    value, error = basic.run('<stdin>', text)
    if error: return None, error
    assert isinstance(value, basic.List)
    return value.elements, None


def run_single_statement(text):
    """Run `text`, assert it produced exactly one statement, return (value, error)."""
    elements, error = run_statements(text)
    if error: return None, error
    assert len(elements) == 1
    return elements[0], None


def test_lexing_float_plus_int():
    """Test lexing the input '3.4+2'"""
    lexer = basic.Lexer('<stdin>', "3.4+2")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce SOF, FLOAT, PLUS, INT, EOF
    assert len(tokens) == 5
    assert tokens[0].type == basic.SOF
    assert tokens[1].type == basic.FLOAT
    assert tokens[1].value == 3.4
    assert tokens[2].type == basic.PLUS
    assert tokens[3].type == basic.INT
    assert tokens[3].value == 2
    assert tokens[4].type == EOF


def test_lexing_float_multiply_float():
    """Test lexing the input '2.5 * 2.5'"""
    lexer = basic.Lexer('<stdin>', "2.5 * 2.5")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce SOF, FLOAT, MUL, FLOAT, EOF
    assert len(tokens) == 5
    assert tokens[0].type == basic.SOF
    assert tokens[1].type == basic.FLOAT
    assert tokens[1].value == 2.5
    assert tokens[2].type == basic.MUL
    assert tokens[3].type == basic.FLOAT
    assert tokens[3].value == 2.5
    assert tokens[4].type == EOF


def test_lexing_int_plus_int():
    """Test lexing the input '1 + 2'"""
    lexer = basic.Lexer('<stdin>', "1 + 2")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce SOF, INT, PLUS, INT, EOF
    assert len(tokens) == 5
    assert tokens[0].type == basic.SOF
    assert tokens[1].type == basic.INT
    assert tokens[1].value == 1
    assert tokens[2].type == basic.PLUS
    assert tokens[3].type == basic.INT
    assert tokens[3].value == 2
    assert tokens[4].type == EOF


def test_lexing_illegal_char():
    """Test lexing the input '1 + d' where 'd' is now a valid identifier"""
    lexer = basic.Lexer('<stdin>', "1 + d")
    tokens, error = lexer.make_tokens()

    assert error is None
    assert len(tokens) == 5
    assert tokens[0].type == basic.SOF
    assert tokens[1].type == basic.INT
    assert tokens[2].type == basic.PLUS
    assert tokens[3].type == basic.IDENTIFIER
    assert tokens[4].type == EOF


def test_parsing_syntax_error_missing_operand():
    """Test parsing with a syntax error: missing operand after operator"""
    ast, error = basic.run('<stdin>', "1 +")
    
    assert error is not None
    assert isinstance(error, basic.InvalidSyntaxError)
    assert error.error_name == 'Illegal Syntax'
    assert ast is None


def test_parsing_comprehensive_valid_ast():
    """Test parsing a complex expression that covers all main features:
    - Integers and floats
    - All operators: +, -, *, /
    - Operator precedence (multiplication/division before addition/subtraction)
    """
    # Expression: 10 + 2.5 * 3 - 4.2 / 2
    # This tests: addition, multiplication, subtraction, division
    # with both integers and floats, and proper operator precedence
    # Expected AST structure: (10 + (2.5 * 3)) - (4.2 / 2)
    value, error = run_single_statement("10 + 2.5 * 3 - 4.2 / 2")

    assert error is None
    assert isinstance(value, basic.Number)


def test_parsing_unary_minus():
    """Test parsing unary minus operator: -5"""
    value, error = run_single_statement("-5")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == -5


def test_parsing_unary_plus():
    """Test parsing unary plus operator: +3.5"""
    value, error = run_single_statement("+3.5")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 3.5


def test_parsing_parentheses():
    """Test parsing parentheses for grouping: (1 + 2) * 3"""
    value, error = run_single_statement("(1 + 2) * 3")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 9


def test_parsing_unary_with_parentheses():
    """Test parsing unary operator with parentheses: -(1 + 2)"""
    value, error = run_single_statement("-(1 + 2)")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == -3


def test_stupidly_simple_not():
    ast, error = basic.run('<stdin>', f"{tokens.NOT} {tokens.TRUE} == {tokens.NULL}")
    assert error is None


def test_function_def_and_calls():
    # def f(a,b) -> a+b
    value, error = run_single_statement("def f(a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # f(8,9)
    value, error = run_single_statement("f(8,9)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 17

    # no args, call
    value, error = run_single_statement("f()")
    assert error is not None
    assert isinstance(error, basic.RTError)

    # call
    value, error = run_single_statement("f(3,4,5)")
    assert error is not None
    assert isinstance(error, basic.RTError)

    # var func = f
    value, error = run_single_statement("var func = f")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # func
    value, error = run_single_statement("func")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # func(2,3)
    value, error = run_single_statement("func(2,3)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 5

    # def (a, b) -> a + b
    value, error = run_single_statement("def (a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "<anonymous>"

    # var ano = def (a, b) -> a + b
    value, error = run_single_statement("var ano = def (a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)

    # ano(3,3)
    value, error = run_single_statement("ano(3,3)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 6

    # def zero(a) -> a/0
    value, error = run_single_statement("def zero(a) -> a/0")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "zero"

    # zero(9)
    value, error = run_single_statement("zero(9)")
    assert error is not None
    assert isinstance(error, basic.RTError)


def test_learned_infix_operators():
    """Infix template ops (e.g. 8 times 8); meaning from data_generator templates + symbol table only."""
    from . import data_generator as dg

    text = '7+8 times 8'
    lexer = basic.Lexer('<stdin>', text)
    tokens, err = lexer.make_tokens()
    assert err is None
    ast = basic.Parser(tokens).parse()
    assert ast.error is None

    context = basic.Context('<t>')
    context.symbol_table = basic.get_symbol_table()
    dg._load_template_functions_into_context(context)
    res = basic.Interpreter().visit(ast.node, context)
    assert res.error is None
    assert res.value.value == 7 + 64


def test_arithmetic_styles():
    # Std function
    res, _ = basic.run_ai('<stdin>', "sum(3,4)")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 7

    # Missing parenthesis
    res, _ = basic.run_ai('<stdin>', "mul(8 8)")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 64

    # Infix functions
    res, _ = basic.run_ai('<stdin>', "4 plus 4")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 8

    # Infix with spaces
    res, _ = basic.run_ai('<stdin>', "3     times    3")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 9

    # Calling method
    res, _ = basic.run_ai('<stdin>', "3.sum(4)")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 7


    # Calling method, missing parent
    res, _ = basic.run_ai('<stdin>', "3.times 4")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 12

    # Calculator style
    res, _ = basic.run_ai('<stdin>', "3 + 4")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 7

    # Calculator style 2
    res, _ = basic.run_ai('<stdin>', "3 * 4")
    assert res.error is None
    assert isinstance(res.value, basic.Number)
    assert res.value.value == 12


def test_power_operator():
    """Test the power operator: 2 ^ 3"""
    value, error = run_single_statement("2 ^ 3")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 8


def test_power_to_zero():
    """Test the power operator with exponent 0: 5 ^ 0"""
    value, error = run_single_statement("5 ^ 0")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 1


def test_power_is_right_associative():
    """Test that power is right-associative: 2 ^ 3 ^ 2 == 2 ^ (3 ^ 2) == 512"""
    value, error = run_single_statement("2 ^ 3 ^ 2")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 512


def test_string_literal():
    """Test a simple string literal: "hello\""""
    value, error = run_single_statement('"hello"')

    assert error is None
    assert isinstance(value, basic.String)
    assert value.value == "hello"


def test_string_concatenation():
    """Test concatenating two strings: "foo" + "bar\""""
    value, error = run_single_statement('"foo" + "bar"')

    assert error is None
    assert isinstance(value, basic.String)
    assert value.value == "foobar"


def test_string_repetition():
    """Test repeating a string with a number: "ab" * 3"""
    value, error = run_single_statement('"ab" * 3')

    assert error is None
    assert isinstance(value, basic.String)
    assert value.value == "ababab"


def test_string_escape_chars():
    """Test that escape sequences are handled during lexing: "a\\nb\""""
    lexer = basic.Lexer('<stdin>', '"a\\nb"')
    tokens, error = lexer.make_tokens()

    assert error is None
    # Should produce SOF, STRING, EOF
    assert tokens[1].type == basic.STRING
    assert tokens[1].value == "a\nb"


########################
# BUILT-IN FUNCTIONS
########################

def test_list_literal():
    """Test a list literal evaluates to a List with the right elements: [1, 2, 3]"""
    value, error = run_single_statement("[1, 2, 3]")

    assert error is None
    assert isinstance(value, basic.List)
    assert [el.value for el in value.elements] == [1, 2, 3]


def test_is_number_builtin():
    """is_num returns true for numbers and false otherwise."""
    value, error = run_single_statement("is_num(5)")
    assert error is None
    assert value.value == 1

    value, error = run_single_statement('is_num("hi")')
    assert error is None
    assert value.value == 0


def test_is_string_builtin():
    """is_str returns true for strings and false otherwise."""
    value, error = run_single_statement('is_str("hi")')
    assert error is None
    assert value.value == 1

    value, error = run_single_statement("is_str(5)")
    assert error is None
    assert value.value == 0


def test_is_list_builtin():
    """is_list returns true for lists and false otherwise."""
    value, error = run_single_statement("is_list([1, 2])")
    assert error is None
    assert value.value == 1

    value, error = run_single_statement("is_list(5)")
    assert error is None
    assert value.value == 0


def test_is_function_builtin():
    """is_fun returns true for functions (including built-ins) and false otherwise."""
    value, error = run_single_statement("is_fun(pop)")
    assert error is None
    assert value.value == 1

    value, error = run_single_statement("is_fun(5)")
    assert error is None
    assert value.value == 0


def test_append_builtin():
    """append mutates the list in place and returns null."""
    value, error = run_single_statement("var append_list = [1, 2]")
    assert error is None

    value, error = run_single_statement("append(append_list, 3)")
    assert error is None
    assert value.value == basic.Number.null.value

    value, error = run_single_statement("append_list")
    assert error is None
    assert [el.value for el in value.elements] == [1, 2, 3]


def test_append_non_list_errors():
    """append on a non-list first argument raises a runtime error."""
    value, error = run_single_statement("append(5, 3)")
    assert error is not None
    assert isinstance(error, basic.RTError)


def test_pop_builtin():
    """pop removes and returns the element at the given index."""
    value, error = run_single_statement("var pop_list = [10, 20, 30]")
    assert error is None

    value, error = run_single_statement("pop(pop_list, 1)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 20

    value, error = run_single_statement("pop_list")
    assert error is None
    assert [el.value for el in value.elements] == [10, 30]


def test_pop_out_of_range_errors():
    """pop with an out-of-range index raises a runtime error."""
    value, error = run_single_statement("pop([1, 2], 5)")
    assert error is not None
    assert isinstance(error, basic.RTError)


def test_extend_builtin():
    """extend appends all elements of the second list onto the first and returns null."""
    value, error = run_single_statement("var extend_list = [1, 2]")
    assert error is None

    value, error = run_single_statement("extend(extend_list, [3, 4])")
    assert error is None
    assert value.value == basic.Number.null.value

    value, error = run_single_statement("extend_list")
    assert error is None
    assert [el.value for el in value.elements] == [1, 2, 3, 4]


def test_extend_non_list_errors():
    """extend with a non-list second argument raises a runtime error."""
    value, error = run_single_statement("extend([1, 2], 3)")
    assert error is not None
    assert isinstance(error, basic.RTError)


########################
# MULTILINE / MULTI-STATEMENT
########################

def test_lexing_newline_char():
    """A '\\n' between expressions lexes to a NEWLINE token."""
    lexer = basic.Lexer('<stdin>', "1\n2")
    token_list, error = lexer.make_tokens()

    assert error is None
    # SOF, INT, NEWLINE, INT, EOF
    assert [t.type for t in token_list] == [
        basic.SOF, basic.INT, tokens.NEWLINE, basic.INT, EOF]


def test_lexing_semicolon_is_a_newline():
    """';' is a statement separator and lexes to the same NEWLINE token as '\\n'."""
    lexer = basic.Lexer('<stdin>', "1;2")
    token_list, error = lexer.make_tokens()

    assert error is None
    assert [t.type for t in token_list] == [
        basic.SOF, basic.INT, tokens.NEWLINE, basic.INT, EOF]


def test_lexing_end_is_a_keyword():
    """'end' closes a block, so it must lex as KEYWORD and not as an IDENTIFIER."""
    lexer = basic.Lexer('<stdin>', "end")
    token_list, error = lexer.make_tokens()

    assert error is None
    assert token_list[1].type == basic.KEYWORD
    assert token_list[1].value == tokens.END


def test_two_statements_separated_by_semicolon():
    """Each top-level statement contributes one value to the result list."""
    elements, error = run_statements("1 + 1; 2 + 2")

    assert error is None
    assert [el.value for el in elements] == [2, 4]


def test_two_statements_separated_by_newline():
    """'\\n' separates statements exactly like ';'."""
    elements, error = run_statements("1 + 1\n2 + 2")

    assert error is None
    assert [el.value for el in elements] == [2, 4]


def test_blank_lines_are_ignored():
    """Leading, trailing and repeated newlines produce no extra statements."""
    elements, error = run_statements("\n\n1 + 1\n\n\n2 + 2\n\n")

    assert error is None
    assert [el.value for el in elements] == [2, 4]


def test_single_statement_is_still_wrapped_in_a_list():
    """Even a lone expression comes back wrapped, with its value as the only element."""
    value, error = run_single_statement("40 + 2")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 42


def test_later_statement_sees_earlier_assignment():
    """Statements share one symbol table, so a var is visible to the statements after it."""
    elements, error = run_statements("var multi_a = 5; multi_a * 2")

    assert error is None
    assert [el.value for el in elements] == [5, 10]


def test_inline_if_returns_its_value():
    """The single-line `if ... then ... else ...` form stays assignable to a variable."""
    value, error = run_single_statement("if 1 then 10 else 20")

    assert error is None
    assert value.value == 10


def test_multiline_if_returns_null():
    """A block-bodied if is a statement, not an expression: it evaluates to null."""
    value, error = run_single_statement("if 1 then\n10\nend")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == basic.Number.null.value


def test_multiline_if_executes_its_body():
    """The block body really runs — observed through a var it assigns."""
    elements, error = run_statements(
        "var multi_if = 0\nif 1 then\nvar multi_if = 99\nend\nmulti_if")

    assert error is None
    assert elements[-1].value == 99


def test_multiline_if_takes_elif_branch():
    """An elif block runs when its condition is the first true one."""
    elements, error = run_statements(
        "var multi_elif = 0\n"
        "if 0 then\nvar multi_elif = 1\n"
        "elif 1 then\nvar multi_elif = 2\n"
        "end\nmulti_elif")

    assert error is None
    assert elements[-1].value == 2


def test_multiline_if_takes_else_branch():
    """The else block runs when no condition matched."""
    elements, error = run_statements(
        "var multi_else = 0\n"
        "if 0 then\nvar multi_else = 1\n"
        "else\nvar multi_else = 3\n"
        "end\nmulti_else")

    assert error is None
    assert elements[-1].value == 3


def test_multiline_if_without_end_errors():
    """A block-bodied if that is never closed is a syntax error."""
    value, error = basic.run('<stdin>', "if 1 then\n10")

    assert error is not None
    assert isinstance(error, basic.InvalidSyntaxError)


def test_inline_for_returns_list_of_body_values():
    """The single-line for form collects one value per iteration."""
    value, error = run_single_statement("for multi_i = 0 to 3 then multi_i")

    assert error is None
    assert isinstance(value, basic.List)
    assert [el.value for el in value.elements] == [0, 1, 2]


def test_multiline_for_returns_null():
    """A block-bodied for discards the per-iteration values and returns null."""
    value, error = run_single_statement("for multi_j = 0 to 3 then\nmulti_j\nend")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == basic.Number.null.value


def test_multiline_for_executes_its_body():
    """Each iteration of the block body runs, appending to a list defined outside it."""
    elements, error = run_statements(
        "var for_acc = []\nfor multi_k = 0 to 3 then\nappend(for_acc, multi_k)\nend\nfor_acc")

    assert error is None
    assert [el.value for el in elements[-1].elements] == [0, 1, 2]


def test_inline_while_returns_list_of_body_values():
    """The single-line while form collects one value per iteration."""
    elements, error = run_statements("var while_a = 0\nwhile while_a < 3 then var while_a = while_a + 1")

    assert error is None
    assert isinstance(elements[-1], basic.List)
    assert [el.value for el in elements[-1].elements] == [1, 2, 3]


def test_multiline_while_returns_null():
    """A block-bodied while returns null rather than the collected values."""
    elements, error = run_statements("var while_b = 0\nwhile while_b < 3 then\nvar while_b = while_b + 1\nend")

    assert error is None
    assert isinstance(elements[-1], basic.Number)
    assert elements[-1].value == basic.Number.null.value


def test_multiline_while_executes_its_body():
    """The loop really iterates until the condition goes false."""
    elements, error = run_statements(
        "var while_c = 0\nwhile while_c < 3 then\nvar while_c = while_c + 1\nend\nwhile_c")

    assert error is None
    assert elements[-1].value == 3


def test_inline_def_returns_body_value_when_called():
    """An arrow-bodied function still returns the value of its body expression."""
    elements, error = run_statements("def multi_g(x) -> x * 2\nmulti_g(4)")

    assert error is None
    assert isinstance(elements[0], basic.Function)
    assert elements[1].value == 8


def test_multiline_def_returns_null_when_called():
    """A block-bodied function returns null; its body value is not propagated."""
    elements, error = run_statements("def multi_h(x)\nx * 2\nend\nmulti_h(4)")

    assert error is None
    assert isinstance(elements[0], basic.Function)
    assert isinstance(elements[1], basic.Number)
    assert elements[1].value == basic.Number.null.value


def test_multiline_def_executes_its_body():
    """The block body runs even though the call evaluates to null."""
    elements, error = run_statements(
        "var def_acc = []\ndef multi_p(x)\nappend(def_acc, x)\nend\nmulti_p(7)\ndef_acc")

    assert error is None
    assert [el.value for el in elements[-1].elements] == [7]


def test_multiline_def_without_end_errors():
    """A block-bodied function that is never closed is a syntax error."""
    value, error = basic.run('<stdin>', "def multi_q(x)\nx * 2")

    assert error is not None
    assert isinstance(error, basic.InvalidSyntaxError)


def test_def_with_neither_arrow_nor_newline_errors():
    """A function header followed by something that is not '->' or a newline is an error."""
    value, error = basic.run('<stdin>', "def multi_r(x) x * 2")

    assert error is not None
    assert isinstance(error, basic.InvalidSyntaxError)


def test_parser_reverse_rewinds_the_token_index():
    """`statements` relies on reverse() to back out of a failed lookahead, so it must go backwards."""
    lexer = basic.Lexer('<stdin>', "1 + 2")
    token_list, error = lexer.make_tokens()
    assert error is None

    parser = basic.Parser(token_list)
    start_idx = parser.tok_idx
    parser.advance()
    parser.advance()
    assert parser.tok_idx == start_idx + 2

    parser.reverse(2)
    assert parser.tok_idx == start_idx
    assert parser.current_tok is token_list[start_idx]


########################
# TOKEN COMPRESSION
########################

def test_get_compressed_tokens_passes_through_plain_text():
    """Text containing no token name comes back as its raw utf-8 byte values."""
    assert data.get_compressed_tokens("a b") == [97, 32, 98]


def test_get_compressed_tokens_maps_a_token_name_to_its_id():
    """A token name is collapsed into the single vocab id reserved for it."""
    assert data.get_compressed_tokens(tokens.SOF) == [tokens.TOKEN_IDS[tokens.SOF]]


def test_get_compressed_tokens_preserves_newline():
    """NEWLINE must compress to its own id rather than be eaten by the shorter NE.

    Substitution walks TOKENS in order and uses a plain string replace, and NE's byte
    pattern '78,69' sits at both ends of NEWLINE's '78,69,87,76,73,78,69'. NE comes
    first in TOKENS, so it claims both ends and NEWLINE never matches.
    """
    assert data.get_compressed_tokens(tokens.NEWLINE) == [tokens.TOKEN_IDS[tokens.NEWLINE]]


def test_get_compressed_tokens_round_trips_every_token():
    """No token may be shadowed by another whose byte pattern is a substring of it."""
    shadowed = {
        name: data.get_compressed_tokens(name)
        for name in tokens.TOKENS
        if data.get_compressed_tokens(name) != [tokens.TOKEN_IDS[name]]
    }

    assert not shadowed, f"tokens not compressing to their own id: {shadowed}"


def test_get_compressed_tokens_keeps_newlines_of_a_lexed_program():
    """Each statement separator in a lexed program survives as one NEWLINE id."""
    token_list, error = basic.Lexer('<stdin>', "1\n2;3").make_tokens()
    assert error is None

    lex_text = ' '.join(tok.__repr__() for tok in token_list)
    ids = data.get_compressed_tokens(lex_text)

    assert ids.count(tokens.TOKEN_IDS[tokens.NEWLINE]) == 2
