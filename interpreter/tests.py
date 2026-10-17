import pytest

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


def test_parsing_reports_error_in_later_semicolon_separated_statement():
    source = "var a=[];var i = 0;while i<10 then;var i=i+1;if i=4 then continue;if i==8 then break;var a=a+i;end"

    ast, error = basic.run('<stdin>', source)

    assert ast is None
    assert isinstance(error, basic.InvalidSyntaxError)
    assert error.pos_start.idx == source.index('=', source.index('if i=4'))


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


@pytest.mark.parametrize('source', ["5(1)", '"text"()', "[1](0)"])
def test_calling_a_non_function_is_a_runtime_error(source):
    value, error = basic.run('<stdin>', source)

    assert isinstance(error, basic.RTError)
    assert error.details == 'Illegal operation'


def test_learned_infix_operators():
    """Infix template ops (e.g. 8 times 8); meaning from the templates + symbol table only."""
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
    # Parsing enters at `statements`, so the result is a List holding one statement value.
    assert [el.value for el in res.value.elements] == [7 + 64]


def test_arithmetic_styles():
    def last(res):
        assert res.error is None
        return res.value.elements[-1]

    # Std function
    number = last(basic.run_ai('<stdin>', "sum(3,4)")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 7

    # Missing parenthesis
    number = last(basic.run_ai('<stdin>', "2 pluss 2")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 4

    # Infix functions
    number = last(basic.run_ai('<stdin>', "4 plus 4")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 8

    # Infix with spaces
    number = last(basic.run_ai('<stdin>', "3     times    3")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 9

    # Calling method
    number = last(basic.run_ai('<stdin>', "3.sum 4 ")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 7

    # Calling method, missing parent
    number = last(basic.run_ai('<stdin>', "3.times 4")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 12

    # Calculator style
    number = last(basic.run_ai('<stdin>', "3 + 4")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 7

    # Calculator style 2
    number = last(basic.run_ai('<stdin>', "3 * 4")[0])
    assert isinstance(number, basic.Number)
    assert number.value == 12


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


@pytest.mark.parametrize('operation, result', [
    ("l + 3", "[1, 2, 3]"),  # append
    ("l * [3]", "[1, 2, 3]"),  # extend
    ("l - 0", "[2]"),  # remove at index
])
def test_list_operators_leave_their_operand_unchanged(operation, result):
    """`+`, `*` and `-` build a new list: `var m = l + 3` must not also append to `l`."""
    elements, error = run_statements(f"var l = [1, 2]; var m = {operation}; l")

    assert error is None
    assert repr(elements[1]) == result
    assert repr(elements[2]) == "[1, 2]"


def test_list_builtins_still_change_the_list_in_place():
    """Unlike the operators, append/extend/pop mutate the list the variable holds."""
    elements, error = run_statements("var l = [1, 2]; append(l, 3); extend(l, [4]); pop(l, 0); l")

    assert error is None
    assert repr(elements[-1]) == "[2, 3, 4]"


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


def test_inline_while_control_flow_does_not_add_nulls_to_accumulator():
    source = (
        "var a=[];var i = 0;while i<20 then;var i=i+1;"
        "if i==4 then continue;if i==8 then break;var a=a+i;end;a"
    )

    elements, error = run_statements(source)

    assert error is None
    assert [el.value for el in elements[-1].elements] == [1, 2, 3, 5, 6, 7]


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


########################
# AST TEXT ROUND TRIP
########################

# One program per shape the AST text has to carry, including each block-vs-inline variant,
# since that flag changes what if/for/while/def evaluate to.
ROUND_TRIP_PROGRAMS = [
    # Arithmetic, logic and assignment: the shapes the first generator emitted.
    "3 + 4.5 * -2",
    "var a = (1 - 2) / 3 ^ 2",
    "not True == None and 1 <= 2 or 3 != 4",
    "def times(a, b) -> a * b; 7 times 8",
    # Calls
    "def f(a, b) -> a + b; f(8, 9)",
    "def g() -> 1; g()",
    "(def (a) -> a * 2)(4)",
    "var h = def (x) -> x; h(h(1))",
    # Strings: spaces, brackets, quotes, colons and newlines must not split the AST text.
    '"plain"',
    r'"spaces (parens) [brackets] \"quotes\" back\\slash: colon\nnewline" + "x"',
    "'single' * 3",
    # Lists: a one-element list literal must not collapse into a lone statement.
    "[]",
    "[7]",
    '[1, [2, [3]], "s"]',
    "var l = [1, 2]; l / 0",
    # If
    "if 1 then 2",
    "if 0 then 2 else 3",
    "if 0 then 1 elif 1 then 2 else 3",
    "var r = if 0 then 1 elif 0 then 2",
    "if 1 then\nvar t = 1\nend",
    "if 0 then\n1\nelif 1 then\nvar t = 2\nvar u = 3\nelse\n4\nend",
    "if 1 then 5 else\nvar t = 6\nend",
    # For
    "for i = 0 to 5 then i * 2",
    "for i = 10 to 0 step -2 then i",
    "var s = 0; for i = 0 to 4 then\nvar s = s + i\nend",
    "for i = 0 to 9 then\nif i == 2 then continue\nif i == 5 then break\nvar last = i\nend",
    # While
    "var n = 0; while n < 3 then var n = n + 1",
    "var n = 0; while 1 then\nvar n = n + 1\nif n > 4 then break\nend",
    # Block-bodied defs and return
    "def k(x)\nif x > 0 then return x\nreturn 0 - x\nend; k(-3)",
    "def nothing()\nreturn\nend; nothing()",
    "def one()\n1\nend; one()",
    "def first()\nfor i = 0 to 10 then\nif i == 3 then return i\nend\nend; first()",
]


def parse(source):
    token_list, error = basic.Lexer('<stdin>', source).make_tokens()
    assert error is None
    ast = basic.Parser(token_list).parse()
    assert ast.error is None, ast.error.as_string()
    return ast.node


def ast_shape(value):
    """Structure of an AST with source positions dropped, so two trees compare by content."""
    if isinstance(value, basic.Token):
        return value.type, value.value
    if isinstance(value, (list, tuple)):
        return tuple(ast_shape(v) for v in value)
    if hasattr(value, '__dict__'):
        fields = sorted((k, v) for k, v in vars(value).items() if k not in ('pos_start', 'pos_end'))
        return (type(value).__name__,) + tuple((k, ast_shape(v)) for k, v in fields)
    return value


def as_program(node):
    """Wrap a rebuilt node the way `basic.inference` does, unless it already is the statements."""
    if isinstance(node, basic.BlockNode):
        return node
    return basic.BlockNode([node], node.pos_start, node.pos_end)


def execute(node):
    res, context = basic.run_interpreter(basic.get_symbol_table(), as_program(node))
    symbols = {name: repr(value) for name, value in context.symbol_table.symbols.items()}
    return res.error, repr(res.value), symbols


@pytest.mark.parametrize('source', ROUND_TRIP_PROGRAMS)
def test_ast_text_round_trips(source):
    ast = parse(source)
    text = repr(ast)

    rebuilt = basic.AstTextReader.get_tree_from_string(text)

    assert repr(rebuilt) == text
    # A lone statement renders as itself, so it comes back without its statements wrapper.
    expected = ast.statement_nodes[0] if len(ast.statement_nodes) == 1 else ast
    assert ast_shape(rebuilt) == ast_shape(expected)


@pytest.mark.parametrize('source', ROUND_TRIP_PROGRAMS)
def test_ast_text_round_trip_computes_the_same(source):
    ast = parse(source)
    rebuilt = basic.AstTextReader.get_tree_from_string(repr(ast))

    error, value, symbols = execute(ast)
    assert error is None, error.as_string()
    assert execute(rebuilt) == (None, value, symbols)


def test_ast_text_of_generator_shapes_is_unchanged():
    """The trained checkpoint emits this text, so these shapes must keep their spelling."""
    assert repr(parse("var a = 3 + -2 * (4 plus 5)")) == \
        "IDENTIFIER:a:(int:3 PLUS ((MINUS int:2) MUL (int:4 IDENTIFIER:plus int:5)))"
    assert repr(parse("not True == None and 1.5 < 2; x")) == \
        "(((KEYWORD:not (IDENTIFIER:True EE IDENTIFIER:None)) KEYWORD:and (float:1.5 LT int:2)) NEWLINE IDENTIFIER:x)"


@pytest.mark.parametrize('source', ["[7]", "if 1 then\n7\nend"])
def test_list_literals_and_blocks_are_different_nodes(source):
    """`[7]` is a ListNode, while a program and a block body are BlockNodes, also when rebuilt from AST text."""
    for node in (parse(source), basic.AstTextReader.get_tree_from_string(repr(parse(source)))):
        program = as_program(node)
        assert isinstance(program, basic.BlockNode)
        statement = program.statement_nodes[0]
        if source == "[7]":
            assert isinstance(statement, basic.ListNode)
        else:
            assert isinstance(statement.cases[0][1], basic.BlockNode)


def test_ast_text_call_is_not_a_unary_op():
    """Args glued to the callee are a call; `(MINUS int:1)` with a space is a unary op."""
    assert isinstance(basic.AstTextReader.get_tree_from_string("(IDENTIFIER:f(int:1))"), basic.CallNode)
    assert isinstance(basic.AstTextReader.get_tree_from_string("(MINUS (int:1))"), basic.UnaryOpNode)


def test_ast_text_identifier_named_like_a_float_stays_a_name():
    node = basic.AstTextReader.get_tree_from_string("IDENTIFIER:inf:int:1")
    assert node.var_name_tok.value == "inf"


@pytest.mark.parametrize('text', [
    "(int:1 PLUS int:2",
    "(int:1 PLUS int:2))",
    "(int:1 int:2 int:3 int:4)",
    "()",
    "[int:1",
    'string:"unterminated',
    "(def int:1 -> int:2)",
])
def test_ast_text_malformed_raises(text):
    """`basic.inference` falls back to the classical parser when rebuilding raises."""
    with pytest.raises(ValueError):
        basic.AstTextReader.get_tree_from_string(text)


########################
# TEMPLATE INLINING
########################

@pytest.mark.parametrize('source, expected', [
    ("sum(9, 8)", "(int:9 PLUS int:8)"),
    ("9 sum 8", "(int:9 PLUS int:8)"),
    ("halve(sum(1, 2))", "((int:1 PLUS int:2) DIV int:2)"),
    # `sum` is a+b: an argument naming the later parameter `b` must not be substituted again.
    ("sum(b, 1)", "(IDENTIFIER:b PLUS int:1)"),
    ("b tilt 1", "(IDENTIFIER:b MINUS int:1)"),
    # Arguments containing spaces in their AST text.
    ('sum("a b", "c")', '(string:"a b" PLUS string:"c")'),
    ("sum([1, 2], 3)", "([int:1 int:2] PLUS int:3)"),
    # Inside control flow and function bodies.
    ("for i = 0 to 3 then double(i)", "(for IDENTIFIER:i int:0 int:3 (IDENTIFIER:i PLUS IDENTIFIER:i))"),
    ("def f(x) -> halve(x)", "(def IDENTIFIER:f(IDENTIFIER:x) -> (IDENTIFIER:x DIV int:2))"),
    # Left alone: wrong arity, and not a template.
    ("sum(1)", "(IDENTIFIER:sum(int:1))"),
    ("f(1, 2)", "(IDENTIFIER:f(int:1 int:2))"),
])
def test_inline_templates(source, expected):
    from . import data_generator as dg

    assert repr(dg.inline_templates(parse(source))) == expected


@pytest.mark.parametrize('source', [
    "var b = 5; sum(b, 1)",
    "var b = 2; b tilt 11",
    "7 + 8 times 8",
    "var x = 3; mid3(x, cube(x), x sum 1)",
])
def test_inlined_templates_compute_like_the_template_functions(source):
    from . import data_generator as dg

    def run_with_templates(node):
        context = basic.Context('<t>')
        context.symbol_table = basic.get_symbol_table()
        dg._load_template_functions_into_context(context)
        res = basic.Interpreter().visit(node, context)
        assert res.error is None
        return repr(res.value)

    ast = parse(source)
    assert run_with_templates(dg.inline_templates(ast)) == run_with_templates(ast)


########################
# STEP BUDGET
########################

def run_with_budget(source, max_steps):
    context = basic.Context('<program>')
    context.symbol_table = basic.get_symbol_table()
    return basic.Interpreter(max_steps=max_steps).visit(parse(source), context)


# Loops are bounded, so a budget that fails to apply makes the test fail instead of hang.
@pytest.mark.parametrize('source', [
    "var i = 0; while i < 1000 then var i = i + 1",
    "for i = 0 to 1000 then i",
    # Function bodies run on the caller's Interpreter, by call and by infix use.
    "def f() -> for i = 0 to 1000 then i; f()",
    "def g(a, b) -> for i = 0 to 1000 then i; 1 g 2",
    "def h()\nfor i = 0 to 1000 then i\nend; [h()]",
])
def test_step_budget_stops_long_programs(source):
    res = run_with_budget(source, max_steps=200)

    assert isinstance(res.error, basic.StepLimitError)
    assert run_with_budget(source, max_steps=None).error is None


def test_step_budget_lets_short_programs_finish():
    res = run_with_budget("for i = 0 to 10 then i * 2", max_steps=200)

    assert res.error is None
    assert repr(res.value) == "[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]]"


########################
# AST GENERATOR
########################

@pytest.fixture(scope='module')
def generated_programs():
    import random
    from collections import Counter
    from . import data_generator as dg

    state = random.getstate()
    random.seed(1234)
    stats = dg.STATS
    dg.STATS = Counter()
    try:
        texts = set()
        programs = [dg.generate_program(texts) for _ in range(400)]
        yield programs, dg.STATS
    finally:
        dg.STATS = stats
        random.setstate(state)


def test_generated_programs_run_and_fit_the_block(generated_programs):
    """Checked again from scratch, the way `data_generator.generate` builds samples."""
    from . import data_generator as dg

    programs, _ = generated_programs
    for program in programs:
        context = basic.Context('<program>')
        context.symbol_table = basic.get_symbol_table()
        dg._load_template_functions_into_context(context)
        for text in program:
            node = parse(text)
            lex_text = ' '.join(repr(t) for t in basic.Lexer('<stdin>', text).make_tokens()[0])
            ast_text = f'{tokens.SOF} {dg.inline_templates(node)} {tokens.EOF}'
            assert len(data.encode(lex_text, {})) <= basic.BLOCK_SIZE, text
            assert len(data.encode(ast_text, {})) <= basic.BLOCK_SIZE, text
            res = basic.Interpreter(max_steps=100_000).visit(node, context)
            assert res.error is None, f'{text}\n{res.error.as_string()}'


def test_generator_rejects_nothing_for_syntax_or_python_errors(generated_programs):
    """Typing is what rules these out; rejections are only for length, duplicates or values."""
    _, stats = generated_programs
    rejected = {key for key in stats if key.startswith('rejected')}

    assert rejected <= {'rejected: too long', 'rejected: duplicate', 'rejected: runtime error', 'rejected: step limit',
                        'rejected: inlining changes the result'}
    assert stats['rejected: runtime error'] < 0.01 * sum(n for key, n in stats.items() if key.startswith('drawn'))


def test_generated_programs_cover_the_language(generated_programs):
    from . import data_generator as dg

    programs, _ = generated_programs
    features = set().union(*(dg.ast_features(parse(text)) for program in programs for text in program))

    missing = dg.LANGUAGE_FEATURES - features
    assert not missing, f'never generated: {sorted(missing)}'


def test_build_samples_keeps_every_generated_program():
    """The generator already checks each statement, so building samples never fails."""
    import random
    from . import data_generator as dg

    state = random.getstate()
    random.seed(7)
    try:
        rows, failed = dg.build_samples(30)
    finally:
        random.setstate(state)

    assert (len(rows), failed) == (30, 0)
    assert all(len(row['x_in']) == len(row['x_out']) for row in rows)  # One input/target pair per statement.


def generator_after(*statements):
    """A ProgramGenerator that has already kept `statements`."""
    from . import data_generator as dg

    generator = dg.ProgramGenerator(set(statements))
    generator.accepted = [(text, parse(text)) for text in statements]
    generator.symbols, _ = generator.run([node for _, node in generator.accepted])
    generator.pending = {}
    return generator


def test_generator_rejects_a_statement_whose_inlined_target_computes_differently():
    """square(x) inlines to x*x, so `pop` would run twice in the target."""
    generator = generator_after("var l = [1, 2, 3]")

    assert generator.try_accept("var x = square(pop(l, 0))") == 'inlining changes the result'
    assert generator.try_accept("var x = square(l / 0)") is None


def test_generator_retries_the_kind_a_rejected_setup_was_for(monkeypatch):
    """A while loop with no counter around gets a setup first; when that setup is rejected, the
    retry is still for the while loop, not some other statement."""
    from . import data_generator as dg

    generator = generator_after("var l = [1]")  # A list (so a built-in would fit), no number variable.
    generator.texts.add("var a = 1")
    setups = iter([["var a = 1"], ["var b = 1"]])  # The first is a duplicate.
    monkeypatch.setattr(generator, 'setup_statement', lambda kind: next(setups))
    monkeypatch.setattr(dg.ProgramGenerator, 'KINDS', {'while': 1})

    assert generator.add_statement()
    assert [text for text, _ in generator.accepted] == ["var l = [1]", "var b = 1"]
    assert generator.next_kind == 'while'


def test_lexer_stream_quotes_strings():
    """The model reads this stream: a space or bracket inside a string must not look like a token boundary."""
    token_list, error = basic.Lexer('<stdin>', 'var p = "by ]"').make_tokens()
    assert error is None

    assert ' '.join(repr(t) for t in token_list) == 'SOF KEYWORD:var IDENTIFIER:p EQ string:"by ]" EOF'


@pytest.mark.parametrize('text, fixed', [
    ('(IDENTIFIER:e:(string:")" MUL int:0))', '(IDENTIFIER:e:(string:")" MUL int:0))'),  # Balanced: untouched.
    ('(string:"z(y" PLUS string:"\\"(")', '(string:"z(y" PLUS string:"\\"(")'),  # Escaped quote inside.
    ('(int:1 PLUS (int:2 MUL int:3)', '(int:1 PLUS (int:2 MUL int:3))'),
    ('int:1 PLUS int:2)', '(int:1 PLUS int:2)'),
])
def test_fix_unmatched_parenthesis_ignores_parens_in_strings(text, fixed):
    from .train_module import CrossAttentionTransformer

    assert CrossAttentionTransformer.fix_unmatched_parenthesis(text) == fixed


def test_symbol_tables_compare_lists_strings_and_functions_by_content():
    from .samples import symbol_tables_equal

    def symbols(source):
        context = basic.Context('<program>')
        context.symbol_table = basic.SymbolTable()
        basic.Interpreter().visit(parse(source), context)
        return context.symbol_table.symbols

    program = 'var l = [1, "a"]; var s = "x"; def f(a) -> a'
    assert symbols(program) is not symbols(program)
    assert symbol_tables_equal(symbols(program), symbols(program))
    assert not symbol_tables_equal(symbols(program), symbols('var l = [1, "b"]; var s = "x"; def f(a) -> a'))


@pytest.mark.parametrize('target, predicted', [
    ('def f(a) -> a + 1', 'def f(a) -> a * 2'),  # Same name, different body.
    ('def f(a) -> a', 'def f(b) -> b'),  # Different parameter.
    ('def f(a) -> a', 'def f(a)\nreturn a\nend'),  # Arrow vs block body.
    ('var l = [def (a) -> a + 1]', 'var l = [def (a) -> a * 2]'),  # A function inside a list.
])
def test_symbol_tables_compare_functions_by_body_not_name(target, predicted):
    from .samples import symbol_tables_equal

    def symbols(source):
        context = basic.Context('<program>')
        context.symbol_table = basic.SymbolTable()
        basic.Interpreter().visit(parse(source), context)
        return context.symbol_table.symbols

    assert symbol_tables_equal(symbols(target), symbols(target))
    assert not symbol_tables_equal(symbols(target), symbols(predicted))


def test_generator_report_measures_the_generator(monkeypatch):
    pytest.importorskip('coverage')
    import random
    from . import generator_report as report

    state = random.getstate()
    monkeypatch.setattr(report, 'REPORT_PROGRAMS', 50)
    try:
        result = report.measure()
    finally:
        random.setstate(state)

    interpreter = report.language_functions(result['functions'], 'Interpreter').values()
    lines_run = sum(s['covered_lines'] for s in interpreter) / sum(s['num_statements'] for s in interpreter)

    assert len(result['rows']) == 50
    assert lines_run > 0.7  # 2,000 programs run ~95%; even 50 run most of the interpreter.
