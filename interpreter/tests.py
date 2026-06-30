from . import basic
from . import tokens
from .basic import EOF


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
    value, error = basic.run('<stdin>', "10 + 2.5 * 3 - 4.2 / 2")
    
    assert error is None
    assert isinstance(value, basic.Number)


def test_parsing_unary_minus():
    """Test parsing unary minus operator: -5"""
    value, error = basic.run('<stdin>', "-5")
    
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == -5


def test_parsing_unary_plus():
    """Test parsing unary plus operator: +3.5"""
    value, error = basic.run('<stdin>', "+3.5")
    
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 3.5


def test_parsing_parentheses():
    """Test parsing parentheses for grouping: (1 + 2) * 3"""
    value, error = basic.run('<stdin>', "(1 + 2) * 3")
    
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 9


def test_parsing_unary_with_parentheses():
    """Test parsing unary operator with parentheses: -(1 + 2)"""
    value, error = basic.run('<stdin>', "-(1 + 2)")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == -3


def test_stupidly_simple_not():
    ast, error = basic.run('<stdin>', f"{tokens.NOT} {tokens.TRUE} == {tokens.NULL}")
    assert error is None


def test_function_def_and_calls():
    # def f(a,b) -> a+b
    value, error = basic.run('<stdin>', "def f(a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # f(8,9)
    value, error = basic.run('<stdin>', "f(8,9)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 17

    # no args, call
    value, error = basic.run('<stdin>', "f()")
    assert error is not None
    assert isinstance(error, basic.RTError)

    # call
    value, error = basic.run('<stdin>', "f(3,4,5)")
    assert error is not None
    assert isinstance(error, basic.RTError)

    # var func = f
    value, error = basic.run('<stdin>', "var func = f")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # func
    value, error = basic.run('<stdin>', "func")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "f"

    # func(2,3)
    value, error = basic.run('<stdin>', "func(2,3)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 5

    # def (a, b) -> a + b
    value, error = basic.run('<stdin>', "def (a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "<anonymous>"

    # var ano = def (a, b) -> a + b
    value, error = basic.run('<stdin>', "var ano = def (a, b) -> a + b")
    assert error is None
    assert isinstance(value, basic.Function)

    # ano(3,3)
    value, error = basic.run('<stdin>', "ano(3,3)")
    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 6

    # def zero(a) -> a/0
    value, error = basic.run('<stdin>', "def zero(a) -> a/0")
    assert error is None
    assert isinstance(value, basic.Function)
    assert value.name == "zero"

    # zero(9)
    value, error = basic.run('<stdin>', "zero(9)")
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
    value, error = basic.run('<stdin>', "2 ^ 3")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 8


def test_power_to_zero():
    """Test the power operator with exponent 0: 5 ^ 0"""
    value, error = basic.run('<stdin>', "5 ^ 0")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 1


def test_power_is_right_associative():
    """Test that power is right-associative: 2 ^ 3 ^ 2 == 2 ^ (3 ^ 2) == 512"""
    value, error = basic.run('<stdin>', "2 ^ 3 ^ 2")

    assert error is None
    assert isinstance(value, basic.Number)
    assert value.value == 512


def test_string_literal():
    """Test a simple string literal: "hello\""""
    value, error = basic.run('<stdin>', '"hello"')

    assert error is None
    assert isinstance(value, basic.String)
    assert value.value == "hello"


def test_string_concatenation():
    """Test concatenating two strings: "foo" + "bar\""""
    value, error = basic.run('<stdin>', '"foo" + "bar"')

    assert error is None
    assert isinstance(value, basic.String)
    assert value.value == "foobar"


def test_string_repetition():
    """Test repeating a string with a number: "ab" * 3"""
    value, error = basic.run('<stdin>', '"ab" * 3')

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
