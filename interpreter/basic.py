########################
# IMPORTS
########################
import re
import math
import torch
import string

from . import data
from .util import device, BLOCK_SIZE, MODEL_NAME
from .train_module import CrossAttentionTransformer

########################
# CONSTANTS
########################

DIGITS = '0123456789'
LETTERS = string.ascii_letters + '_'
LETTERS_DIGITS = LETTERS + DIGITS
QUOTES = '"\''


def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0:
        idx_end = len(text)

    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0:
            idx_end = len(text)

    return result.replace('\t', '')


########################
# ERRORS
########################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.error_name = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        result += f'\nFile: {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

class IllegalError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Illegal Syntax', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime error', details)
        self.context = context

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result += f'    File {pos.fn} line {str(pos.ln+1)}, in {ctx.display_name}\n'
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        return f'Traceback (most recent call last):\n{result}'

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

########################
# POSITION
########################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

########################
# TOKENS
########################
from .tokens import *


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value is not None: return f'{self.type}:{self.value}'
        return f'{self.type}'


########################
# LEXER
########################


class Lexer:
    def __init__(self,fn, text):
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        token_list = [Token(SOF), ]
        while self.current_char is not None:
            if self.current_char in [' ', '\t']:
                self.advance()
            elif self.current_char == '#':
                self.skip_comment()
            elif self.current_char in ';\n':
                token_list.append(Token(NEWLINE, pos_start=self.pos))
                self.advance()
            elif self.current_char in DIGITS + '.':
                token_list.append(self.make_number())
            elif self.current_char in LETTERS:
                token_list.append(self.make_identifier())
            elif self.current_char in QUOTES:
                token_list.append(self.make_string())
            elif self.current_char == '+':
                token_list.append(Token(PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                token_list.append(self.make_minus_or_arrow())
            elif self.current_char == '*':
                token_list.append(Token(MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                token_list.append(Token(DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                token_list.append(Token(POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                token_list.append(Token(LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                token_list.append(Token(RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '[':
                token_list.append(Token(LSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == ']':
                token_list.append(Token(RSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                token_list.append(tok)
            elif self.current_char == '=':
                token_list.append(self.make_equals())
            elif self.current_char == '<':
                token_list.append(self.make_less_than())
            elif self.current_char == '>':
                token_list.append(self.make_greater_than())
            elif self.current_char == ',':
                token_list.append(Token(COMMA, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalError(pos_start, self.pos, f'"{char}"')

        token_list.append(Token(EOF, pos_start=self.pos))
        return token_list, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(INT, int(num_str), pos_start, self.pos)
        else:
            return Token(FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        s = ''
        pos_start = self.pos.copy()
        is_escape_char = False
        quote_char = self.current_char
        self.advance()
        escape_chars = {
            'n': '\n',
            't': '\t',
        }

        while self.current_char is not None and (self.current_char != quote_char or is_escape_char):
            if is_escape_char:
                s += escape_chars.get(self.current_char, self.current_char)
                is_escape_char = False
            else:
                if self.current_char == '\\':
                    is_escape_char = True
                else:
                    s += self.current_char
            self.advance()

        self.advance()
        return Token(STRING, s, pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS:
            id_str += self.current_char
            self.advance()

        tok_type = KEYWORD if id_str in KEYWORDS else IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_minus_or_arrow(self):
        tok_type = MINUS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = ARROW

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(NE, pos_start=pos_start, pos_end=self.pos), None
        
        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = EQ
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = EE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = LT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = LTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = GT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = GTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def skip_comment(self):
        self.advance()  # advance past comment
        while self.current_char is not None and self.current_char not in '\n':
            self.advance()


########################
# NODES
########################


class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class StringNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        # A lone statement renders as itself, so single-statement AST text is unchanged
        # by `statements` becoming the parse entry point. Several are NEWLINE-separated,
        # which `get_tree_from_string` splits on before its arity dispatch.
        if len(self.element_nodes) == 1:
            return f'{self.element_nodes[0]}'
        separator = f' {NEWLINE} '
        return f'({separator.join(str(node) for node in self.element_nodes)})'


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

    def __repr__(self):
        return f'{self.var_name_tok}'


class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

    def __repr__(self):
        return f'{self.var_name_tok}:{self.value_node}'


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_tok} {self.node})'


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases)-1])[0].pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end


class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end

class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.should_auto_return = should_auto_return

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end

    def __repr__(self):
        args = ' '.join(str(t) for t in self.arg_name_toks)
        name = self.var_name_tok if self.var_name_tok else ''
        return f'({FUN} {name}({args}) -> {self.body_node})'


class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes)-1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end

    def __repr__(self):
        args = ' '.join(str(a) for a in self.arg_nodes)
        return f'({self.node_to_call}({args}))'


class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return
        self.pos_start = pos_start
        self.pos_end = pos_end


class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


########################
# PARSER RESULT
########################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advance(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:  # haven't advanced since
            self.error = error
        return self

########################
# PARSER
########################

class Parser:
    def __init__(self, token_list):
        self.token_list = token_list
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        self.update_current_tok()
        return self.current_tok

    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.current_tok

    def update_current_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.token_list):
            self.current_tok = self.token_list[self.tok_idx]

    @staticmethod
    def get_tree_from_string(text):
        """
        Parse a tree string representation and rebuild AST nodes recursively.
        Format examples:
        - (INT:2 MUL INT:2) -> BinOpNode
        - (MINUS INT:5) -> UnaryOpNode
        - INT:3 -> NumberNode
        - IDENTIFIER:x -> VarAccessNode (variable retrieval)
        - IDENTIFIER:x:INT:5 -> VarAssignNode (assignment; value can be any expr string)
        - IDENTIFIER:x:(INT:2 ADD INT:3) -> VarAssignNode with expr value
        """

        text = text.strip()
        
        # Helper function to parse a token string (e.g., "INT:2" or "MUL")
        def parse_token(token_str):
            token_str = token_str.strip()
            # Create a dummy position for token_list
            dummy_pos = Position(0, 0, 0, '<string>', '')
            if ':' in token_str:
                parts = token_str.split(':', 1)
                token_type = parts[0]
                # Try to parse value as int, float, or keep as string
                value_str = parts[1]
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str
                # Create a token with dummy position
                return Token(token_type, value, pos_start=dummy_pos)
            else:
                return Token(token_str, pos_start=dummy_pos)
        
        # Helper function to find matching closing parenthesis
        def find_matching_paren(text, start_idx):
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return i
            return -1
        
        # Try to parse as a simple token or identifier (no parentheses)
        if not text.startswith('('):
            # Variable: IDENTIFIER:name (access) or IDENTIFIER:name:expr (assignment)
            if text.startswith('IDENTIFIER:'):
                parts = text.split(':', 2)  # at most 3 parts so value can contain colons
                if len(parts) == 2:
                    # VarAccessNode: IDENTIFIER:name_var
                    tok = parse_token(f"IDENTIFIER:{parts[1]}")
                    return VarAccessNode(tok)
                elif len(parts) == 3:
                    # VarAssignNode: IDENTIFIER:name_var:value (value is expr string)
                    tok = parse_token(f"IDENTIFIER:{parts[1]}")
                    value_node = Parser.get_tree_from_string(parts[2].strip())
                    return VarAssignNode(tok, value_node)
                else:
                    raise ValueError(f"Invalid IDENTIFIER format: {text}")
            # Number or other token (e.g. "INT:2", "FLOAT:3.14", "MUL")
            if re.match(r'^[A-Z_]+(:.+)?$', text):
                tok = parse_token(text)
                return NumberNode(tok)
            else:
                tok = parse_token(text)
                return NumberNode(tok)
        
        # Parse as BinOpNode or UnaryOpNode (both start with '(')
        # Extract content inside the outermost parentheses
        end_idx = find_matching_paren(text, 0)
        if end_idx == -1:
            raise ValueError(f"Unmatched parenthesis in: {text}")
        
        content = text[1:end_idx].strip()
        
        # Split content by spaces, but preserve parentheses groups
        token_list = []
        current_token = ""
        paren_depth = 0
        
        i = 0
        while i < len(content):
            char = content[i]
            if char == '(':
                if paren_depth == 0 and current_token.strip():
                    token_list.append(current_token.strip())
                    current_token = ""
                current_token += char
                paren_depth += 1
            elif char == ')':
                current_token += char
                paren_depth -= 1
                if paren_depth == 0:
                    token_list.append(current_token.strip())
                    current_token = ""
            elif char == ' ' and paren_depth == 0:
                if current_token.strip():
                    token_list.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
            i += 1
        
        if current_token.strip():
            token_list.append(current_token.strip())
        
        token_list = [t for t in token_list if t]  # Remove empty token_list

        # Statements: (stmt NEWLINE stmt ...). Must come before the arity checks below,
        # or a two-statement list is indistinguishable from a BinOpNode.
        if NEWLINE in token_list:
            dummy_pos = Position(0, 0, 0, '<string>', '')
            element_nodes = [Parser.get_tree_from_string(t) for t in token_list if t != NEWLINE]
            return ListNode(element_nodes, dummy_pos, dummy_pos)

        if len(token_list) == 1:
            return Parser.get_tree_from_string(token_list[0])
        elif len(token_list) == 2:
            # UnaryOpNode: (op node)
            op_tok = parse_token(token_list[0])
            node = Parser.get_tree_from_string(token_list[1])
            return UnaryOpNode(op_tok, node)
        elif len(token_list) == 3:
            # BinOpNode: (left op right)
            left = Parser.get_tree_from_string(token_list[0])
            op_tok = parse_token(token_list[1])
            right = Parser.get_tree_from_string(token_list[2])
            return BinOpNode(left, op_tok, right)
        else:
            raise ValueError(f"Unexpected number of tokens: {len(token_list)} in: {content}")

    def parse(self):
        if self.current_tok.type == SOF:
            self.advance()

        res = self.statements()
        if not res.error and self.current_tok.type != EOF:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '+', '-', '*' or '/' but got {self.current_tok.type}"
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (INT, FLOAT):
            res.register_advance()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == IDENTIFIER:
            res.register_advance()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == STRING:
            res.register_advance()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == LPAREN:
            res.register_advance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == RPAREN:
                res.register_advance()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ')' but got {self.current_tok.type}"
                ))

        elif tok.type == LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)

        elif tok.matches(KEYWORD, IF):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(KEYWORD, FOR):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(KEYWORD, WHILE):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif tok.matches(KEYWORD, FUN):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        # Includes errors from power and factor too, because it's only called from there.
        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            f"Expected '{IF}', '{FOR}', '{WHILE}', '{FUN}', {VAR}  int, float, '+', '-', '(', '[' or identifier but got '{tok.type}'"
        ))

    def list_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != LSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f'Expected "["'
            ))

        res.register_advance()
        self.advance()
        if self.current_tok.type == RSQUARE:
            res.register_advance()
            self.advance()
            return res.success(
                ListNode([], pos_start, self.current_tok.pos_end.copy())
            )

        element_nodes = [res.register(self.expr())]
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ']', '{VAR}', '{IF}', '{FOR}', '{WHILE}', '{FUN}', 'int', 'float', identifier, '+', '-', '(', '[' or {NOT} but got {self.current_tok.type}"
            ))

        while self.current_tok.type == COMMA:
            res.register_advance()
            self.advance()

            element_nodes.append(res.register(self.expr()))
            if res.error: return res

        if self.current_tok.type != RSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ',', ']' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        return res.success(
            ListNode(element_nodes,
                     pos_start,
                     self.current_tok.pos_end.copy()))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases(IF))
        if res.error: return res
        cases, else_case = all_cases  # Needs to unpack after checking for errors, as res.node = None when res.error.
        return res.success(IfNode(cases, else_case))

    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(KEYWORD, case_keyword):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{case_keyword}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        if self.current_tok.type == NEWLINE:
            res.register_advance()
            self.advance()

            statements = res.register(self.statements())
            if res.error: return res
            cases.append((condition, statements, True))  # Non-assignable to variable

            if self.current_tok.matches(KEYWORD, END):
                res.register_advance()
                self.advance()
            elif self.current_tok.matches(KEYWORD, ELIF) or self.current_tok.matches(KEYWORD, ELSE):
                all_cases = res.register(self.if_expr_b_or_c())
                if res.error: return res
                new_cases, else_case = all_cases  # Needs to unpack after checking for errors, as res.node = None when res.error.
                cases.extend(new_cases)
            else:
                # Grammar requires one of (END|if-expr-b|if-expr-c) to close a block body.
                # if_expr_c matches empty for the inline branch, so it cannot enforce this.
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '{END}', '{ELIF}' or '{ELSE}' but got {self.current_tok.type}"
                ))
        else:
            statement = res.register(self.statement())
            if res.error: return res
            cases.append((condition, statement, False))  # Assignable to variable

            all_cases = res.register(self.if_expr_b_or_c())
            if res.error: return res
            new_cases, else_case = all_cases  # Needs to unpack after checking for errors, as res.node = None when res.error.
            cases.extend(new_cases)

        return res.success((cases, else_case))

    def if_expr_b_or_c(self):
        res = ParseResult()
        cases, else_case = [], None
        
        if self.current_tok.matches(KEYWORD, ELIF):
            all_cases = res.register(self.if_expr_b())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error: return res

        return res.success((cases, else_case))

    def if_expr_b(self):
        return self.if_expr_cases(ELIF)

    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(KEYWORD, ELSE):
            res.register_advance()
            self.advance()

            if self.current_tok.type == NEWLINE:
                res.register_advance()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)  # Non-assignable to variable

                if self.current_tok.matches(KEYWORD, END):
                    res.register_advance()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected '{END}' but got {self.current_tok.type}"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error: return res
                else_case = (expr, False)  # Assignable to variable

        return res.success(else_case)

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, FOR):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{FOR}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        if self.current_tok.type != IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier but got {self.current_tok.type}"
            ))

        var_name = self.current_tok
        res.register_advance()
        self.advance()

        if self.current_tok.type != EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '=' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, TO):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{TO}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(KEYWORD, STEP):
            res.register_advance()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))
        res.register_advance()
        self.advance()

        if self.current_tok.type == NEWLINE:
            res.register_advance()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(KEYWORD, END):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '{END}' but got {self.current_tok.type}"
                ))
            res.register_advance()
            self.advance()

            return res.success(ForNode(var_name_tok=var_name,
                                       start_value_node=start_value,
                                       end_value_node=end_value,
                                       step_value_node=step_value,
                                       body_node=body,
                                       should_return_null=True))

        body = res.register(self.statement())
        if res.error: return res
        return res.success(ForNode(var_name_tok=var_name,
                start_value_node=start_value,
                end_value_node=end_value,
                step_value_node=step_value,
                body_node=body,
                should_return_null=False))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, WHILE):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{WHILE}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))
        res.register_advance()
        self.advance()

        if self.current_tok.type == NEWLINE:
            res.register_advance()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(KEYWORD, END):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '{END}' but got {self.current_tok.type}"
                ))
            res.register_advance()
            self.advance()
            return res.success(WhileNode(condition, body, True))

        body = res.register(self.statement())
        if res.error: return res
        return res.success(WhileNode(condition, body, False))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, FUN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{FUN}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        if self.current_tok.type == IDENTIFIER:
            var_name_tok = self.current_tok
            res.register_advance()
            self.advance()
            if self.current_tok.type != LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '(' but got {self.current_tok.type}"
                ))
        else:
            var_name_tok = None
            if self.current_tok.type != LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or '(' but got {self.current_tok.type}"
                ))

        res.register_advance()
        self.advance()
        arg_name_toks = []

        if self.current_tok.type == IDENTIFIER:
            arg_name_toks.append(self.current_tok)
            res.register_advance()
            self.advance()

            while self.current_tok.type == COMMA:
                res.register_advance()
                self.advance()

                if self.current_tok.type != IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected an identifier but got {self.current_tok.type}"
                    ))

                arg_name_toks.append(self.current_tok)
                res.register_advance()
                self.advance()

            if self.current_tok.type != RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected an ',' or '(' but got {self.current_tok.type}"
                ))
        else:
            if self.current_tok.type != RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier  or ')' but got {self.current_tok.type}"
                ))
        res.register_advance()
        self.advance()

        if self.current_tok.type == NEWLINE:
            res.register_advance()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(KEYWORD, END):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '{END}' but got {self.current_tok.type}"
                ))
            res.register_advance()
            self.advance()
            return res.success(FuncDefNode(var_name_tok, arg_name_toks, body, False))
        elif self.current_tok.type == ARROW:
            res.register_advance()
            self.advance()

            body = res.register(self.expr())
            if res.error: return res
            return res.success(FuncDefNode(var_name_tok, arg_name_toks, body, True))
        else:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '->', '\\n' or ';' but got {self.current_tok.type}"
            ))

    def term(self):
        return self.bin_op(self.factor, (MUL, DIV, IDENTIFIER))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (PLUS, MINUS):
            res.register_advance()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def power(self):
        return self.bin_op(self.call, (POW, ), self.factor)

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type != LPAREN:
            return res.success(atom)

        res.register_advance()
        self.advance()
        if self.current_tok.type == RPAREN:
            res.register_advance()
            self.advance()
            return res.success(CallNode(atom, []))

        arg_nodes = [res.register(self.expr())]
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ')', '{VAR}', '{IF}', '{FOR}', '{WHILE}', '{FUN}', 'int', 'float', identifier, '+', '-', '(', '[' or {NOT} but got {self.current_tok.type}"
            ))

        while self.current_tok.type == COMMA:
            res.register_advance()
            self.advance()

            arg_nodes.append(res.register(self.expr()))
            if res.error: return res

        if self.current_tok.type != RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ',', ')' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        return res.success(CallNode(atom, arg_nodes))

    def arith_expr(self):
        return self.bin_op(self.term, (PLUS, MINUS))

    def comp_expr(self):
        res = ParseResult()
        if self.current_tok.matches(KEYWORD, NOT):
            op_tok = self.current_tok
            res.register_advance()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (EE, NE, LT, GT, LTE, GTE)))
        if res.error:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f'Expected int, float, identifier "+", "-", "(", "[", or "{NOT}" but got "{self.current_tok.type}"'
            ))
        return res.success(node)

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.matches(KEYWORD, RETURN):
            res.register_advance()
            self.advance()

            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(
                expr, pos_start, self.current_tok.pos_end.copy(),
            ))

        elif self.current_tok.matches(KEYWORD, CONTINUE):
            res.register_advance()
            self.advance()
            return res.success(ContinueNode(
                pos_start,
                self.current_tok.pos_end.copy(),
            ))

        elif self.current_tok.matches(KEYWORD, BREAK):
            res.register_advance()
            self.advance()
            return res.success(BreakNode(
                pos_start,
                self.current_tok.pos_end.copy(),
            ))

        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '{RETURN}', '{CONTINUE}', '{BREAK}', '{IF}', '{FOR}', '{WHILE}', '{FUN}', {VAR}  int, float, '+', '-', '(', '[' or identifier but got '{self.current_tok.type}'"
            ))
        return res.success(expr)

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == NEWLINE:
            res.register_advance()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True
        while True:
            newline_count = 0
            while self.current_tok.type == NEWLINE:
                res.register_advance()
                self.advance()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements: break
            statement_res = self.statement()
            statement = res.try_register(statement_res)
            if not statement:
                if statement_res.advance_count > 0:
                    return res.failure(statement_res.error)
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(ListNode(
            statements,
            pos_start,
            self.current_tok.pos_end.copy(),
        ))

    def expr(self):
        res = ParseResult()
        if self.current_tok.matches(KEYWORD, VAR):
            res.register_advance()
            self.advance()
            if self.current_tok.type != IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                        pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                        details=f'In assignment was looking for IDENTIFIER but got {self.current_tok.type}'
                    )
                )
            var_name = self.current_tok

            res.register_advance()
            self.advance()
            if self.current_tok.type != EQ:
                return res.failure(InvalidSyntaxError(
                        pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                        details=f'In assignment was looking for EQ but got {self.current_tok.type}'
                    )
                )
            res.register_advance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((KEYWORD, AND), (KEYWORD, OR))))
        if res.error:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '{IF}', '{FOR}', '{WHILE}', '{FUN}', {VAR}  int, float, '+', '-', '(', '[' or identifier but got '{self.current_tok.type}'"
            ))
        return res.success(node)

    def bin_op(self, func_a, ops, func_b=None):
        if func_b is None: func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advance()
            self.advance()
            right = res.register(func_b())
            if res.error: return res

            left = BinOpNode(left, op_tok, right)
        return res.success(left)

########################
# RUNTIME RESULT
########################

class RTResult:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self

    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        return (
            self.error or
            self.func_return_value or
            self.loop_should_continue or
            self.loop_should_break
        )

########################
# VALUES
########################

class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def to_json(self):
        return {"type": type(self).__name__.lower(), "value": repr(self)}

    def add_to(self, other):
        return None, self.illegal_operation(other)

    def sub_by(self, other):
        return None, self.illegal_operation(other)

    def mul_by(self, other):
        return None, self.illegal_operation(other)

    def div_by(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self):
        return None, self.illegal_operation()

    def execute(self, args):
        return None, self.illegal_operation()

    def copy(self):
        raise Exception('No copy method defined')

    def illegal_operation(self, other=None):
        if not other: other = self  # Unary ops pass no operand, so the error spans this value alone.
        return RTError(
            self.pos_start, other.pos_end,
            'Illegal operation',
            self.context
        )

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def add_to(self, other):
        if isinstance(other, Number):
            return Number((self.value + other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def sub_by(self, other):
        if isinstance(other, Number):
            return Number((self.value - other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def mul_by(self, other):
        if isinstance(other, Number):
            return Number((self.value * other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(pos_start=self.pos_start,
                                     pos_end=self.pos_end,
                                     details='Division by zero :(',
                                     context=self.context)
            return Number((self.value / other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def pow_by(self, other):
        if isinstance(other, Number):
            return Number((self.value ** other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def notted(self):
        return Number(int(self.value == 0)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def to_json(self):
        return {"type": "number", "value": self.value}

    def __repr__(self):
        return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)


class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def add_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def mul_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or '<anonymous>'

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_same_len(self, arg_names, args):
        res = RTResult()

        if len(args) != len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"'{self.name}' function takes {len(arg_names)} args but {len(args)} were given",
                self.context
            ))

        return res.success(None)

    def populate_args(self, arg_names, args, exec_ctx):
        for arg_name, arg_value in zip(arg_names, args):
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)

    def check_and_populate_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_same_len(arg_names, args))
        if res.should_return(): return res
        self.populate_args(arg_names, args, exec_ctx)
        return res.success(None)


class Function(BaseFunction):
    """Callable value: used for both ``f(x,y)`` and infix ``x f y`` when ``f`` names this function."""

    def __init__(self, name, body_node, arg_names, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_auto_return = should_auto_return

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_new_context()
        res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
        if res.should_return(): return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return_value is None: return res

        ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def to_json(self):
        return {"type": "function", "name": self.name, "args": self.arg_names}

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res

        return res.success(return_value)


    def no_visit_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    def execute_print(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number.null)
    execute_print.arg_names = ['value']

    def execute_print_ret(self, exec_ctx):
        return RTResult().success(
            String(str(exec_ctx.symbol_table.get('value')))
        )
    execute_print_ret.arg_names = ['value']

    def execute_input(self, exec_ctx):
        text = input()
        return RTResult().success(String(text))
    execute_input.arg_names = []

    def execute_input_int(self, exec_ctx):
        while True:
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RTResult().success(Number(number))
    execute_input_int.arg_names = []

    def execute_clear(self, exec_ctx):
        print("\033[H\033[J", end="", flush=True)  # ANSI clear: no subprocess, no TERM needed
        return RTResult().success(Number.null)
    execute_clear.arg_names = []

    def execute_is_number(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get('value'), Number)
        return RTResult().success(Number.true if is_number else Number.false)
    execute_is_number.arg_names = ['value']

    def execute_is_string(self, exec_ctx):
        is_string = isinstance(exec_ctx.symbol_table.get('value'), String)
        return RTResult().success(Number.true if is_string else Number.false)
    execute_is_string.arg_names = ['value']

    def execute_is_list(self, exec_ctx):
        is_list = isinstance(exec_ctx.symbol_table.get('value'), List)
        return RTResult().success(Number.true if is_list else Number.false)
    execute_is_list.arg_names = ['value']

    def execute_is_function(self, exec_ctx):
        is_function = isinstance(exec_ctx.symbol_table.get('value'), BaseFunction)
        return RTResult().success(Number.true if is_function else Number.false)
    execute_is_function.arg_names = ['value']

    def execute_append(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get('list')
        value = exec_ctx.symbol_table.get('value')

        if not isinstance(list_, List):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end,
                        'First argument must be a list',
                        exec_ctx))

        list_.elements.append(value)
        return RTResult().success(Number.null)
    execute_append.arg_names = ['list', 'value']

    def execute_pop(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get('list')
        index = exec_ctx.symbol_table.get('index')

        if not isinstance(list_, List):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end,
                        'First argument must be a list',
                        exec_ctx))

        if not isinstance(index, Number):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end,
                        'Second argument must be a number',
                        exec_ctx))

        try:
            element = list_.elements.pop(index.value)
        except IndexError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                'Index out of range', exec_ctx))

        return RTResult().success(element)
    execute_pop.arg_names = ['list', 'index']

    def execute_extend(self, exec_ctx):
        list_a = exec_ctx.symbol_table.get('list_a')
        list_b = exec_ctx.symbol_table.get('list_b')

        if not isinstance(list_a, List):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end,
                        'First argument must be a list',
                        exec_ctx))

        if not isinstance(list_b, List):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end,
                        'Second argument must be a list',
                        exec_ctx))

        list_a.elements.extend(list_b.elements)
        return RTResult().success(Number.null)
    execute_extend.arg_names = ['list_a', 'list_b']

    def execute_len(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get('list')
        if not isinstance(list_, List):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end, "Argument must be a list", exec_ctx))

        return RTResult().success(Number(len(list_.elements)))
    execute_len.arg_names = ['list']

    def execute_run(self, exec_ctx):
        fn = exec_ctx.symbol_table.get('fn')
        if not isinstance(fn, String):
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end, "Argument must be a string", exec_ctx))

        fn = fn.value  # Python string

        try:
            with open(fn, 'r') as f:
                script = f.read()
        except Exception as e:
            return RTResult().failure(
                RTError(self.pos_start, self.pos_end, f'Failed to load script "{fn}"\n{str(e)}', exec_ctx,))

        _, error = run(fn, script)
        if error:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end, f'Failed to finish executing script "{fn}"\n{error.as_string()}',
                exec_ctx,))

        return RTResult().success(Number.null)
    execute_run.arg_names = ['fn']


BuiltInFunction.print        = BuiltInFunction("print")
BuiltInFunction.print_ret    = BuiltInFunction("print_ret")
BuiltInFunction.input        = BuiltInFunction("input")
BuiltInFunction.input_int    = BuiltInFunction("input_int")
BuiltInFunction.clear        = BuiltInFunction("clear")
BuiltInFunction.is_number    = BuiltInFunction("is_number")
BuiltInFunction.is_string    = BuiltInFunction("is_string")
BuiltInFunction.is_list      = BuiltInFunction("is_list")
BuiltInFunction.is_function  = BuiltInFunction("is_function")
BuiltInFunction.append       = BuiltInFunction("append")
BuiltInFunction.pop          = BuiltInFunction("pop")
BuiltInFunction.extend       = BuiltInFunction("extend")
BuiltInFunction.len          = BuiltInFunction("len")
BuiltInFunction.run          = BuiltInFunction("run")


class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def __str__(self):
        return ', '.join([str(e) for e in self.elements])

    def __repr__(self):
        return f'[{', '.join([repr(e) for e in self.elements])}]'

    def add_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def mul_by(self, other):
        if not isinstance(other, List):
            return None, Value.illegal_operation(self, other)

        new_list = self.copy()
        new_list.elements.extend(other.elements)
        return new_list, None

    def sub_by(self, other):
        if not isinstance(other, Number):
            return None, Value.illegal_operation(self, other)

        new_list = self.copy()
        try:
            new_list.elements.pop(other.value)
            return new_list, None
        except IndexError:
            return None, RTError(
                other.pos_start, other.pos_end,
                'Element at this index could not be removed from list because index is out or range',
                self.context
            )

    def div_by(self, other):
        if not isinstance(other, Number):
            return None, Value.illegal_operation(self, other)

        try:
            return self.elements[other.value], None
        except IndexError:
            return None, RTError(
                other.pos_start, other.pos_end,
                'Element at this index could not be retrieved from list because index is out or range',
                self.context
            )

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

########################
# CONTEXT
########################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


########################
# SYMBOL TABLE
########################

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name)
        if value is None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]

    def to_json(self):
        if self.symbols:
            return {name: val.to_json() for name, val in self.symbols.items()}
        else:
            return None

    @staticmethod
    def from_json(symbols):
        if symbols:
            table = SymbolTable()
            table.symbols = {name: val.from_json() for name, val in symbols.items()}
            return table
        else:
            return None


########################
# INTERPRETER
########################

class Interpreter:
    """Expression execution.
    User-defined *operators* and *functions* are the same abstraction: a name bound to a
    ``Function`` value. Infix ``a op b`` is executed by the same mechanism as call syntax
    ``op(a, b)`` — both resolve the name and apply :meth:`Function.execute` to the operands.
    """

    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value is None:
            return res.failure(RTError(
                pos_start=node.pos_start,
                pos_end=node.pos_end,
                details=f'Variable "{var_name}" is not defined',
                context=context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res
        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res

        if node.op_tok.type == IDENTIFIER:
            # Same as call syntax op(left, right): one Function, two args.
            name = node.op_tok.value
            fn = context.symbol_table.get(name)
            if isinstance(fn, Function):
                return fn.execute([left, right])
            return res.failure(RTError(
                node.op_tok.pos_start, node.op_tok.pos_end,
                f'"{name}" is not defined',
                context,
            ))

        if node.op_tok.type == PLUS:
            result, error = left.add_to(right)
        elif node.op_tok.type == MINUS:
            result, error = left.sub_by(right)
        elif node.op_tok.type == MUL:
            result, error = left.mul_by(right)
        elif node.op_tok.type == DIV:
            result, error = left.div_by(right)
        elif node.op_tok.type == POW:
            result, error = left.pow_by(right)
        elif node.op_tok.type == EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(KEYWORD, AND):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(KEYWORD, OR):
            result, error = left.ored_by(right)
        else:
            return res.failure(RTError(
                node.op_tok.pos_start, node.op_tok.pos_end,
                f'Unsupported binary op {node.op_tok.type}',
                context,
            ))

        if error:
            return res.failure(error)
        return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return(): return res

        error = None
        if node.op_tok.type == MINUS:
            number, error = number.mul_by(Number(-1))
        elif node.op_tok.matches(KEYWORD, NOT):
            number, error = number.notted()

        if error:
            return res.failure(error)

        return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_ListNode(self, node, context):
        res = RTResult()
        element_values = []

        for element_node in node.element_nodes:
            element_values.append(res.register(self.visit(element_node, context)))
            if res.should_return(): return res

        return res.success(
            List(element_values).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr, should_return_null in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return(): return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Number.null if should_return_null else expr_value)

        if node.else_case:
            expr, should_return_null = node.else_case
            else_value = res.register(self.visit(expr, context))
            if res.should_return(): return res
            return res.success(Number.null if should_return_null else else_value)

        return res.success(Number.null)

    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.should_return(): return res
        else:
            step_value = Number(1)  # default step is 1

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() \
                and res.loop_should_continue is False \
                and res.loop_should_break is False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        if node.should_return_null:
            return res.success(Number.null)

        return res.success(
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res

            if not condition.is_true(): break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() \
                    and res.loop_should_continue is False \
                    and res.loop_should_break is False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        if node.should_return_null:
            return res.success(Number.null)

        return res.success(
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos()

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res


        return_value = res.register(value_to_call.execute(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Number.null

        return res.success_return(value)

    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()

    def visit_BreakNode(self, node, context):
        return RTResult().success_break()


########################
# RUN
########################

def get_symbol_table():
    table = SymbolTable()
    # Start from the same built-ins as the global symbol table (NULL/TRUE/FALSE).
    table.symbols.update(global_symbol_table.symbols)
    table.set(NULL, Number(0))
    table.set(TRUE, Number(1))
    table.set(FALSE, Number(0))
    return table


global_symbol_table = SymbolTable()
global_symbol_table.set(NULL, Number.null)
global_symbol_table.set(FALSE, Number.false)
global_symbol_table.set(TRUE, Number.true)
global_symbol_table.set("math_pi", Number.math_PI)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("input_int", BuiltInFunction.input_int)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("is_num", BuiltInFunction.is_number)
global_symbol_table.set("is_str", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_fun", BuiltInFunction.is_function)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("run", BuiltInFunction.run)


def run(fn, text):
    lexer = Lexer(fn, text)
    token_list, error = lexer.make_tokens()
    if error: return None, error

    # Generate AST
    parser = Parser(token_list)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    res = interpreter.visit(ast.node, context)

    return res.value, res.error


def inference(token_list):
    """
    Given lexer `token_list` (as produced by `Lexer.make_tokens`),
    use the trained code transformer to predict an AST and
    return the corresponding AST node.
    """
    lex_text = ' '.join(t.__repr__() for t in token_list)

    lex_merges, ast_merges = {}, {}  # data.get_merges()
    lex_encoded = data.encode(lex_text, lex_merges)
    lex_encoded = data.add_pad_tokens_and_trim(lex_encoded, BLOCK_SIZE)

    model = CrossAttentionTransformer().to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    model.eval()

    predicted_ast_text = model.inference(lex_encoded, ast_merges)

    try:
        ast_node = Parser.get_tree_from_string(predicted_ast_text)
    except Exception:
        # If we cannot rebuild the tree, fall back to the standard parser
        parser = Parser(token_list)
        ast = parser.parse()
        return ast.node

    return ListNode([ast_node], ast_node.pos_start, ast_node.pos_end)


def run_interpreter(symbol_table, ast_node):
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = symbol_table if symbol_table else global_symbol_table
    return interpreter.visit(ast_node, context), context


def run_ai(fn, text, symbol_table=None, force_ai=False, force_interpreter=False):
    lexer = Lexer(fn, text)
    token_list, error = lexer.make_tokens()
    if error:
        context = Context('<program>')
        context.symbol_table = symbol_table
        return RTResult().failure(error), context

    # Generate AST
    parser = Parser(token_list)
    ast = parser.parse()
    if (force_ai or ast.error) and not force_interpreter:  # Uses AI if the base interpreter fails.
        ast_node = inference(token_list)
        return run_interpreter(symbol_table, ast_node)
    elif ast.error:  # force_interpreter: report the parse error rather than visiting a None AST.
        context = Context('<program>')
        context.symbol_table = symbol_table
        return RTResult().failure(ast.error), context
    else:
        res, context = run_interpreter(symbol_table, ast.node)
        if res.error and not force_interpreter:   # Try again and use AI.
            ast_node = inference(token_list)
            return run_interpreter(symbol_table, ast_node)

        return res, context
