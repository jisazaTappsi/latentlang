INT = 'INT'
FLOAT = 'FLOAT'
IDENTIFIER = 'IDENTIFIER'
KEYWORD = 'KEYWORD'
PLUS = 'PLUS'
MINUS = 'MINUS'
MUL = 'MUL'
DIV = 'DIV'
POW = 'POW'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
EQ = 'EQ'
EE = 'EE'
NE = 'NE'
GT = 'GT'
LT = 'LT'
GTE = 'GTE'
LTE = 'LTE'
AND = 'and'
OR = 'or'
NOT = 'not'
VAR = 'var'
IF = 'if'
THEN = 'then'
ELIF = 'elif'
ELSE = 'else'
FOR = 'for'
TO = 'to'
STEP = 'step'
WHILE = 'while'
COMMA = 'comma'
ARROW = 'arrow'
FUN = 'def'
# Add new tokens here


NULL = 'None'
TRUE = 'True'
FALSE = 'False'
EOF = 'EOF'  # End of file
SOF = 'SOF'  # Start of file
PAD = 'PAD'  # After EOF

TOKENS = [
    INT,
    FLOAT,
    IDENTIFIER,
    KEYWORD,
    PLUS,
    MINUS,
    MUL,
    DIV,
    POW,
    LPAREN,
    RPAREN,
    EQ,
    EE,
    NE,
    GT,
    LT,
    GTE,
    LTE,
    AND,
    OR,
    NOT,
    VAR,
    IF,
    THEN,
    ELIF,
    ELSE,
    FOR,
    TO,
    STEP,
    WHILE,
    COMMA,
    ARROW,
    FUN,
    # Add new tokens here

    EOF,
    SOF,
    PAD,
]

KEYWORDS = [
    VAR,
    AND,
    OR,
    NOT,
    IF,
    THEN,
    ELIF,
    ELSE,
    FOR,
    TO,
    STEP,
    WHILE,
    FUN,
]

TOKEN_BASIC_IDS = {
    256 + idx: ','.join(str(int(i)) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
TOKEN_IDS = {name: 256 + idx for idx, name in enumerate(TOKENS)}
TOKEN_BYTES = {
    256 + idx: bytes(int(i) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
