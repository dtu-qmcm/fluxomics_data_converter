"""Minimal MATLAB reader for INCA model scripts.

INCA model files are MATLAB programs, not declarative documents: they build
objects through constructor calls and then mutate them through property
assignments.  Reading one therefore means replaying a small, well-defined
subset of MATLAB rather than pattern-matching lines.

This module provides that subset in two layers:

- :func:`read_statements` — strips comments, joins ``...`` continuations and
  splits the source into logical statements while tracking original line
  numbers for diagnostics.
- :func:`evaluate` — evaluates the literal value grammar (numbers, strings,
  vectors, cell arrays, ranges, arithmetic, member access) against an
  environment.  Anything outside that grammar — most importantly opaque
  function calls such as ``mod2stoich(m)'`` or ``normrnd(...)`` — evaluates
  to :data:`UNKNOWN` instead of raising, which is what lets the caller walk
  past statements it cannot reproduce.

Nothing here is INCA-specific; :mod:`inca_parser` supplies the semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple


class _Unknown:
    """Sentinel for a value this reader cannot reproduce.

    Produced by calls to functions outside the supported grammar (for
    example MATLAB built-ins or INCA solver helpers).  It is deliberately
    falsy so callers can test it with ``if value is UNKNOWN`` or a plain
    truth test.
    """

    _instance: "_Unknown | None" = None

    def __new__(cls) -> "_Unknown":
        """Return the single shared instance.

        Returns:
            The singleton ``_Unknown`` value.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        """Return ``False`` so unknown values are falsy.

        Returns:
            Always ``False``.
        """
        return False

    def __repr__(self) -> str:
        """Return the debug representation.

        Returns:
            The string ``"UNKNOWN"``.
        """
        return "UNKNOWN"


UNKNOWN = _Unknown()

# Characters that make a following quote a transpose operator rather than
# the start of a character array.
_TRANSPOSE_AFTER = "_)]}.'"


@dataclass(frozen=True)
class Statement:
    """One logical MATLAB statement.

    Attributes:
        text: Statement source with comments removed and continuations
            joined.
        line: 1-based line number in the original file where the statement
            starts, used for diagnostics.
    """

    text: str
    line: int


def _is_transpose(prev_significant: str) -> bool:
    """Report whether a quote at this position is the transpose operator.

    MATLAB overloads ``'`` as both the character-array delimiter and the
    transpose operator.  A quote is a transpose only when it directly
    follows a value, i.e. an identifier, a number, or a closing bracket.

    Args:
        prev_significant: Last non-whitespace character emitted before the
            quote, or ``""`` at the start of input.

    Returns:
        ``True`` if the quote should be read as transpose.
    """
    if not prev_significant:
        return False
    return prev_significant.isalnum() or prev_significant in _TRANSPOSE_AFTER


def clean_source(source: str) -> Tuple[str, List[int]]:
    """Strip comments and join line continuations.

    Comments run from an unquoted ``%`` to end of line.  A trailing ``...``
    joins the current line with the next one; INCA scripts also use ``...``
    alone as a spacer line and as ``... % comment``, both of which collapse
    to a single space.

    Args:
        source: Raw file contents.

    Returns:
        A tuple of the cleaned source and a parallel list giving, for each
        character of the cleaned source, its 1-based line number in the
        original file.
    """
    out: List[str] = []
    lines: List[int] = []
    line = 1
    index = 0
    length = len(source)
    in_string = False
    prev_significant = ""

    def emit(char: str) -> None:
        out.append(char)
        lines.append(line)

    while index < length:
        char = source[index]

        if in_string:
            if char == "'":
                # A doubled quote is an escaped quote, not a terminator.
                if index + 1 < length and source[index + 1] == "'":
                    emit("'")
                    emit("'")
                    index += 2
                    continue
                in_string = False
                prev_significant = char
            if char == "\n":
                line += 1
            emit(char)
            index += 1
            continue

        if char == "'":
            if not _is_transpose(prev_significant):
                in_string = True
            emit(char)
            prev_significant = char
            index += 1
            continue

        if char == "%":
            while index < length and source[index] != "\n":
                index += 1
            continue

        if source.startswith("...", index):
            while index < length and source[index] != "\n":
                index += 1
            if index < length:
                index += 1
                line += 1
            emit(" ")
            continue

        if char == "\n":
            emit(char)
            line += 1
            prev_significant = ""
            index += 1
            continue

        emit(char)
        if not char.isspace():
            prev_significant = char
        index += 1

    return "".join(out), lines


def split_statements(cleaned: str, line_map: Sequence[int]) -> List[Statement]:
    """Split cleaned source into logical statements.

    Statements are separated by ``;`` or a newline at bracket depth zero and
    outside character arrays.  Separators inside ``(``, ``[`` or ``{`` are
    part of the enclosed expression and are preserved.

    Args:
        cleaned: Source returned by :func:`clean_source`.
        line_map: Per-character line numbers returned by
            :func:`clean_source`.

    Returns:
        The non-empty statements in source order.
    """
    statements: List[Statement] = []
    buffer: List[str] = []
    start = 0
    depth = 0
    index = 0
    length = len(cleaned)
    in_string = False
    prev_significant = ""

    def flush(end: int) -> None:
        text = "".join(buffer).strip()
        if text:
            line = line_map[start] if start < len(line_map) else 0
            statements.append(Statement(text=text, line=line))
        buffer.clear()

    while index < length:
        char = cleaned[index]

        if not buffer:
            start = index

        if in_string:
            buffer.append(char)
            if char == "'":
                if index + 1 < length and cleaned[index + 1] == "'":
                    buffer.append("'")
                    index += 2
                    continue
                in_string = False
            index += 1
            continue

        if char == "'":
            if not _is_transpose(prev_significant):
                in_string = True
            buffer.append(char)
            prev_significant = char
            index += 1
            continue

        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and char in ";\n":
            flush(index)
            prev_significant = ""
            index += 1
            continue

        buffer.append(char)
        if not char.isspace():
            prev_significant = char
        index += 1

    flush(length)
    return statements


def read_statements(source: str) -> List[Statement]:
    """Read MATLAB source into logical statements.

    Args:
        source: Raw file contents.

    Returns:
        The statements in source order, comments removed and continuations
        joined.
    """
    cleaned, line_map = clean_source(source)
    return split_statements(cleaned, line_map)


# --------------------------------------------------------------------------
# Expression evaluation
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
    | (?P<name>[A-Za-z_]\w*)
    | (?P<op>\.\*|\./|\.\^|\.\'|[-+*/^:,;()\[\]{}.'])
    | (?P<space>\s+)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    """A single expression token.

    Attributes:
        kind: One of ``number``, ``name``, ``string``, ``op``.
        text: Literal token text; for strings, the unescaped contents.
        spaced: Whether whitespace immediately preceded this token, used to
            detect space-separated matrix elements such as ``[1 2 3]``.
    """

    kind: str
    text: str
    spaced: bool = False


def _tokenize(expr: str) -> List[_Token]:
    """Tokenize a MATLAB expression.

    Args:
        expr: Expression source.

    Returns:
        The token list.

    Raises:
        ValueError: If the expression contains an unterminated character
            array or a character that cannot be tokenized.
    """
    tokens: List[_Token] = []
    index = 0
    length = len(expr)
    spaced = False

    while index < length:
        char = expr[index]
        previous = (
            tokens[-1].text[-1]
            if tokens and tokens[-1].kind != "string"
            else ""
        )

        if char == "'" and not _is_transpose(previous):
            index += 1
            chars: List[str] = []
            while index < length:
                if expr[index] == "'":
                    if index + 1 < length and expr[index + 1] == "'":
                        chars.append("'")
                        index += 2
                        continue
                    break
                chars.append(expr[index])
                index += 1
            if index >= length:
                raise ValueError(f"Unterminated string in: {expr!r}")
            index += 1
            tokens.append(_Token("string", "".join(chars), spaced))
            spaced = False
            continue

        match = _TOKEN_RE.match(expr, index)
        if match is None:
            raise ValueError(
                f"Cannot tokenize {expr[index]!r} in expression: {expr!r}"
            )
        index = match.end()
        if match.lastgroup == "space":
            spaced = True
            continue
        tokens.append(_Token(match.lastgroup or "op", match.group(), spaced))
        spaced = False

    return tokens


def _as_list(value: Any) -> List[float]:
    """Coerce a scalar or list to a list.

    Args:
        value: Scalar number or list of numbers.

    Returns:
        A list of numbers.
    """
    return list(value) if isinstance(value, list) else [value]


def _binary(op: str, left: Any, right: Any) -> Any:
    """Apply a binary operator with MATLAB broadcasting.

    Scalar-scalar, vector-scalar and elementwise vector-vector combinations
    are supported; anything else yields :data:`UNKNOWN`.

    Args:
        op: Operator text (``+``, ``-``, ``*``, ``/``, ``^`` and their
            elementwise ``.`` forms).
        left: Left operand.
        right: Right operand.

    Returns:
        The result, or :data:`UNKNOWN` if the operands are not numeric.
    """
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    if isinstance(left, str) or isinstance(right, str):
        return UNKNOWN

    scalar_ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        ".*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b else UNKNOWN,
        "./": lambda a, b: a / b if b else UNKNOWN,
        "^": lambda a, b: a**b,
        ".^": lambda a, b: a**b,
    }
    func = scalar_ops.get(op)
    if func is None:
        return UNKNOWN

    left_is_vec = isinstance(left, list)
    right_is_vec = isinstance(right, list)

    if not left_is_vec and not right_is_vec:
        return func(left, right)
    if left_is_vec and not right_is_vec:
        return [func(item, right) for item in left]
    if right_is_vec and not left_is_vec:
        return [func(left, item) for item in right]
    if len(left) != len(right):
        return UNKNOWN
    return [func(a, b) for a, b in zip(left, right)]


class _Parser:
    """Recursive-descent parser for the supported expression grammar."""

    def __init__(self, tokens: List[_Token], env: Mapping[str, Any]):
        """Initialise the parser.

        Args:
            tokens: Token list from :func:`_tokenize`.
            env: Variable environment for identifier lookups.
        """
        self.tokens = tokens
        self.env = env
        self.pos = 0

    def peek(self) -> _Token | None:
        """Return the next token without consuming it.

        Returns:
            The next token, or ``None`` at end of input.
        """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def at_op(self, *ops: str) -> bool:
        """Report whether the next token is one of the given operators.

        Args:
            *ops: Operator texts to test against.

        Returns:
            ``True`` if the next token matches.
        """
        token = self.peek()
        return token is not None and token.kind == "op" and token.text in ops

    def take(self) -> _Token:
        """Consume and return the next token.

        Returns:
            The consumed token.

        Raises:
            ValueError: If input is exhausted.
        """
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token

    def parse(self) -> Any:
        """Parse a complete expression.

        Returns:
            The evaluated value, or :data:`UNKNOWN`.
        """
        value = self.parse_range()
        if self.peek() is not None:
            return UNKNOWN
        return value

    def parse_range(self) -> Any:
        """Parse a colon range or a plain additive expression.

        Returns:
            A list for ``a:b`` and ``a:s:b`` forms, otherwise the scalar or
            vector produced by the operand.
        """
        first = self.parse_add()
        if not self.at_op(":"):
            return first
        self.take()
        second = self.parse_add()
        step: Any = 1.0
        stop = second
        if self.at_op(":"):
            self.take()
            step = second
            stop = self.parse_add()
        if UNKNOWN in (first, step, stop):
            return UNKNOWN
        if any(isinstance(v, list) for v in (first, step, stop)):
            return UNKNOWN
        if step == 0:
            return UNKNOWN
        values: List[float] = []
        count = int((stop - first) / step + 1e-9) + 1
        for i in range(max(count, 0)):
            values.append(first + i * step)
        return values

    def parse_add(self) -> Any:
        """Parse an additive expression.

        Returns:
            The evaluated value.
        """
        left = self.parse_mul()
        while self.at_op("+", "-"):
            op = self.take().text
            left = _binary(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> Any:
        """Parse a multiplicative expression.

        Returns:
            The evaluated value.
        """
        left = self.parse_unary()
        while self.at_op("*", "/", ".*", "./"):
            op = self.take().text
            left = _binary(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Any:
        """Parse a unary plus or minus.

        Returns:
            The evaluated value.
        """
        if self.at_op("-"):
            self.take()
            return _binary("*", -1.0, self.parse_unary())
        if self.at_op("+"):
            self.take()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self) -> Any:
        """Parse an exponentiation.

        Returns:
            The evaluated value.
        """
        base = self.parse_postfix()
        if self.at_op("^", ".^"):
            op = self.take().text
            return _binary(op, base, self.parse_unary())
        return base

    def parse_postfix(self) -> Any:
        """Parse a primary followed by member access, calls or transpose.

        Returns:
            The evaluated value.
        """
        value = self.parse_primary()
        while True:
            if self.at_op("'", ".'"):
                self.take()
                continue
            if self.at_op("."):
                self.take()
                token = self.peek()
                if token is None or token.kind != "name":
                    return UNKNOWN
                self.take()
                if isinstance(value, dict):
                    value = value.get(token.text, UNKNOWN)
                else:
                    value = UNKNOWN
                continue
            if self.at_op("("):
                self._skip_group("(", ")")
                value = UNKNOWN
                continue
            if self.at_op("{"):
                self._skip_group("{", "}")
                value = UNKNOWN
                continue
            return value

    def _skip_group(self, opener: str, closer: str) -> None:
        """Consume a balanced bracket group without evaluating it.

        Args:
            opener: Opening bracket text.
            closer: Matching closing bracket text.
        """
        depth = 0
        while self.peek() is not None:
            token = self.take()
            if token.kind == "op" and token.text == opener:
                depth += 1
            elif token.kind == "op" and token.text == closer:
                depth -= 1
                if depth == 0:
                    return

    def parse_primary(self) -> Any:
        """Parse a literal, identifier, or bracketed expression.

        Returns:
            The evaluated value.
        """
        token = self.peek()
        if token is None:
            return UNKNOWN

        if token.kind == "number":
            self.take()
            return float(token.text)

        if token.kind == "string":
            self.take()
            return token.text

        if token.kind == "name":
            self.take()
            lowered = token.text.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            if self.at_op("("):
                # Function call or indexing: outside the supported grammar.
                self._skip_group("(", ")")
                return UNKNOWN
            return self.env.get(token.text, UNKNOWN)

        if token.kind == "op" and token.text == "(":
            self.take()
            value = self.parse_range()
            if self.at_op(")"):
                self.take()
            return value

        if token.kind == "op" and token.text == "[":
            self.take()
            return self._parse_matrix()

        if token.kind == "op" and token.text == "{":
            self.take()
            return self._parse_cell()

        self.take()
        return UNKNOWN

    def _parse_matrix(self) -> Any:
        """Parse a bracketed numeric vector.

        Rows separated by ``;`` are concatenated, matching the flat vectors
        INCA scripts use.  Elements may be separated by commas or by
        whitespace.

        Returns:
            A list of values, or :data:`UNKNOWN` if any element is unknown.
        """
        values: List[Any] = []
        while True:
            if self.peek() is None:
                return UNKNOWN
            if self.at_op("]"):
                self.take()
                break
            if self.at_op(",", ";"):
                self.take()
                continue
            element = self.parse_range()
            if element is UNKNOWN:
                return UNKNOWN
            values.extend(_as_list(element))
        return values

    def _parse_cell(self) -> Any:
        """Parse a brace-delimited cell array.

        Returns:
            A flat list of the cell contents, or :data:`UNKNOWN` if any
            element falls outside the supported grammar.
        """
        values: List[Any] = []
        while True:
            if self.peek() is None:
                return UNKNOWN
            if self.at_op("}"):
                self.take()
                break
            if self.at_op(",", ";"):
                self.take()
                continue
            element = self.parse_range()
            if element is UNKNOWN:
                return UNKNOWN
            values.append(element)
        return values


def evaluate(expr: str, env: Mapping[str, Any] | None = None) -> Any:
    """Evaluate a MATLAB expression against an environment.

    Supports numbers, character arrays, ``[...]`` vectors, ``{...}`` cell
    arrays, ``a:b`` and ``a:s:b`` ranges, the arithmetic operators with
    scalar broadcasting, ``true``/``false``, identifier lookup and struct
    member access.  Function calls and any other construct evaluate to
    :data:`UNKNOWN` rather than raising, so callers can skip statements they
    cannot reproduce.

    Args:
        expr: Expression source.
        env: Variable environment for identifier lookups.  Struct-valued
            entries should be dictionaries so that ``x.field`` resolves.

    Returns:
        A ``float``, ``str``, ``bool``, ``list``, or :data:`UNKNOWN`.
    """
    text = expr.strip()
    if not text:
        return UNKNOWN
    try:
        tokens = _tokenize(text)
    except ValueError:
        return UNKNOWN
    if not tokens:
        return UNKNOWN
    try:
        return _Parser(tokens, env or {}).parse()
    except (ValueError, ArithmeticError, TypeError):
        return UNKNOWN


def split_call(expr: str) -> Tuple[str, str] | None:
    """Split a bare function call into its name and argument source.

    Args:
        expr: Expression source, for example ``"reaction({'A -> B'})"``.

    Returns:
        A ``(name, arguments)`` tuple, or ``None`` if *expr* is not a single
        call spanning the whole expression.
    """
    match = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)$", expr.strip(), re.DOTALL)
    if match is None:
        return None
    name, args = match.group(1), match.group(2)
    # Reject cases like "f(1) + g(2)" where the trailing paren is not the
    # partner of the first one.
    depth = 0
    for index, char in enumerate(args):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
    return (name, args) if depth == 0 else None
