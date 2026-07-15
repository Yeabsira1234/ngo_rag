import re
from dataclasses import dataclass
from typing import Any

from src.sql.generation import GeneratedSQL
from src.sql.schema import SQLSchemaCatalog


class SQLValidationError(RuntimeError): pass
class SQLUnsafeQueryError(SQLValidationError): pass
class SQLUnapprovedObjectError(SQLValidationError): pass
class SQLPrivacyError(SQLValidationError): pass


TOKEN = re.compile(
    r"\s+|(?P<string>N?'(?:''|[^'])*')|(?P<bracket>\[[^\]]+\])|"
    r"(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<number>\d+(?:\.\d+)?)|"
    r"(?P<parameter>\?)|(?P<operator><>|!=|<=|>=|[=<>+*/-])|(?P<punct>[(),.;])"
)

KEYWORDS = {
    "select", "top", "distinct", "from", "as", "inner", "left", "right",
    "full", "outer", "join", "on", "where", "and", "or", "not", "is",
    "null", "group", "by", "having", "order", "asc", "desc", "case", "when",
    "then", "else", "end", "like", "in", "between", "dateadd", "day", "getdate",
}
FUNCTIONS = {"count", "count_big", "sum", "avg", "min", "max", "coalesce"}
DANGEROUS = {
    "insert", "update", "delete", "merge", "drop", "alter", "truncate", "create",
    "grant", "revoke", "deny", "exec", "execute", "openrowset", "opendatasource",
    "bulk", "declare", "set", "use", "into", "union", "intersect", "except",
    "waitfor", "backup", "restore", "with",
}


@dataclass(frozen=True, slots=True)
class ApprovedSQL:
    sql: str
    parameters: tuple[Any, ...]


class SQLValidator:
    def __init__(self, schema: SQLSchemaCatalog, max_rows: int) -> None:
        self.schema = schema
        self.max_rows = max_rows

    def validate(self, candidate: GeneratedSQL) -> ApprovedSQL:
        sql = candidate.sql.strip()
        if "--" in sql or "/*" in sql or "*/" in sql:
            raise SQLUnsafeQueryError("SQL comments are not allowed.")
        tokens = self._tokens(sql)
        lowered = [token.casefold() for token in tokens]
        if not tokens or lowered[0] != "select":
            raise SQLUnsafeQueryError("Only SELECT statements are allowed.")
        if any(token in DANGEROUS or token.startswith(("xp_", "sp_")) for token in lowered):
            raise SQLUnsafeQueryError("Unsafe SQL construct rejected.")
        if any(token in {"sys", "information_schema"} for token in lowered):
            raise SQLUnapprovedObjectError("System catalogs are not approved.")
        semicolons = [index for index, token in enumerate(tokens) if token == ";"]
        if len(semicolons) > 1 or (semicolons and semicolons[0] != len(tokens) - 1):
            raise SQLUnsafeQueryError("Multiple statements are not allowed.")
        if any(token.startswith("'") or token.casefold().startswith("n'") for token in tokens):
            raise SQLUnsafeQueryError("Literal strings must use parameters.")
        self._require_top(tokens, lowered)
        if lowered.count("?") != len(candidate.parameters):
            raise SQLValidationError("Parameter count does not match placeholders.")
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in candidate.parameters):
            raise SQLValidationError("Unsupported parameter type.")
        aliases, table_by_alias = self._approved_tables(tokens, lowered)
        self._validate_columns(tokens, lowered, aliases, table_by_alias)
        self._enforce_privacy(tokens, lowered, table_by_alias)
        return ApprovedSQL(sql, tuple(candidate.parameters))

    @staticmethod
    def _tokens(sql: str) -> list[str]:
        result: list[str] = []
        position = 0
        for match in TOKEN.finditer(sql):
            if sql[position:match.start()].strip():
                raise SQLUnsafeQueryError("Unsupported SQL syntax.")
            position = match.end()
            text = match.group(0)
            if text.isspace():
                continue
            result.append(text[1:-1] if match.lastgroup == "bracket" else text)
        if sql[position:].strip():
            raise SQLUnsafeQueryError("Unsupported SQL syntax.")
        return result

    def _require_top(self, tokens: list[str], lowered: list[str]) -> None:
        if len(tokens) < 3 or lowered[1] != "top":
            raise SQLUnsafeQueryError("A bounded TOP clause is required.")
        if tokens[2] == "(":
            if len(tokens) < 5 or not tokens[3].isdigit() or tokens[4] != ")":
                raise SQLUnsafeQueryError("TOP must use a fixed numeric limit.")
            limit = int(tokens[3])
        elif tokens[2].isdigit():
            limit = int(tokens[2])
        else:
            raise SQLUnsafeQueryError("TOP must use a fixed numeric limit.")
        if limit <= 0 or limit > self.max_rows:
            raise SQLUnsafeQueryError("TOP exceeds the configured row limit.")

    def _approved_tables(self, tokens, lowered):
        aliases: dict[str, str] = {}
        tables = self.schema.table_map
        for index, token in enumerate(lowered):
            if token not in {"from", "join"}:
                continue
            cursor = index + 1
            if cursor + 2 >= len(tokens) or lowered[cursor] != "dbo" or tokens[cursor + 1] != ".":
                raise SQLUnapprovedObjectError("Only approved dbo tables may be queried.")
            table_name = lowered[cursor + 2]
            if table_name not in tables:
                raise SQLUnapprovedObjectError("Unapproved table rejected.")
            alias = table_name
            cursor += 3
            if cursor < len(tokens) and lowered[cursor] == "as":
                cursor += 1
            if cursor < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[cursor]) and lowered[cursor] not in KEYWORDS:
                alias = lowered[cursor]
            aliases[alias] = table_name
            aliases[table_name] = table_name
        if not aliases:
            raise SQLUnapprovedObjectError("An approved table is required.")
        return aliases, aliases

    def _validate_columns(self, tokens, lowered, aliases, table_by_alias):
        tables = self.schema.table_map
        for index in range(len(tokens) - 2):
            if tokens[index + 1] != ".":
                continue
            qualifier = lowered[index]
            column = lowered[index + 2]
            if qualifier == "dbo":
                if column not in tables:
                    raise SQLUnapprovedObjectError("Unapproved table rejected.")
                continue
            table_name = table_by_alias.get(qualifier)
            if table_name is None:
                raise SQLUnapprovedObjectError("Cross-database or unknown qualifier rejected.")
            approved = {name.casefold() for name in tables[table_name].columns}
            if column not in approved:
                raise SQLUnapprovedObjectError("Unapproved column rejected.")
        approved_columns = {
            column.casefold()
            for table in tables.values()
            for column in table.columns
        }
        output_aliases = {
            lowered[index + 1]
            for index, token in enumerate(lowered[:-1])
            if token == "as"
        }
        known = (
            KEYWORDS | FUNCTIONS | set(tables) | set(aliases)
            | approved_columns | output_aliases | {"dbo"}
        )
        for index, token in enumerate(tokens):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                continue
            if lowered[index] not in known:
                raise SQLUnapprovedObjectError("Unapproved column or identifier rejected.")
        # Three-part names represent cross-database/server access.
        for index in range(len(tokens) - 4):
            if tokens[index + 1] == "." and tokens[index + 3] == ".":
                schema_name = lowered[index]
                table_name = lowered[index + 2]
                column_name = lowered[index + 4]
                is_approved_three_part_column = (
                    schema_name == "dbo"
                    and table_name in tables
                    and column_name
                    in {name.casefold() for name in tables[table_name].columns}
                )
                if not is_approved_three_part_column:
                    raise SQLUnapprovedObjectError(
                        "Cross-database references are not allowed."
                    )
        # A star is valid only inside COUNT(*) or COUNT_BIG(*).
        for index, token in enumerate(tokens):
            if token != "*":
                continue
            if index < 2 or lowered[index - 2] not in {"count", "count_big"} or tokens[index - 1] != "(":
                raise SQLUnsafeQueryError("SELECT * is not allowed.")

    def _enforce_privacy(self, tokens, lowered, table_by_alias):
        from_index = lowered.index("from")
        projection = lowered[:from_index]
        tables = self.schema.table_map
        for index in range(len(projection) - 2):
            if projection[index + 1] != ".":
                continue
            table_name = table_by_alias.get(projection[index])
            if table_name is None:
                continue
            restricted = {name.casefold() for name in tables[table_name].relationship_only_columns}
            if projection[index + 2] in restricted:
                inside_aggregate = (
                    index >= 2
                    and projection[index - 1] == "("
                    and projection[index - 2] in FUNCTIONS
                )
                if not inside_aggregate:
                    raise SQLPrivacyError("Internal identifiers cannot be returned.")
        if "clients" in table_by_alias.values() and not any(fn in lowered for fn in FUNCTIONS):
            raise SQLPrivacyError("Client-level records are not permitted.")
