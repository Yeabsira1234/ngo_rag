import pytest

from src.sql.generation import GeneratedSQL
from src.sql.schema import APPROVED_SCHEMA
from src.sql.validation import (
    SQLPrivacyError,
    SQLUnapprovedObjectError,
    SQLUnsafeQueryError,
    SQLValidator,
)


validator = SQLValidator(APPROVED_SCHEMA, max_rows=100)


def candidate(sql: str, parameters=None) -> GeneratedSQL:
    return GeneratedSQL(
        sql=sql,
        parameters=parameters or [],
        intent_summary="Safe aggregate",
    )


def test_safe_aggregate_and_approved_join_are_accepted() -> None:
    approved = validator.validate(candidate(
        "SELECT TOP (100) o.OfficeName, COUNT_BIG(*) AS ProgramCount "
        "FROM dbo.Offices AS o INNER JOIN dbo.Programs AS p "
        "ON p.OfficeID = o.OfficeID WHERE p.IsActive = ? "
        "GROUP BY o.OfficeName ORDER BY ProgramCount DESC, o.OfficeName ASC",
        [True],
    ))
    assert approved.parameters == (True,)


@pytest.mark.parametrize("sql", [
    "SELECT TOP (10) x.Value FROM dbo.Secrets AS x ORDER BY x.Value",
    "SELECT TOP (10) o.SecretColumn FROM dbo.Offices AS o ORDER BY o.SecretColumn",
    "SELECT TOP (10) o.OfficeName FROM OtherDB.dbo.Offices AS o ORDER BY o.OfficeName",
    "SELECT TOP (10) s.name FROM sys.tables AS s ORDER BY s.name",
])
def test_unapproved_tables_columns_and_catalogs_are_rejected(sql: str) -> None:
    with pytest.raises(SQLUnapprovedObjectError):
        validator.validate(candidate(sql))


@pytest.mark.parametrize("sql", [
    "SELECT TOP (10) o.OfficeName FROM dbo.Offices AS o; DELETE FROM dbo.Offices",
    "SELECT TOP (10) o.OfficeName FROM dbo.Offices AS o -- comment",
    "SELECT TOP (10) * FROM dbo.Offices AS o",
    "EXEC xp_cmdshell ?",
    "SELECT o.OfficeName FROM dbo.Offices AS o",
    "SELECT TOP (101) o.OfficeName FROM dbo.Offices AS o",
])
def test_unsafe_shapes_and_unbounded_results_are_rejected(sql: str) -> None:
    with pytest.raises(SQLUnsafeQueryError):
        validator.validate(candidate(sql))


def test_client_level_identifiers_and_row_dumps_are_rejected() -> None:
    with pytest.raises(SQLPrivacyError):
        validator.validate(candidate(
            "SELECT TOP (100) c.ClientID, c.PreferredLanguage "
            "FROM dbo.Clients AS c ORDER BY c.ClientID"
        ))


def test_client_aggregate_is_allowed() -> None:
    approved = validator.validate(candidate(
        "SELECT TOP (100) c.PreferredLanguage, COUNT_BIG(*) AS ClientCount "
        "FROM dbo.Clients AS c WHERE c.CaseStatus = ? "
        "GROUP BY c.PreferredLanguage "
        "ORDER BY ClientCount DESC, c.PreferredLanguage ASC",
        ["Open"],
    ))
    assert approved.parameters == ("Open",)


def test_internal_identifier_may_be_counted_but_not_returned() -> None:
    approved = validator.validate(candidate(
        "SELECT TOP (100) o.OfficeName, COUNT_BIG(p.ProgramID) AS ProgramCount "
        "FROM dbo.Offices AS o INNER JOIN dbo.Programs AS p "
        "ON p.OfficeID = o.OfficeID GROUP BY o.OfficeName "
        "ORDER BY ProgramCount DESC, o.OfficeName ASC"
    ))
    assert approved.sql.startswith("SELECT TOP")


def test_approved_schema_table_column_qualification_is_supported() -> None:
    approved = validator.validate(candidate(
        "SELECT TOP (1) dbo.Offices.OfficeName, COUNT_BIG(*) AS ProgramCount "
        "FROM dbo.Offices JOIN dbo.Programs "
        "ON dbo.Programs.OfficeID = dbo.Offices.OfficeID "
        "WHERE dbo.Programs.IsActive = ? "
        "GROUP BY dbo.Offices.OfficeName "
        "ORDER BY ProgramCount DESC, dbo.Offices.OfficeName ASC",
        [True],
    ))
    assert approved.parameters == (True,)
