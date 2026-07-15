from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SQLTable:
    name: str
    columns: tuple[str, ...]
    description: str
    relationship_only_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SQLSchemaCatalog:
    tables: tuple[SQLTable, ...]
    relationships: tuple[str, ...]

    @property
    def table_map(self) -> dict[str, SQLTable]:
        return {table.name.casefold(): table for table in self.tables}

    def prompt_text(self) -> str:
        table_lines = [
            f"dbo.{table.name}: {', '.join(table.columns)}. {table.description}"
            + (
                " Relationship-only columns (JOIN/ON only; never SELECT, GROUP BY, "
                f"or ORDER BY): {', '.join(table.relationship_only_columns)}."
                if table.relationship_only_columns
                else ""
            )
            for table in self.tables
        ]
        return "\n".join(table_lines + ["Relationships:", *self.relationships])


APPROVED_SCHEMA = SQLSchemaCatalog(
    tables=(
        SQLTable("Offices", ("OfficeID", "OfficeName", "City", "StateCode", "Phone", "Email", "IsActive"), "Office directory.", ("OfficeID",)),
        SQLTable("Programs", ("ProgramID", "ProgramName", "Category", "Description", "OfficeID", "StartDate", "IsActive"), "Programs offered by offices.", ("ProgramID", "OfficeID")),
        SQLTable("Staff", ("StaffID", "FirstName", "LastName", "JobTitle", "Department", "OfficeID", "ManagerID", "HireDate", "IsActive"), "Staff assignments; staff names may be returned for operational questions.", ("StaffID", "OfficeID", "ManagerID")),
        SQLTable("Clients", ("ClientID", "EnrollmentDate", "PreferredLanguage", "City", "StateCode", "CaseStatus"), "Aggregate client demographics only; never return client-level identifiers.", ("ClientID",)),
        SQLTable("Cases", ("CaseID", "ClientID", "ProgramID", "AssignedStaffID", "OpenDate", "CloseDate", "PriorityLevel", "CaseStatus"), "Case operations; expose aggregates rather than identifiers.", ("CaseID", "ClientID", "ProgramID", "AssignedStaffID")),
        SQLTable("ServiceEvents", ("ServiceEventID", "CaseID", "ServiceType", "ServiceDate", "DurationMinutes", "Outcome"), "Service activity without client identifiers.", ("ServiceEventID", "CaseID")),
        SQLTable("Referrals", ("ReferralID", "ClientID", "ReferralDate", "ReferredTo", "ReferralReason", "ReferralStatus"), "Referral aggregates and operational categories.", ("ReferralID", "ClientID")),
    ),
    relationships=(
        "Programs.OfficeID = Offices.OfficeID",
        "Staff.OfficeID = Offices.OfficeID; Staff.ManagerID = Staff.StaffID",
        "Cases.ClientID = Clients.ClientID; Cases.ProgramID = Programs.ProgramID; Cases.AssignedStaffID = Staff.StaffID",
        "ServiceEvents.CaseID = Cases.CaseID",
        "Referrals.ClientID = Clients.ClientID",
    ),
)
