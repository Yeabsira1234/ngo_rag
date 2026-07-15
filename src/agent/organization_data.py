from dataclasses import dataclass
from enum import Enum


class OrganizationInfoCategory(str, Enum):
    ORGANIZATION_NAME = "organization_name"
    SUPPORT_OFFICE_HOURS = "support_office_hours"
    GENERAL_CONTACT_EMAIL = "general_contact_email"
    MAIN_OFFICE_LOCATION = "main_office_location"
    SERVICE_CATEGORIES = "service_categories"


@dataclass(frozen=True, slots=True)
class SampleOrganizationInfo:
    """Fictional structured organization data used by the sample agent tool."""

    organization_name: str
    support_office_hours: str
    general_contact_email: str
    main_office_location: str
    service_categories: tuple[str, ...]

    def value_for(self, category: OrganizationInfoCategory) -> str:
        value = getattr(self, category.value)
        if isinstance(value, tuple):
            return ", ".join(value)
        return value


SAMPLE_ORGANIZATION_INFO = SampleOrganizationInfo(
    organization_name="Community Support Network",
    support_office_hours="Monday through Friday, 9:00 a.m. to 5:00 p.m.",
    general_contact_email="hello@example.org",
    main_office_location="100 Example Avenue, Example City, EX 00000",
    service_categories=(
        "Case Management",
        "Community Education",
        "Employment Support",
        "Language Assistance",
    ),
)
