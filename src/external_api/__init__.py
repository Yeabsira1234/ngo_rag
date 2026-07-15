"""Approved read-only external API integrations."""

from src.external_api.client import OpenMeteoClient
from src.external_api.models import WeatherReport

__all__ = ["OpenMeteoClient", "WeatherReport"]
