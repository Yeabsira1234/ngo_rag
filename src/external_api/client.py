import logging
import time
from typing import Any

import httpx

from src.external_api.models import (
    ResolvedLocation,
    WeatherReport,
    WeatherResponseError,
)

logger = logging.getLogger(__name__)


class WeatherAPIError(RuntimeError):
    """Safe base error for the approved weather integration."""


class WeatherLocationNotFoundError(WeatherAPIError):
    pass


class WeatherAPIUnavailableError(WeatherAPIError):
    pass


class WeatherAPIResponseError(WeatherAPIError):
    pass


class OpenMeteoClient:
    """Synchronous GET-only client restricted to approved Open-Meteo endpoints."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    USER_AGENT = "ngo-rag-weather/1.0"

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": self.USER_AGENT, "Accept": "application/json"},
        )

    def weather_for_city(self, city: str) -> WeatherReport:
        normalized_city = self._validate_city(city)
        geocoding = self._get_json(
            "geocoding",
            self.GEOCODING_URL,
            {"name": normalized_city, "count": 1, "language": "en", "format": "json"},
        )
        try:
            location = ResolvedLocation.from_geocoding_json(geocoding)
        except WeatherResponseError as error:
            if isinstance(geocoding, dict) and geocoding.get("results") in (None, []):
                raise WeatherLocationNotFoundError(
                    "No approved weather location was found."
                ) from error
            raise WeatherAPIResponseError(
                "The weather service returned invalid location data."
            ) from error

        forecast = self._get_json(
            "forecast",
            self.FORECAST_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": (
                    "temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 1,
            },
        )
        try:
            return WeatherReport.from_forecast_json(location, forecast)
        except WeatherResponseError as error:
            raise WeatherAPIResponseError(
                "The weather service returned invalid forecast data."
            ) from error

    @staticmethod
    def _validate_city(city: str) -> str:
        if not isinstance(city, str):
            raise ValueError("city must be a string.")
        normalized = " ".join(city.split())
        if not 1 <= len(normalized) <= 100:
            raise ValueError("city must contain between 1 and 100 characters.")
        if not all(character.isalnum() or character in " .,'-" for character in normalized):
            raise ValueError("city contains unsupported characters.")
        if not any(character.isalpha() for character in normalized):
            raise ValueError("city must contain letters.")
        return normalized

    def _get_json(
        self, endpoint: str, url: str, params: dict[str, Any]
    ) -> Any:
        if url not in {self.GEOCODING_URL, self.FORECAST_URL}:
            raise ValueError("Unapproved external API endpoint.")
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            status_code: int | None = None
            try:
                response = self.client.get(
                    url, params=params, timeout=self.timeout_seconds
                )
                status_code = response.status_code
                latency_ms = round((time.monotonic() - started) * 1000)
                if response.status_code == 429 or response.status_code >= 500:
                    logger.warning(
                        "event=external_api_request endpoint=%s status=%d latency_ms=%d outcome=transient_failure",
                        endpoint,
                        response.status_code,
                        latency_ms,
                    )
                    if attempt < self.max_retries:
                        continue
                    raise WeatherAPIUnavailableError(
                        "The weather service is temporarily unavailable."
                    )
                if response.status_code >= 400:
                    logger.warning(
                        "event=external_api_request endpoint=%s status=%d latency_ms=%d outcome=http_failure",
                        endpoint,
                        response.status_code,
                        latency_ms,
                    )
                    raise WeatherAPIUnavailableError(
                        "The weather request could not be completed."
                    )
                try:
                    payload = response.json()
                except (ValueError, TypeError) as error:
                    logger.warning(
                        "event=external_api_request endpoint=%s status=%d latency_ms=%d outcome=invalid_json",
                        endpoint,
                        response.status_code,
                        latency_ms,
                    )
                    raise WeatherAPIResponseError(
                        "The weather service returned invalid data."
                    ) from error
                logger.info(
                    "event=external_api_request endpoint=%s status=%d latency_ms=%d outcome=success",
                    endpoint,
                    response.status_code,
                    latency_ms,
                )
                return payload
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                latency_ms = round((time.monotonic() - started) * 1000)
                logger.warning(
                    "event=external_api_request endpoint=%s status=%s latency_ms=%d outcome=network_failure",
                    endpoint,
                    status_code if status_code is not None else "none",
                    latency_ms,
                )
                if attempt >= self.max_retries:
                    raise WeatherAPIUnavailableError(
                        "The weather service is temporarily unavailable."
                    ) from error
        raise AssertionError("Weather retry loop exited unexpectedly.")
