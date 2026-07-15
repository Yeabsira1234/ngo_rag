from dataclasses import dataclass
from typing import Any


class WeatherResponseError(ValueError):
    """Raised when an approved API returns an unsupported JSON shape."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeatherResponseError(f"{field} must be an object.")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WeatherResponseError(f"{field} must be non-empty text.")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherResponseError(f"{field} must be numeric.")
    return float(value)


def _first(values: Any, field: str) -> Any:
    if not isinstance(values, list) or not values:
        raise WeatherResponseError(f"{field} must be a non-empty array.")
    return values[0]


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    name: str
    latitude: float
    longitude: float
    country: str
    administrative_area: str | None = None

    @classmethod
    def from_geocoding_json(cls, payload: Any) -> "ResolvedLocation":
        root = _mapping(payload, "geocoding response")
        results = root.get("results")
        item = _mapping(_first(results, "results"), "results[0]")
        admin = item.get("admin1")
        if admin is not None and not isinstance(admin, str):
            raise WeatherResponseError("admin1 must be text when present.")
        return cls(
            name=_text(item.get("name"), "name"),
            latitude=_number(item.get("latitude"), "latitude"),
            longitude=_number(item.get("longitude"), "longitude"),
            country=_text(item.get("country"), "country"),
            administrative_area=admin.strip() if admin else None,
        )


@dataclass(frozen=True, slots=True)
class WeatherReport:
    location: ResolvedLocation
    observed_at: str
    timezone: str
    temperature_c: float
    apparent_temperature_c: float
    weather_code: int
    wind_speed_kmh: float
    forecast_date: str
    high_temperature_c: float
    low_temperature_c: float
    precipitation_probability_percent: float

    @property
    def conditions(self) -> str:
        if self.weather_code == 0:
            return "Clear sky"
        if self.weather_code in {1, 2, 3}:
            return "Mainly clear to overcast"
        if self.weather_code in {45, 48}:
            return "Fog"
        if self.weather_code in {51, 53, 55, 56, 57}:
            return "Drizzle"
        if self.weather_code in {61, 63, 65, 66, 67, 80, 81, 82}:
            return "Rain"
        if self.weather_code in {71, 73, 75, 77, 85, 86}:
            return "Snow"
        if self.weather_code in {95, 96, 99}:
            return "Thunderstorm"
        return "Unknown conditions"

    @classmethod
    def from_forecast_json(
        cls, location: ResolvedLocation, payload: Any
    ) -> "WeatherReport":
        root = _mapping(payload, "forecast response")
        current = _mapping(root.get("current"), "current")
        daily = _mapping(root.get("daily"), "daily")
        weather_code = _number(current.get("weather_code"), "weather_code")
        if not weather_code.is_integer():
            raise WeatherResponseError("weather_code must be an integer.")
        return cls(
            location=location,
            observed_at=_text(current.get("time"), "current.time"),
            timezone=_text(root.get("timezone"), "timezone"),
            temperature_c=_number(current.get("temperature_2m"), "temperature_2m"),
            apparent_temperature_c=_number(
                current.get("apparent_temperature"), "apparent_temperature"
            ),
            weather_code=int(weather_code),
            wind_speed_kmh=_number(current.get("wind_speed_10m"), "wind_speed_10m"),
            forecast_date=_text(_first(daily.get("time"), "daily.time"), "daily.time[0]"),
            high_temperature_c=_number(
                _first(daily.get("temperature_2m_max"), "temperature_2m_max"),
                "temperature_2m_max[0]",
            ),
            low_temperature_c=_number(
                _first(daily.get("temperature_2m_min"), "temperature_2m_min"),
                "temperature_2m_min[0]",
            ),
            precipitation_probability_percent=_number(
                _first(
                    daily.get("precipitation_probability_max"),
                    "precipitation_probability_max",
                ),
                "precipitation_probability_max[0]",
            ),
        )

    def to_model_output(self) -> dict[str, Any]:
        place = self.location.name
        if self.location.administrative_area:
            place += f", {self.location.administrative_area}"
        place += f", {self.location.country}"
        return {
            "provider": "Open-Meteo",
            "location": place,
            "observed_at": self.observed_at,
            "timezone": self.timezone,
            "current": {
                "temperature_c": self.temperature_c,
                "apparent_temperature_c": self.apparent_temperature_c,
                "weather_code": self.weather_code,
                "conditions": self.conditions,
                "wind_speed_kmh": self.wind_speed_kmh,
            },
            "today": {
                "date": self.forecast_date,
                "high_temperature_c": self.high_temperature_c,
                "low_temperature_c": self.low_temperature_c,
                "precipitation_probability_percent": (
                    self.precipitation_probability_percent
                ),
            },
        }
