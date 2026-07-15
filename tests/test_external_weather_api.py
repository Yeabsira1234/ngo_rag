import httpx
import pytest

from src.external_api.client import (
    OpenMeteoClient,
    WeatherAPIResponseError,
    WeatherAPIUnavailableError,
    WeatherLocationNotFoundError,
)


GEOCODING = {
    "results": [
        {
            "name": "Arlington",
            "latitude": 38.88,
            "longitude": -77.10,
            "country": "United States",
            "admin1": "Virginia",
        }
    ]
}
FORECAST = {
    "timezone": "America/New_York",
    "current": {
        "time": "2026-07-15T14:00",
        "temperature_2m": 28.5,
        "apparent_temperature": 30.1,
        "weather_code": 2,
        "wind_speed_10m": 11.4,
    },
    "daily": {
        "time": ["2026-07-15"],
        "temperature_2m_max": [31.0],
        "temperature_2m_min": [22.0],
        "precipitation_probability_max": [35],
    },
}


def client_with(handler, *, retries: int = 2) -> OpenMeteoClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=transport,
        headers={"User-Agent": OpenMeteoClient.USER_AGENT},
    )
    return OpenMeteoClient(max_retries=retries, client=http_client)


def test_success_uses_only_approved_get_endpoints_and_returns_typed_report() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = GEOCODING if "geocoding-api" in request.url.host else FORECAST
        return httpx.Response(200, json=payload)

    report = client_with(handler).weather_for_city("  Arlington, Virginia ")

    assert [request.method for request in requests] == ["GET", "GET"]
    assert [request.url.host for request in requests] == [
        "geocoding-api.open-meteo.com",
        "api.open-meteo.com",
    ]
    assert requests[0].headers["user-agent"] == OpenMeteoClient.USER_AGENT
    assert report.location.name == "Arlington"
    assert report.temperature_c == 28.5
    assert report.high_temperature_c == 31.0
    assert report.to_model_output()["provider"] == "Open-Meteo"


def test_malformed_json_is_rejected_safely_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="not-json")

    with pytest.raises(WeatherAPIResponseError):
        client_with(handler).weather_for_city("Arlington")
    assert calls == 1


def test_timeout_is_retried_then_returns_safe_unavailable_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private timeout detail", request=request)

    with pytest.raises(WeatherAPIUnavailableError) as captured:
        client_with(handler, retries=2).weather_for_city("Arlington")
    assert calls == 3
    assert "private timeout detail" not in str(captured.value)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_errors_are_retried(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": True})

    with pytest.raises(WeatherAPIUnavailableError):
        client_with(handler, retries=1).weather_for_city("Arlington")
    assert calls == 2


def test_non_transient_4xx_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": True})

    with pytest.raises(WeatherAPIUnavailableError):
        client_with(handler).weather_for_city("Arlington")
    assert calls == 1


def test_empty_geocoding_results_are_a_typed_not_found() -> None:
    client = client_with(lambda request: httpx.Response(200, json={"results": []}))
    with pytest.raises(WeatherLocationNotFoundError):
        client.weather_for_city("Not A Real City")


@pytest.mark.parametrize("city", ["", "12345", "Arlington/../../host", "x" * 101])
def test_invalid_city_is_rejected_before_http(city: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=GEOCODING)

    with pytest.raises(ValueError):
        client_with(handler).weather_for_city(city)
    assert calls == 0


def test_unapproved_endpoint_is_rejected_before_http() -> None:
    client = client_with(lambda request: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="Unapproved"):
        client._get_json("unknown", "https://example.com/private", {})
