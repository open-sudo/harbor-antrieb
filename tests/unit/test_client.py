from typing import Any

import httpx
import pytest

from harbor_antrieb.client import AntriebClient, AntriebMCPError
from harbor_antrieb.errors import ClusterExpiredError


@pytest.mark.asyncio
async def test_client_classifies_cluster_expired_protocol_error() -> None:
    class FakeHTTPClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32000, "message": "cluster_expired"},
                },
                request=httpx.Request("POST", "https://antrieb.sh/mcp"),
            )

        async def aclose(self) -> None:
            pass

    client = AntriebClient("https://antrieb.sh/mcp", "test-token")
    await client.close()
    client._client = FakeHTTPClient()  # ty: ignore[invalid-assignment]
    client._mcp_session_id = "mcp-session"

    with pytest.raises(ClusterExpiredError, match="cluster lease expired"):
        await client.call_tool_raw("exec", {"session_id": "expired"})


@pytest.mark.asyncio
async def test_client_surfaces_only_the_provider_message_for_protocol_errors() -> None:
    class FakeHTTPClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {
                        "code": -32000,
                        "message": "We cannot satisfy request at this time",
                        "data": {"internal": "scheduler payload"},
                    },
                },
                request=httpx.Request("POST", "https://antrieb.sh/mcp"),
            )

        async def aclose(self) -> None:
            pass

    client = AntriebClient("https://antrieb.sh/mcp", "test-token")
    await client.close()
    client._client = FakeHTTPClient()  # ty: ignore[invalid-assignment]
    client._mcp_session_id = "mcp-session"

    with pytest.raises(AntriebMCPError) as raised:
        await client.call_tool_raw("provision", {"cluster": ["alma9 x4"]})

    assert str(raised.value) == "We cannot satisfy request at this time"
    assert raised.value.payload["error"]["data"]["internal"] == "scheduler payload"


@pytest.mark.asyncio
async def test_client_preserves_http_error_body_without_dumping_it_in_message() -> None:
    class FakeHTTPClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> httpx.Response:
            return httpx.Response(
                503,
                json={
                    "message": "No matching cluster capacity is available",
                    "debug": {"candidates": ["host1", "host2"]},
                },
                request=httpx.Request("POST", "https://antrieb.sh/mcp"),
            )

        async def aclose(self) -> None:
            pass

    client = AntriebClient("https://antrieb.sh/mcp", "test-token")
    await client.close()
    client._client = FakeHTTPClient()  # ty: ignore[invalid-assignment]
    client._mcp_session_id = "mcp-session"

    with pytest.raises(AntriebMCPError) as raised:
        await client.call_tool_raw("provision", {"cluster": ["alma9 x4"]})

    assert str(raised.value) == "No matching cluster capacity is available"
    assert raised.value.payload["http_status"] == 503
