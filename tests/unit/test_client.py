from typing import Any

import httpx
import pytest

from harbor_antrieb.client import AntriebClient
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
