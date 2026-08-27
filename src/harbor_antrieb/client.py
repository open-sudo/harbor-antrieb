from __future__ import annotations

import json
from typing import Any

import httpx

from harbor_antrieb.errors import ClusterExpiredError, is_cluster_expired


class AntriebMCPError(RuntimeError):
    """An MCP error that retains the provider's raw JSON-RPC payload."""

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.payload = payload


class AntriebClient:
    """Minimal streamable-HTTP client for the existing Antrieb MCP API."""

    def __init__(self, endpoint: str, token: str, timeout_sec: float = 600) -> None:
        self._endpoint = endpoint
        self._client = httpx.AsyncClient(
            timeout=timeout_sec,
            headers={"Authorization": f"Bearer {token}"},
        )
        self._mcp_session_id: str | None = None
        self._request_id = 0

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id
        response = await self._client.post(
            self._endpoint,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                error_payload: Any = response.json()
            except (ValueError, json.JSONDecodeError):
                error_payload = response.text
            if response.status_code == 410 or is_cluster_expired(error_payload):
                raise ClusterExpiredError(
                    "The Antrieb managed cluster lease expired"
                ) from exc
            raise
        self._mcp_session_id = response.headers.get(
            "Mcp-Session-Id", self._mcp_session_id
        )
        payload = self._decode_response(response)
        if "error" in payload:
            if is_cluster_expired(payload["error"]):
                raise ClusterExpiredError("The Antrieb managed cluster lease expired")
            raise AntriebMCPError(
                f"Antrieb MCP error: {payload['error']}", payload
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Antrieb MCP returned a malformed result")
        return result

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response.json()
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line.removeprefix("data:").strip())
        raise RuntimeError("Antrieb MCP returned an empty event stream")

    async def initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "harbor_antrieb", "version": "0.1.0"},
            },
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.call_tool_raw(name, arguments)
        return self.parse_tool_result(name, result)

    @staticmethod
    def parse_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Parse the JSON payload returned in an MCP tool result."""
        content = result.get("content", [])
        if not content or not isinstance(content[0], dict):
            raise RuntimeError(f"Antrieb tool {name!r} returned no content")
        text = content[0].get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Antrieb tool {name!r} returned non-text content")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Antrieb tool {name!r} returned malformed JSON")
        return parsed

    async def call_tool_raw(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the unmodified MCP tool result for protocol proxying."""
        if self._mcp_session_id is None:
            await self.initialize()
        return await self._request("tools/call", {"name": name, "arguments": arguments})
