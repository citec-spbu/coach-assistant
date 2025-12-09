import pytest
import httpx

FASTAPI_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_send_in_progress():
    async with httpx.AsyncClient(base_url=FASTAPI_URL) as client:
        resp = await client.post("/api/send/", json={
            "upload_url": "http://localhost:3000/uploads/test.mp4"
        })
    assert resp.status_code == 204
