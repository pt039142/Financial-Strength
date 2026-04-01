import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Test that the API health check returns a 200 status."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"