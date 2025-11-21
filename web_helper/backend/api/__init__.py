"""API router modules."""

from fastapi import APIRouter

from .artifacts import router as artifacts_router
from .column_mappings import router as column_mappings_router
from .components import router as components_router
from .configs import router as configs_router
from .correlations import router as correlations_router
from .devices import router as devices_router
from .events import router as events_router
from .metrics import router as metrics_router
from .outliers import router as outliers_router
from .projects import router as projects_router
from .queue import router as queue_router
from .recommendations import router as recommendations_router
from .runs import router as runs_router
from .sync import router as sync_router
from .trends import router as trends_router


def create_router() -> APIRouter:
    """Create the main API router with all sub-routers."""
    api_router = APIRouter(prefix="/api")

    # Include all sub-routers
    api_router.include_router(projects_router, tags=["projects"])
    api_router.include_router(runs_router, tags=["runs"])
    api_router.include_router(devices_router, tags=["devices"])
    api_router.include_router(queue_router, tags=["queue"])
    api_router.include_router(events_router, tags=["events"])
    api_router.include_router(components_router, tags=["components"])
    api_router.include_router(configs_router, tags=["configs"])
    api_router.include_router(metrics_router, tags=["metrics"])
    api_router.include_router(column_mappings_router, tags=["column_mappings"])
    api_router.include_router(outliers_router, tags=["outliers"])
    api_router.include_router(trends_router, tags=["trends"])
    api_router.include_router(recommendations_router, tags=["recommendations"])
    api_router.include_router(sync_router, tags=["sync"])
    api_router.include_router(correlations_router, tags=["correlations"])
    api_router.include_router(artifacts_router, tags=["artifacts"])
    # Note: experiments router removed - all experiment execution now goes through queue

    # Root API endpoint
    @api_router.get("/")
    async def root():
        return {
            "message": "CVLab-Kit Web Helper",
            "version": "1.0.0",
            "status": "running",
        }

    return api_router
