"""Local web application entry point."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="FermaYT")
app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the local application homepage."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "FermaYT"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Return application health status."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
