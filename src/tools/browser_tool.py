"""
browser_tool.py — Playwright-based browser automation tool (Large Action Model).

This module demonstrates LAM (Large Action Model) behaviour: the agent can
navigate pages, extract structured metadata, and take screenshots — actions
that go beyond simple HTTP requests.

Playwright is optional.  If it is not installed or the browser binary is
missing, every function degrades gracefully and logs a warning.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from src.config import config

logger = logging.getLogger(__name__)

# ── System chromium detection ─────────────────────────────────────────────────
# On Railway (and other containerised environments) Playwright's own binary may
# not be available if the build cache is cleared between phases.  We prefer the
# system-installed chromium from nixpkgs (added to PATH by the nix setup phase)
# and fall back to whatever Playwright downloaded, if anything.
_SYSTEM_CHROME: str | None = (
    shutil.which("chromium-browser")
    or shutil.which("chromium")
    or shutil.which("google-chrome-stable")
    or shutil.which("google-chrome")
    or None
)
if _SYSTEM_CHROME:
    logger.info("Browser tool: using system chromium at %s", _SYSTEM_CHROME)
else:
    logger.info("Browser tool: no system chromium found — will use Playwright's bundled binary.")

TOOL_SPEC = {
    "name": "browser_navigate",
    "description": (
        "Use a real browser to open a URL, extract metadata, and take a screenshot. "
        "Useful for JS-rendered pages that plain HTTP cannot access."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "screenshot": {"type": "boolean", "default": True},
        },
        "required": ["url"],
    },
}


async def _navigate_async(url: str, take_screenshot: bool = True) -> dict[str, Any]:
    """
    Core async Playwright logic.

    1. Launch headless Chromium.
    2. Navigate to the URL with a timeout guard.
    3. Extract: title, meta description, Open Graph tags, h1–h2 headings, first 2 000 chars of text.
    4. Optionally take a screenshot (saved to generated_images/).

    This is the LAM core: the model can instruct the browser to perform
    structured actions (navigate, extract, screenshot) rather than just
    issuing HTTP GET requests.
    """
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError:
        logger.warning("Playwright is not installed — browser actions unavailable.")
        return {"error": "playwright_not_installed", "url": url}

    result: dict[str, Any] = {"url": url, "screenshot_path": None}

    async with async_playwright() as p:
        # Build launch kwargs — must include sandbox-disabling flags for
        # containerised environments (Docker / Railway / Render / etc.).
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
            ],
        }
        if _SYSTEM_CHROME:
            launch_kwargs["executable_path"] = _SYSTEM_CHROME

        try:
            browser = await p.chromium.launch(**launch_kwargs)
        except Exception as launch_exc:
            logger.warning(
                "Chromium launch failed (%s).  "
                "Browser agent will be skipped for this URL.", launch_exc,
            )
            result["error"] = f"Browser unavailable: {launch_exc}"
            return result

        page = await browser.new_page()

        try:
            # Navigate with a 30-second timeout
            await page.goto(url, timeout=30_000, wait_until="domcontentloaded")

            # ── Extract metadata ──────────────────────────────────────────────
            result["title"] = await page.title()

            # Read Open Graph / standard meta tags
            for meta_name in ("description", "og:description", "og:title", "og:type"):
                val = await page.evaluate(
                    f"""() => {{
                        const el = document.querySelector('meta[name="{meta_name}"], meta[property="{meta_name}"]');
                        return el ? el.content : null;
                    }}"""
                )
                if val:
                    result[meta_name.replace(":", "_")] = val

            # Collect headings for structural analysis
            headings = await page.evaluate(
                """() => Array.from(document.querySelectorAll('h1,h2'))
                              .map(h => h.innerText.trim())
                              .filter(t => t.length > 0)
                              .slice(0, 10)"""
            )
            result["headings"] = headings

            # Body text (first 2 000 chars)
            body_text = await page.evaluate(
                """() => {
                    const body = document.querySelector('article') || document.body;
                    return body ? body.innerText.slice(0, 2000) : '';
                }"""
            )
            result["body_text"] = body_text

            # ── Screenshot ────────────────────────────────────────────────────
            if take_screenshot:
                slug = url.split("//")[-1].replace("/", "_")[:60]
                screenshot_path = config.images_dir / f"browser_{slug}.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
                result["screenshot_path"] = str(screenshot_path)
                logger.info("Screenshot saved: %s", screenshot_path)

        except Exception as exc:
            # Timeout, navigation error, etc. — record but don't crash
            logger.warning("Playwright navigation error for %s: %s", url, exc)
            result["error"] = str(exc)
        finally:
            await browser.close()

    return result


def browser_navigate(url: str, take_screenshot: bool = True) -> dict[str, Any]:
    """
    Synchronous wrapper around the async Playwright navigator.

    Runs the async code in a dedicated worker thread so it ALWAYS gets a
    fresh event loop — completely isolated from Streamlit's own event loop.
    Without this isolation, calling asyncio.run() from a Streamlit context
    that already has a running loop raises RuntimeError on the 2nd+ pipeline
    run, crashing the whole pipeline mid-stream.

    A hard 40-second wall-clock timeout prevents the browser from hanging
    indefinitely when the Playwright binary is missing or a site is very slow.
    """
    import concurrent.futures  # noqa: PLC0415

    def _run_in_thread() -> dict[str, Any]:
        # Worker thread has NO existing event loop → asyncio.run is always safe here.
        return asyncio.run(_navigate_async(url, take_screenshot))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            try:
                return future.result(timeout=40)          # hard 40-second cutoff
            except concurrent.futures.TimeoutError:
                logger.warning("browser_navigate timed out after 40s for %s", url)
                return {"url": url, "error": "Browser navigation timed out (>40s)"}
    except Exception as exc:
        logger.error("browser_navigate failed for %s: %s", url, exc)
        return {"url": url, "error": str(exc)}
