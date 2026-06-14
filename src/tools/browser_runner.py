"""
browser_runner.py — Isolated Playwright subprocess entry point.

Called by browser_tool.py as a child process:
    python src/tools/browser_runner.py <url> [screenshot_path]

Running Chromium in a child process means an OOM kill only terminates
this process — the parent Streamlit app keeps running and reads the
non-zero exit code as a graceful error.

Outputs a single JSON object on stdout.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

# Make project root importable so src.config works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def _run(url: str, screenshot_path: str | None) -> dict:
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError:
        return {"url": url, "error": "playwright_not_installed"}

    system_chrome = (
        shutil.which("chromium-browser")
        or shutil.which("chromium")
        or shutil.which("google-chrome-stable")
        or shutil.which("google-chrome")
    )

    result: dict = {"url": url, "screenshot_path": None}

    async with async_playwright() as p:
        launch_kwargs: dict = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
                # Single-process mode — dramatically cuts RAM usage on Railway
                # (avoids spawning separate renderer/GPU/network processes).
                "--single-process",
                "--no-zygote",
                "--disable-extensions",
                "--disable-software-rasterizer",
                "--disable-background-networking",
                "--disable-sync",
                "--disable-default-apps",
                "--no-first-run",
            ],
        }
        if system_chrome:
            launch_kwargs["executable_path"] = system_chrome

        try:
            browser = await p.chromium.launch(**launch_kwargs)
        except Exception as exc:
            return {"url": url, "error": f"Chromium launch failed: {exc}"}

        page = await browser.new_page()
        try:
            await page.goto(url, timeout=25_000, wait_until="domcontentloaded")

            result["title"] = await page.title()

            for meta_name in ("description", "og:description", "og:title"):
                val = await page.evaluate(
                    f"""() => {{
                        const el = document.querySelector(
                            'meta[name="{meta_name}"], meta[property="{meta_name}"]');
                        return el ? el.content : null;
                    }}"""
                )
                if val:
                    result[meta_name.replace(":", "_")] = val

            result["headings"] = await page.evaluate(
                """() => Array.from(document.querySelectorAll('h1,h2'))
                              .map(h => h.innerText.trim())
                              .filter(t => t.length > 0)
                              .slice(0, 10)"""
            )
            result["body_text"] = await page.evaluate(
                """() => {
                    const b = document.querySelector('article') || document.body;
                    return b ? b.innerText.slice(0, 2000) : '';
                }"""
            )

            if screenshot_path:
                await page.screenshot(path=screenshot_path, full_page=False)
                result["screenshot_path"] = screenshot_path

        except Exception as exc:
            result["error"] = str(exc)
        finally:
            await browser.close()

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: browser_runner.py <url> [screenshot_path]"}))
        sys.exit(1)

    _url = sys.argv[1]
    _shot = sys.argv[2] if len(sys.argv) > 2 else None

    _result = asyncio.run(_run(_url, _shot))
    print(json.dumps(_result))
