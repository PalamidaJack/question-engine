"""Browser automation tool -- navigate, extract, screenshot, interact."""

from __future__ import annotations

import logging
import re
from typing import Any

from qe.runtime.tools import ToolSpec

log = logging.getLogger(__name__)

browser_spec = ToolSpec(
    name="browser_navigate",
    description=(
        "Navigate to a URL and perform browser actions: extract page content, "
        "take a screenshot, click an element, or fill a form."
    ),
    requires_capability="browser_control",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to"},
            "action": {
                "type": "string",
                "enum": ["extract", "screenshot", "click", "fill_form"],
                "description": "Action to perform on the page",
                "default": "extract",
            },
            "selectors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "CSS selectors -- for 'click' the first is clicked; "
                    "for 'fill_form' pairs of (selector, value) are expected; "
                    "for 'extract' limits content to these selectors"
                ),
            },
        },
        "required": ["url"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "action": {"type": "string"},
            "content": {"type": "string"},
            "success": {"type": "boolean"},
            "error": {"type": "string"},
        },
    },
    timeout_seconds=60,
    category="browser",
)


async def browser_navigate(
    url: str,
    action: str = "extract",
    selectors: list[str] | None = None,
) -> dict[str, Any]:
    """Navigate to *url* and perform *action*.

    Actions
    -------
    extract
        Return the main text content of the page.
    screenshot
        Take a screenshot and return the base64-encoded PNG.
    click
        Click the first matching selector.
    fill_form
        Fill form fields; *selectors* should be pairs of
        ``[selector, value, selector, value, ...]``.
    """
    try:
        return await _playwright_action(url, action, selectors)
    except ImportError:
        log.debug("browser.playwright_unavailable, trying fallback")

    # Fallback for extract action only
    if action == "extract":
        return await _httpx_extract(url, selectors)

    return {
        "url": url,
        "action": action,
        "content": "",
        "success": False,
        "error": (
            "playwright is not installed and only the 'extract' action "
            "is available via fallback. Install with: pip install playwright "
            "&& playwright install"
        ),
    }


# ------------------------------------------------------------------
# Playwright backend
# ------------------------------------------------------------------


async def _playwright_action(
    url: str,
    action: str,
    selectors: list[str] | None,
) -> dict[str, Any]:
    """Perform the browser action via Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            if action == "extract":
                content = await _pw_extract(page, selectors)
                return {
                    "url": url,
                    "action": action,
                    "content": content,
                    "success": True,
                    "error": "",
                }

            if action == "screenshot":
                screenshot_bytes = await page.screenshot(full_page=True)
                import base64

                b64 = base64.b64encode(screenshot_bytes).decode()
                return {
                    "url": url,
                    "action": action,
                    "content": b64,
                    "success": True,
                    "error": "",
                }

            if action == "click":
                if not selectors:
                    return {
                        "url": url,
                        "action": action,
                        "content": "",
                        "success": False,
                        "error": "No selector provided for click action",
                    }
                await page.click(selectors[0], timeout=10_000)
                # Wait for navigation / dynamic content
                await page.wait_for_load_state("domcontentloaded")
                content = await _pw_extract(page, None)
                return {
                    "url": url,
                    "action": action,
                    "content": content,
                    "success": True,
                    "error": "",
                }

            if action == "fill_form":
                if not selectors or len(selectors) < 2:
                    return {
                        "url": url,
                        "action": action,
                        "content": "",
                        "success": False,
                        "error": "fill_form requires selector/value pairs",
                    }
                for i in range(0, len(selectors) - 1, 2):
                    sel = selectors[i]
                    val = selectors[i + 1]
                    await page.fill(sel, val, timeout=10_000)
                return {
                    "url": url,
                    "action": action,
                    "content": "Form filled successfully",
                    "success": True,
                    "error": "",
                }

            return {
                "url": url,
                "action": action,
                "content": "",
                "success": False,
                "error": f"Unknown action: {action}",
            }

        except Exception as exc:
            log.exception("browser.playwright_error url=%s action=%s", url, action)
            return {
                "url": url,
                "action": action,
                "content": "",
                "success": False,
                "error": str(exc),
            }
        finally:
            await browser.close()


async def _pw_extract(page: Any, selectors: list[str] | None) -> str:
    """Extract text content from a Playwright page."""
    if selectors:
        parts: list[str] = []
        for sel in selectors:
            elements = await page.query_selector_all(sel)
            for el in elements:
                text = await el.inner_text()
                if text:
                    parts.append(text.strip())
        return "\n\n".join(parts)

    return await page.inner_text("body")


# ------------------------------------------------------------------
# httpx + trafilatura fallback
# ------------------------------------------------------------------


async def _httpx_extract(
    url: str, selectors: list[str] | None
) -> dict[str, Any]:
    """Fallback extraction using httpx + trafilatura."""
    try:
        import httpx
    except ImportError:
        return {
            "url": url,
            "action": "extract",
            "content": "",
            "success": False,
            "error": "httpx is not installed",
        }

    try:
        async with httpx.AsyncClient(
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QE/1.0)"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        text = _trafilatura_extract(html)
        if not text:
            text = _simple_extract(html)

        return {
            "url": str(resp.url),
            "action": "extract",
            "content": text,
            "success": True,
            "error": "",
        }
    except Exception as exc:
        log.exception("browser.httpx_extract_error url=%s", url)
        return {
            "url": url,
            "action": "extract",
            "content": "",
            "success": False,
            "error": str(exc),
        }


def _trafilatura_extract(html: str) -> str:
    """Try trafilatura for high-quality content extraction."""
    try:
        import trafilatura

        result = trafilatura.extract(
            html, include_comments=False, include_tables=True
        )
        return result or ""
    except ImportError:
        return ""


def _simple_extract(html: str) -> str:
    """Minimal HTML-to-text without external dependencies."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return text
