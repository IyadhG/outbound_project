"""
Apify API enricher — wraps website-content-crawler and google-search-scraper actors.
Requirements: 1.2, 2.2
"""

import asyncio
import logging

import httpx
import os


logger = logging.getLogger(__name__)


class ApifyEnricher:
    APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
    BASE_URL = "https://api.apify.com/v2"

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.APIFY_API_KEY}",
            "Content-Type": "application/json",
        }

    async def _start_run(self, client: httpx.AsyncClient, actor: str, body: dict) -> str | None:
        """Start an Apify actor run and return the run_id, or None on failure."""
        url = f"{self.BASE_URL}/acts/{actor}/runs"
        try:
            response = await client.post(url, json=body, headers=self._auth_headers())
            response.raise_for_status()
            data = response.json()
            return data["data"]["id"]
        except Exception as exc:
            logger.warning("Failed to start Apify actor %s: %s", actor, exc)
            return None

    async def _poll_run(
        self,
        client: httpx.AsyncClient,
        run_id: str,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
    ) -> str | None:
        """Poll until run status is SUCCEEDED or FAILED. Returns dataset_id or None."""
        url = f"{self.BASE_URL}/actor-runs/{run_id}"
        elapsed = 0.0
        while elapsed < timeout:
            try:
                response = await client.get(url, headers=self._auth_headers())
                response.raise_for_status()
                data = response.json()["data"]
                status = data.get("status")
                if status == "SUCCEEDED":
                    return data.get("defaultDatasetId")
                if status == "FAILED":
                    logger.warning("Apify run %s failed.", run_id)
                    return None
            except Exception as exc:
                logger.warning("Error polling Apify run %s: %s", run_id, exc)
                return None
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning("Apify run %s timed out after %ss.", run_id, timeout)
        return None

    async def _fetch_dataset_items(
        self, client: httpx.AsyncClient, dataset_id: str
    ) -> list:
        """Fetch items from an Apify dataset."""
        url = f"{self.BASE_URL}/datasets/{dataset_id}/items"
        try:
            response = await client.get(url, headers=self._auth_headers())
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("Failed to fetch dataset %s: %s", dataset_id, exc)
            return []

    async def crawl_website(self, domain: str, timeout: float = 60.0) -> dict:
        """
        Run the apify~website-content-crawler actor for the given domain.
        Returns the first dataset item as a dict, or {} on failure/timeout.
        """
        actor = "apify~website-content-crawler"
        body = {
            "startUrls": [{"url": f"https://{domain}"}],
            "maxCrawlPages": 3,
        }
        try:
            async with httpx.AsyncClient() as client:
                run_id = await self._start_run(client, actor, body)
                if run_id is None:
                    return {}
                dataset_id = await self._poll_run(client, run_id, timeout=timeout)
                if dataset_id is None:
                    return {}
                items = await self._fetch_dataset_items(client, dataset_id)
                return items[0] if items else {}
        except Exception as exc:
            logger.warning("crawl_website(%s) unexpected error: %s", domain, exc)
            return {}

    async def search_news(self, company_name: str, timeout: float = 60.0) -> list:
        """
        Run the apify~google-search-scraper actor for recent news about company_name.
        Returns a list of result dicts, or [] on failure/timeout.
        """
        actor = "apify~google-search-scraper"
        body = {
            "queries": f"{company_name} news",
            "maxPagesPerQuery": 1,
        }
        try:
            async with httpx.AsyncClient() as client:
                run_id = await self._start_run(client, actor, body)
                if run_id is None:
                    return []
                dataset_id = await self._poll_run(client, run_id, timeout=timeout)
                if dataset_id is None:
                    return []
                items = await self._fetch_dataset_items(client, dataset_id)
                return items if isinstance(items, list) else []
        except Exception as exc:
            logger.warning("search_news(%s) unexpected error: %s", company_name, exc)
            return []
