"""
Intent signal collector — gathers recent news, job postings, and technology changes.
Requirements: 1.3, 2.3
"""

import logging

import httpx

from apify_enricher import ApifyEnricher

logger = logging.getLogger(__name__)


class IntentCollector:
    def __init__(self, enricher: ApifyEnricher):
        self.enricher = enricher

    async def collect(self, domain: str, company_name: str) -> dict:
        """
        Collect intent signals for a company.

        Returns:
            {
                "recent_news": [...],
                "job_postings_count": int,
                "technology_changes": [...],
            }
        Each step is independently try/excepted — one failure does not affect others.
        If all calls fail, returns the default empty structure so the pipeline never crashes.
        """
        recent_news = []
        job_postings_count = 0
        technology_changes = []

        # Step 1: Recent news via Google Search scraper
        try:
            recent_news = await self.enricher.search_news(company_name)
            if not isinstance(recent_news, list):
                recent_news = []
        except Exception as exc:
            logger.warning("IntentCollector: search_news failed for %s: %s", company_name, exc)
            recent_news = []

        # Step 2: Job postings count via LinkedIn jobs scraper
        try:
            actor = "apify~linkedin-jobs-scraper"
            body = {
                "queries": [{"query": company_name, "location": ""}],
                "maxResults": 50,
            }
            async with httpx.AsyncClient() as client:
                run_id = await self.enricher._start_run(client, actor, body)
                if run_id is not None:
                    dataset_id = await self.enricher._poll_run(client, run_id)
                    if dataset_id is not None:
                        items = await self.enricher._fetch_dataset_items(client, dataset_id)
                        job_postings_count = len(items) if isinstance(items, list) else 0
        except Exception as exc:
            logger.warning(
                "IntentCollector: LinkedIn jobs scraper failed for %s: %s", company_name, exc
            )
            job_postings_count = 0

        # Step 3: Technology changes from website crawler
        try:
            crawl_result = await self.enricher.crawl_website(domain)
            techs = crawl_result.get("technologies", [])
            if isinstance(techs, list):
                technology_changes = [t for t in techs if isinstance(t, str)]
            else:
                technology_changes = []
        except Exception as exc:
            logger.warning(
                "IntentCollector: crawl_website failed for %s: %s", domain, exc
            )
            technology_changes = []

        return {
            "recent_news": recent_news,
            "job_postings_count": job_postings_count,
            "technology_changes": technology_changes,
        }
