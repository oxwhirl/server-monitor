import asyncio


class StatFetcher:
    """Base class for fetching some statistic from a host"""

    def __init__(self, sem: asyncio.Semaphore = None):
        self.sem = sem

    async def fetch(self, hostname):
        if self.sem is not None:
            async with self.sem:
                data = await self.fetch_data(hostname)
        else:
            data = await self.fetch_data(hostname)
        return data
