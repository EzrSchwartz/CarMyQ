import asyncio
from MyQServer import run_server
from Segment import StreamSegment

async def main():
    await asyncio.gather(
        StreamSegment()
        # ,run_server()
    )

asyncio.run(main())
