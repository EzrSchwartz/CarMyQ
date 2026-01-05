import asyncio
from MyQServer import run_server
from resnetclassification import triggerLoop

async def main():
    await asyncio.gather(
        run_server(),
        triggerLoop()
    )

asyncio.run(main())