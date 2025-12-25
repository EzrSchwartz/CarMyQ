import asyncio
from aiohttp import ClientSession
import pymyq

async def control_myq_device():
    async with ClientSession() as websession:
        myq = await pymyq.login('adam.m.schwartz@gmail.com', 'Myq!2358', websession)
        devices = await myq.get_devices()
        print("MyQ Devices:")
        print(devices)
        for device in devices:
            print(device)



control_myq_device_loop = asyncio.get_event_loop()
control_myq_device_loop.run_until_complete(control_myq_device())
