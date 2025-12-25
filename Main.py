from simplefunctions import triggerDoor, readJson, writeJson
from MyQServer import run_server
from datacollectionNetwork import stream_and_segment_livestream
import asyncio
import time

asyncio.run(stream_and_segment_livestream())

asyncio.run(run_server())