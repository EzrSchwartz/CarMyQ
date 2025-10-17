#pip install open-gopro opencv-python
#pip install goprocam opencv-python



import asyncio
from open_gopro import WirelessGoPro
from open_gopro.util import StreamHelper

async def stream_to_computer():
    gopro = WirelessGoPro()
    await gopro.open()

    stream_url = "udp://127.0.0.1:10000"
    await gopro.media.set_preview_stream(stream_url)

    # You can now use OpenCV to capture the stream from stream_url
    # For example:
    # cap = cv2.VideoCapture("udp://127.0.0.1:10000")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imshow('GoPro Stream', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

asyncio.run(stream_to_computer())
# Note: This code assumes you have the open_gopro library installed and a compatible GoPro camera.

