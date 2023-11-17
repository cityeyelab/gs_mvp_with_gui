import asyncio
import websockets
import json
import cv2
import base64
import numpy as np
import time

async def provide(que1, que2, que3, que_bp):
    q1_frame = np.zeros((1080, 1920, 3),dtype=np.uint8)
    q2_frame = np.zeros((1080, 1920, 3),dtype=np.uint8)
    q3_frame = np.zeros((1080, 1920, 3),dtype=np.uint8)
    bp_frame = np.zeros((1258, 979, 3),dtype=np.uint8)
    cnt = 0
    async with websockets.connect("ws://127.0.0.1:8001/ws/frames_innerpass") as websocket:
        while True:
            try:
                if not que1.empty():
                    q1_frame = que1.get()
                if not que2.empty():
                    q2_frame = que2.get()
                if not que3.empty():
                    q3_frame = que3.get()
                if not que_bp.empty():
                    bp_frame = que_bp.get()
                
                # bp_frame = que_bp.get()
                if cnt%5 == 0:
                    _, frame1 = cv2.imencode('.jpg', q1_frame, [cv2.IMWRITE_JPEG_QUALITY,50])
                    _, frame2 = cv2.imencode('.jpg', q2_frame, [cv2.IMWRITE_JPEG_QUALITY,50])
                    _, frame3 = cv2.imencode('.jpg', q3_frame, [cv2.IMWRITE_JPEG_QUALITY,50])
                    _, frame_bp = cv2.imencode('.jpg', bp_frame, [cv2.IMWRITE_JPEG_QUALITY,50])
                    frame1 = base64.b64encode(frame1)
                    # print('frame1 b64 encoded : ', frame1)
                    frame1 = frame1.decode()
                    # print('frame1 decoded : ', frame1)
                    # frame1.decode('utf-8')
                    frame2 = base64.b64encode(frame2)
                    # frame2.decode('utf-8')
                    frame2 = frame2.decode()
                    frame3 = base64.b64encode(frame3)
                    # frame3.decode('utf-8')
                    frame3 = frame3.decode()
                    frame_bp = base64.b64encode(frame_bp)
                    frame_bp = frame_bp.decode()
                    data = {"frame1": frame1, "frame2": frame2, "frame3": frame3, "frame_bp": frame_bp,}
                    # data = {"frame1": q1_frame, "frame2": q2_frame, "frame3": q3_frame,
                    #         "frame_bp" : bp_frame}
                    json_data = json.dumps(data)
                    await websocket.send(json_data)
                    # await asyncio.sleep(0.01)
                    # print('frame provider data sent')
                    cnt = 0
                else:
                    await asyncio.sleep(0.005)
            except Exception as e:
                print(f'something went wrong in ws send on provider, error : {e}')
            await asyncio.sleep(0.02)
            # time.sleep(0.01)
            cnt += 1

        # try:
        #     retval, buffer = cv2.imencode('.jpg', res_show, [cv2.IMWRITE_JPEG_QUALITY,80])
        #     data = base64.b64encode(buffer)
        #     await websocket.send(data)
        #     await websocket.close()
        # except Exception as e:
        #     print(f'something went wrong in ws send, error : {e}')
        #     await websocket.close()

def run_provide(que1, que2, que3, que_bp ):
    asyncio.get_event_loop().run_until_complete(provide(que1, que2, que3, que_bp ))
    asyncio.get_event_loop().run_forever()