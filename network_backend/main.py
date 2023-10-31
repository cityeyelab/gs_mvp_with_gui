from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import cv2
import base64
import json
import time
import numpy as np
from threading import Thread
import asyncio
import websockets


def create_network_backend():
    run('network_backend.main:app', host='0.0.0.0', port=8001, reload=False)
    # uvicorn.run('app:app', ws_ping_interval=300, ws_ping_timeout=300)
    # run('network_backend.main:app', host='0.0.0.0', port=8001, reload=False, ws_ping_interval=5, ws_ping_timeout=60*30)


class CustomAPI(FastAPI):
    def __init__(self):
        # print('CustomAPI Init!!!!!!')
        # self.bp_heatmap_que = 
        self.bp_collision_heatmap = np.zeros((320, 320, 3), dtype=np.uint8)
        self.bp_st_heatmap = np.zeros((320, 320, 3), dtype=np.uint8)
        # self.bp_heatmap_consumer_thread = Thread(target=self.bp_heatmap_consumer)
        # run_socket_server_thread = Thread(target=self.run_socket_server)
        # run_socket_server_thread.start()
        self.once_collision_heatmap_income = False
        self.once_st_heatmap_income = False
        
        self.col_newdata_come = False
        self.st_newdata_come = False
        
        super().__init__()
    
    # def run_uvicorn(self, bp_heatmap_que):
    #     # self.bp_heatmap_que = bp_heatmap_que
    #     # self.bp_heatmap_consumer_thread.start()
    #     run('network_backend.main:app', host='0.0.0.0', port=8001, reload=False)

    # def bp_heatmap_consumer(self):
    #     while True:
    #         self.bp_collision_heatmap = self.bp_heatmap_que.get()
    #         time.sleep(0.01)
            
    
    # async def accept(self, websocket, path):
    #     # cnt = 0
    #     while True:
    #         # cnt += 1

    #         # await websocket.send('you are connected... ' + str(cnt))
    #         rcv = await websocket.recv()
    #         print('server received: ', rcv)
    #         # await asyncio.sleep(0.5)
            
    # def run_socket_server(self):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     start_server = websockets.serve(self.accept, "localhost", 9000,)
    #     # 비동기로 서버를 대기한다.
    #     asyncio.get_event_loop().run_until_complete(start_server)
    #     asyncio.get_event_loop().run_forever()

# app = FastAPI()
app = CustomAPI()

# origins = [
#     "http://http://14.36.1.6:8000/",
#     "http://localhost",
#     "http://localhost:8000",
#     "http://0.0.0.0:8000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get('/')
def main(request: Request):
    print('client : ', request.client)
    return 'hello'

# @app.get('/run_ws')
# def run_socket_server():
    

@app.websocket("/ws/bp_collision_heatmap_innerpass")
async def heatmap_receive_col(websocket: WebSocket):
    
    await websocket.accept()
    # await websocket.send_text('connected!')
    print(f"client connected collision: {websocket.client}")
    # data = 0
    # frame_cnt = 0
    while True:
        try:
            recieved = await websocket.receive()
            print('something collision recieved!')
            if recieved['type'] == 'websocket.receive':
                print('bp collision byte received')
                recieved_bytes = recieved['bytes']
                # print('recieved_bytes = ' , recieved_bytes)
                # decoded = base64.b64decode(recieved_bytes)
                app.bp_collision_heatmap = recieved_bytes
                # app.once_collision_heatmap_income = True
                app.col_newdata_come = True
                # decoded = base64.b64decode(decoded)
                # result_img = cv2.imdecode(np.frombuffer(decoded, np.uint8), -1)
                # print('decoded = ' , decoded)
                # print('decoded = ' , result)
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # cv2.imshow('frame', decoded)
                # cv2.imshow('frame', result_img)
                # cv2.waitKey(0)
            elif recieved['type'] == 'websocket.disconnect':
                print('ws collision server disconnected')
                # await websocket.close()
                break
            # await websocket.send_text('received!')
            # app.bp_heatmap = recieved
            # print('received data = ' , recieved)
            
        except WebSocketDisconnect:
            print('ws collision heatmap disconnected!')
            # await websocket.close()
            break
        except Exception as e:
            print('something went wrong in WebSockets collision heatmap, error = ' , e)
            # await websocket.close()
            break
        # time.sleep(0.5)
        print('server ws collision while')


@app.websocket("/ws/bp_st_heatmap_innerpass")
async def heatmap_receive_st(websocket: WebSocket):
    
    await websocket.accept()
    # await websocket.send_text('connected!')
    print(f"client connected st : {websocket.client}")
    # data = 0
    # frame_cnt = 0
    while True:
        try:
            recieved = await websocket.receive()
            print('something st recieved!')
            if recieved['type'] == 'websocket.receive':
                print('bp st byte received')
                recieved_bytes = recieved['bytes']
                # print('recieved_bytes = ' , recieved_bytes)
                # decoded = base64.b64decode(recieved_bytes)
                app.bp_st_heatmap = recieved_bytes
                # app.once_st_heatmap_income = True
                app.st_newdata_come = True
                # decoded = base64.b64decode(decoded)
                # result_img = cv2.imdecode(np.frombuffer(decoded, np.uint8), -1)
                # print('decoded = ' , decoded)
                # print('decoded = ' , result)
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # cv2.imshow('frame', decoded)
                # cv2.imshow('frame', result_img)
                # cv2.waitKey(0)
            elif recieved['type'] == 'websocket.disconnect':
                print('ws st server disconnected')
                # await websocket.close()
                break
            # await websocket.send_text('received!')
            # app.bp_heatmap = recieved
            # print('received data = ' , recieved)
            
        except WebSocketDisconnect:
            print('ws st heatmap disconnected!')
            # await websocket.close()
            break
        except Exception as e:
            print('something went wrong in WebSockets st heatmap, error = ' , e)
            # await websocket.close()
            break
        # time.sleep(0.5)
        print('server ws st while')


@app.websocket("/ws/bp-col")
async def websocket_endpoint_col(websocket: WebSocket):
    # path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8'
    # cap_loader = cv2.VideoCapture(path)
    
    
    await websocket.accept()
    print(f"client connected (/ws/col) : {websocket.client}")
    # data = 0
    frame_cnt = 0
    try:
        while True:
            
            # if app.once_collision_heatmap_income:
            if app.col_newdata_come:
                app.col_newdata_come = False
                # print('check1 /ws')
                frame_collision = app.bp_collision_heatmap
                # print('check2 /ws')
                # frame = cv2.resize(frame, (320, 320))
                # retval, buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
                # data = base64.b64encode(buffer)
                # print('frame /ws = ' , frame)
                data = frame_collision.decode('utf-8')
                # print('check3 /ws')
                data_dict = {'image': data, 'frameCnt': frame_cnt}
                # print('check4 /ws')
                json_data = json.dumps(data_dict)
                # print('check5 /ws')
                await websocket.send_json(json_data)
                
                print('/ws col sent data')
                
                frame_cnt += 1
            else:
                print('nothing new in /ws col')
                await websocket.send_text('pong')
                # pass
            
            await asyncio.sleep(1)
            # ping = await websocket.receive_text()
            # print('received ping = ' , ping)
            
            # if ping == 'ping':
            #     await websocket.send_text('pong')
            
            # print('/ws: working')
            # time.sleep(2)
            await asyncio.sleep(1)
            # websocket.
            # print('ws state = ' , websocket.state)
            # print('ws_client state = ' , websocket.client_state)
            # print('ws application state = ' , websocket.application_state)
    except WebSocketDisconnect:
        print('ws disconnected!')
        await websocket.close()
    except Exception as e:
        print('something went wrong in WebSockets col, error : ', e)
        await websocket.close()


@app.websocket("/ws/bp-st")
async def websocket_endpoint_st(websocket: WebSocket):
    # path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8'
    # cap_loader = cv2.VideoCapture(path)
    
    print('ws st endpt start')
    
    await websocket.accept()
    print(f"client connected (/ws/bp-st) : {websocket.client}")
    # data = 0
    frame_cnt = 0
    try:
        while True:
            
            # if app.once_st_heatmap_income:
            if app.st_newdata_come:
                app.st_newdata_come = False
                # print('check1 /ws')
                frame_st = app.bp_st_heatmap
                # print('check2 /ws')
                # frame = cv2.resize(frame, (320, 320))
                # retval, buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
                # data = base64.b64encode(buffer)
                # print('frame /ws = ' , frame)
                data = frame_st.decode('utf-8')
                # print('check3 /ws')
                data_dict = {'image': data, 'frameCnt': frame_cnt}
                # print('check4 /ws')
                json_data = json.dumps(data_dict)
                # print('check5 /ws')
                await websocket.send_json(json_data)
                print('/ws st sent data')
                frame_cnt += 1
            else:
                print('nothing new in /ws st')
                await websocket.send_text('pong')
            
            await asyncio.sleep(1)
            
            # ping = await websocket.receive_text()
            # print('received st ping = ' , ping)
            
            # if ping == 'ping':
            #     await websocket.send_text('pong')
            
            # print('/ws: working')
            # time.sleep(2)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print('ws st disconnected!')
        await websocket.close()
    except Exception as e:
        print('something went wrong in WebSockets st, error : ', e)
        await websocket.close()


# class NetworkBackend():
#     def __init__(self) -> None:
#         self.app = FastAPI()
    
#     # @app.get('/')
#     def hello(self):
#         @self.app.get('/')
#         def inner_hello():
#             return 'hi'


# class NetworkBackend(FastAPI):
#     def __init__(self):
#         super().__init__()
    
#     # @self.get('/')
#     # def hello(self):
#     #     return 'hello'
    
    
#     def hello2(self):
#         # result = ''
#         @self.get('/')
#         def inner_hello():
#             # result = 'hello'
#             return 'hello'
#         inner_hello()


if __name__ == '__main__':
    pass