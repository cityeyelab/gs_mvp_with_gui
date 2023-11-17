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
    # run('network_backend.main:app', host='0.0.0.0', port=8001, reload=False)

    # run('network_backend.main:app', host='0.0.0.0', port=8080, reload=False)
    run('network_backend.main:app', host='0.0.0.0', port=8080, reload=False, proxy_headers=True, forwarded_allow_ips='*',)
    
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

        self.total_cnt = 0
        self.congestion = 0
        self.num_of_waiting_cars = 0
        self.waiting_time = 0
        self.electric_charging_waiting_cnt = 0
        self.car_interior_wash_cnt = 0
        self.rt_newdata_come = False

        pre_frame1 =  np.zeros((1080, 1920, 3), dtype=np.uint8)
        pre_frame2 =  np.zeros((1080, 1920, 3), dtype=np.uint8)
        pre_frame3 =  np.zeros((1080, 1920, 3), dtype=np.uint8)
        pre_frame_bp = np.zeros((1258, 979, 3),dtype=np.uint8)

        _, frame1 = cv2.imencode('.jpg', pre_frame1, [cv2.IMWRITE_JPEG_QUALITY,50])
        _, frame2 = cv2.imencode('.jpg', pre_frame2, [cv2.IMWRITE_JPEG_QUALITY,50])
        _, frame3 = cv2.imencode('.jpg', pre_frame3, [cv2.IMWRITE_JPEG_QUALITY,50])   
        _, frame_bp = cv2.imencode('.jpg', pre_frame_bp, [cv2.IMWRITE_JPEG_QUALITY,50])  
        
        # frame1 = pre_frame1
        frame1 = base64.b64encode(frame1)
        frame1 = frame1.decode()
        # frame2 = pre_frame2
        frame2 = base64.b64encode(frame2)
        frame2 = frame2.decode()
        # frame3 = pre_frame3
        frame3 = base64.b64encode(frame3)
        frame3 = frame3.decode()
        frame_bp = base64.b64encode(frame_bp)
        frame_bp = frame_bp.decode()

        self.frame1 = frame1
        self.frame2 = frame2
        self.frame3 = frame3
        self.frame_bp = frame_bp
        
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


mobile_front_main_dict = {"부천옥길":"bcokgil"}
@app.get('/mobile')
def main(request: Request):
    print('mobile main client : ', request.client)
    return mobile_front_main_dict
# def run_socket_server():

# ws://127.0.0.1:8001/ws/frames_innerpass

@app.websocket("/ws/frames_provider")
async def frames_provider(websocket: WebSocket):
    await websocket.accept()
    print('frames provider accepted')

    # _, frame1 = cv2.imencode('.jpg', app.frame1, [cv2.IMWRITE_JPEG_QUALITY,50])
    # _, frame2 = cv2.imencode('.jpg', app.frame2, [cv2.IMWRITE_JPEG_QUALITY,50])
    # _, frame3 = cv2.imencode('.jpg', app.frame3, [cv2.IMWRITE_JPEG_QUALITY,50])

    data_dict = {'frame1': app.frame1, 'frame2': app.frame2, 'frame3': app.frame3, 'frame_bp': app.frame_bp}
    json_data = json.dumps(data_dict)
    await websocket.send_json(json_data)
    await asyncio.sleep(0.2)

    print('frame provider init sent')

    while True:
        try:
            pass
            data_dict =  data_dict = {'frame1': app.frame1, 'frame2': app.frame2, 'frame3': app.frame3, 'frame_bp': app.frame_bp}
            json_data = json.dumps(data_dict)
            await websocket.send_json(json_data)
            await asyncio.sleep(0.1)
        except Exception as e:
            print('something went wrong in WebSockets frame provider, error = ' , e)
            break
        await asyncio.sleep(0.1)


@app.websocket("/ws/frames_innerpass")
async def frame_provider_innerpass(websocket: WebSocket):
    await websocket.accept()
    print('frame provider innerpass start')
    while True:
        try:
            # print('try to receive')
            recieved = await websocket.receive()
            # print('received data : ', recieved)
            if recieved['type'] == 'websocket.receive':
                data = json.loads(recieved['text'])
                frame1 = data['frame1']
                # frame1 = base64.b64decode(frame1)
                # print('frame1 b64 decoded: ', frame1)
                frame2 = data['frame2']
                # frame2 = base64.b64decode(frame2)
                frame3 = data['frame3']
                # frame3 = base64.b64decode(frame3)
                # frame_bp = data['frame_bp']
                frame_bp = data['frame_bp']
                app.frame1 = frame1
                app.frame2 = frame2
                app.frame3 = frame3
                app.frame_bp = frame_bp

                # cv2.imshow('frmae1', frame1)
                # cv2.waitKey(10)
            elif recieved['type'] == 'websocket.disconnect':
                print('received disconnet msg, frame provider')
                break
        except Exception as e:
            print('something went wrong in rt data inner pass, error = ' , e)
            await websocket.close()
            break
        await asyncio.sleep(0.05)

@app.websocket("/ws/rt_data_innerpass")
async def realtime_data_provider(websocket: WebSocket):
    await websocket.accept()
    print('rt data innerpass start')
    while True:
        try:
            # congestion, numberOfWaitingCars, waitingTime
            recieved = await websocket.receive()
            # print('rt data innerpass receivved : ' , recieved)
            # print('rt data innerpass receivved car wash cnt: ' , recieved['car_interior_wash_cnt'])
            if recieved['type'] == 'websocket.receive':
                data = json.loads(recieved['text'])
                # print('rt data innerpass receivved -> json decode: ' , data)
                # recieved = json.dec
                app.total_cnt = data['total_cnt']
                app.congestion = data['congestion']
                app.num_of_waiting_cars = data['number_of_waiting_cars'] 
                app.waiting_time = data['waiting_time']
                app.electric_charging_waiting_cnt =  data['electric_charging_waiting_cnt']
                app.car_interior_wash_cnt =  data['car_interior_wash_cnt']
                # print('electric waiting cnt : ', data['electric_charging_waiting_cnt'])
                # print('interior wash cnt ; ', data['car_interior_wash_cnt'])
                # data_dict = {'image': data, 'frameCnt': frame_cnt}
                # json_data = json.dumps(data_dict)
                # await websocket.send_json(json_data)
                await websocket.close()
            elif recieved['type'] == 'websocket.disconnect':
                print('received disconnet msg, cc')
                # await websocket.close()
                break
        except Exception as e:
            print('something went wrong in rt data inner pass, error = ' , e)
            await websocket.close()
            break



@app.websocket("/mobile/realtime")
async def realtime_data_provider(websocket: WebSocket):
    await websocket.accept()
    print('rt data provider accepted, client : ', websocket.client)
    print('rt data provider accepted, header : ', websocket.headers)
    data_dict = {'total_cnt':app.total_cnt, 'congestion': app.congestion, 'num_of_waiting_cars': app.num_of_waiting_cars, 'waiting_time':app.waiting_time,
                         'electric_charging_waiting_cnt':app.electric_charging_waiting_cnt,  'car_interior_wash_cnt': app.car_interior_wash_cnt}
    json_data = json.dumps(data_dict)
    await websocket.send_json(json_data)
    await asyncio.sleep(0.2)



    while True:
        try:
            data_dict = {'total_cnt':app.total_cnt, 'congestion': app.congestion, 'num_of_waiting_cars': app.num_of_waiting_cars, 'waiting_time':app.waiting_time,
                         'electric_charging_waiting_cnt':app.electric_charging_waiting_cnt,  'car_interior_wash_cnt': app.car_interior_wash_cnt}
            json_data = json.dumps(data_dict)
            await websocket.send_json(json_data)
            await asyncio.sleep(0.1)
        except Exception as e:
            print('something went wrong in WebSockets rt data provider, error = ' , e)
            break

        await asyncio.sleep(2)



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
    # await asyncio.sleep(1)
    # ready_ping = await websocket.receive_text()
    # await websocket.send_text('pong')

    frame_cnt = 0
    frame_collision = app.bp_collision_heatmap
    data = frame_collision.decode('utf-8')
    data_dict = {'image': data, 'frameCnt': frame_cnt}
    json_data = json.dumps(data_dict)
    
    try:
        await websocket.send_json(json_data)
        await asyncio.sleep(0.1)
        # await websocket.send_text('pong')
        
    except Exception as e:
        print('something went wrong in initial col : ', e)
    
    # print('inital sent data col : ', json_data)

    # try:
    #     await websocket.send_json(json_data)
    #     print('/ws col sent data initially1')
    #     await asyncio.sleep(3)
    #     await websocket.send_json(json_data)
    #     print('/ws col sent data initially2')
    # except websockets.exceptions.ConnectionClosedOK:
    #     print('connection closed')
    # except Exception as e:
    #     print('something went wrong in initial col : ', e)

    # data = 0

    try:
        while True:
            
            # if app.once_collision_heatmap_income:
            if app.col_newdata_come:
            # if False:
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
                # print('nothing new in /ws col')
                await websocket.send_text('pong')
                # print('ws state = ' , websocket.state)
                # print('ws_client state = ' , websocket.client_state)
                # print('ws application state = ' , websocket.application_state)
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
        # print('ws state = ' , websocket.state)
        # print('ws_client state = ' , websocket.client_state)
        # print('ws application state = ' , websocket.application_state)
        # websocket.
        await websocket.close()

        # websockets.exceptions.ConnectionClosedOK: received 1005 (no status received [internal]); then sent 1005 (no status received [internal])


@app.websocket("/ws/bp-st")
async def websocket_endpoint_st(websocket: WebSocket):
    # path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8'
    # cap_loader = cv2.VideoCapture(path)
    
    print('ws st endpt start')

    await websocket.accept()
    print(f"client connected (/ws/bp-st) : {websocket.client}")
    # await asyncio.sleep(1)
    # ready_ping = await websocket.receive_text()
    # await websocket.send_text('pong')

    frame_cnt = 0
    frame_st = app.bp_st_heatmap
    data = frame_st.decode('utf-8')
    data_dict = {'image': data, 'frameCnt': frame_cnt}
    json_data = json.dumps(data_dict)
    try:
        await websocket.send_json(json_data)
        await asyncio.sleep(0.1)
        # await websocket.send_text('pong')
    except Exception as e:
        print('something went wrong in initial st : ', e)
    # print('inital sent data st : ', json_data)
    
    # try:
    #     await websocket.send_json(json_data)
    #     print('/ws st sent data initailly1')
    #     await asyncio.sleep(3)
    #     await websocket.send_json(json_data)
    #     print('/ws st sent data initailly2')
    # except websockets.exceptions.ConnectionClosedOK:
    #     print('connection closed')
    # except Exception as e:
    #     print('something went wrong in initial col : ', e)

    # data = 0
    
    try:
        while True:
            
            # if app.once_st_heatmap_income:
            if app.st_newdata_come:
            # if False:
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
                # print('nothing new in /ws st')
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