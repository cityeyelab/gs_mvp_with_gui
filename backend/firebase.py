import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import firestore


############ !!!!!!!!! Actually, This should be in safe place on actual product !!!!!!!!!! #############
cred = credentials.Certificate('backend/gs-mvp-473aa-firebase-adminsdk-hp2hr-577f04e7ed.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://gs-mvp-473aa-default-rtdb.asia-southeast1.firebasedatabase.app/',
})


# REF_DB = db.reference("bcokgil/")
REF_DB = db.reference("bcokgilNew/")

def firebase_upload(data):

    # data = {"congestion" : {'value': congestion, 'time': current_time},
    #         "numberOfWaitingCars" : {'value' : int(len(aoi_all_data_cls_lst)), "time" : current_time},
    #         "realtimeWaitingTime" : {'value' : int(result_waiting_time), 'time' : current_time}}

    # data = {'congestion':congestion, 'number_of_waiting_cars': number_of_waiting_cars, 'waiting_time':waiting_time}
    # data = {"test" : 0}

    # REF_DB.push(data)
    REF_DB.update(data)

if __name__ == '__main__':
    firebase_upload()