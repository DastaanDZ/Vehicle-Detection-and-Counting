import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import torch

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def count_vehicles(bbox1_idx, counter, first_thres_dn, final_thres_dn, check_thres_dn, first_thres_east, final_thres_east, first_thres_west, final_thres_west, vehicle, frame, name):

    for bbox1 in bbox1_idx:
        for i in vehicle:
            x3, y3, x4, y4, id1 = bbox1
            print("X3,Y3,X4,Y4,ID1")
            print(bbox1)
            cxm = int(x3 + x4) // 2
            cym = int(y3 + y4) // 2

            # Checking vehicles in east direction
            if cxm > first_thres_east and cxm < final_thres_east and id1 not in vh_dn_conf and id1 not in vh_west_conf:
                vh_east.add(id1)
            # if id1 exists in vh_east set to check if that object is going east or down
            if id1 in vh_east:
                # to check if that vehicle actually went in the east direction and not returned again
                if cxm > final_thres_east:
                    vh_east_conf.add(id1)

            # Checking vehicles in down direction
            if cym > first_thres_dn and cym < final_thres_dn and id1 not in vh_east_conf and id1 not in vh_west_conf:
                vh_dn.add(id1)

            if id1 in vh_dn:
                if cym > final_thres_dn:
                    vh_dn_conf.add(id1)

                if cym < check_thres_dn:
                    vh_dn_conf.add(id1)

            # Checking vehicles in west direction
            if cxm < first_thres_west and cxm > final_thres_west and id1 not in vh_dn_conf and id1 not in vh_east_conf:
                print("ID",id1,"Crossed the WEST line")
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),4)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                cv2.putText(frame, f'{name}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                vh_west.add(id1)
            # if id1 exist in vh_west set to check if that object is going east or west
            if id1 in vh_west:
                print("ID",id1,"Crossed the WEST line in the past")
                # to check if that vehicle actually went in the west direction and not returned again
                if cxm <  final_thres_west :
                    print("ID",id1,"GOING WEST CONFIRMED")
                    cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
                    if id1 not in counter:
                        counter.append(id1)


# Load the YOLOv8n model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use GPU 0
model = YOLO('yolov8x.pt')
model.to(device)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture("G:\\NIT STUDENT PROJECT\\Monday Clip\\192.168.1.100_ch8_20230807120002_20230807160003.asf")

my_file = open("C:\\Users\\Dastaan DZ\\Documents\\Projects\\yolo\\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

vh_dn = set()
vh_up = set()
vh_east = set()
vh_west = set()

vh_dn_conf = set()
vh_east_conf = set()
vh_west_conf = set()


count=0

passed={}

first_thres_dn=260
final_thres_dn=280
check_thres_dn=247
first_thres_east = 142
final_thres_east = 172
first_thres_west = 872
final_thres_west = 842



tracker=Tracker()

counter=[]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    
    results = model.track(frame, persist=True)
    print('\nResults:')
    print(results[0])

    a=results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    list=[]
    vehicle=[]

    for index,row in px.iterrows():
        print("+++++++PRINTING INDEX,ROW++++++++")
        print(index,row,class_list[int(row[6])])
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        confidence_score=row[5]
        name=class_list[int(row[6])]

        if name == 'car' and confidence_score >= 0.8:
            list.append([x1,y1,x2,y2])
            vehicle.append(name)
    bbox4_idx=tracker.update(list)
    count_vehicles(bbox4_idx,counter,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,vehicle,frame,'car')

    cv2.line(frame, (96, first_thres_dn), (867, first_thres_dn), (0, 0, 255), 2)
    cv2.line(frame, (0, final_thres_dn), (966, final_thres_dn), (0, 0, 255), 2)
    # Threshold for covering vehicle in east direction 
    cv2.line(frame, (142, 254), (142,497), (0, 255,0), 2)
    cv2.line(frame, (172, 254), (172,497), (0, 255,0), 2)
    # Threshold for covering vehicle in west direction
    cv2.line(frame, (872 ,283), (872, 499), (255, 0,0), 2)
    cv2.line(frame, (842 ,283), (842, 499), (255, 0,0), 2)

    car=(len(counter))
    cvzone.putTextRect(frame,f'car:-{car}',(19,120),2,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
