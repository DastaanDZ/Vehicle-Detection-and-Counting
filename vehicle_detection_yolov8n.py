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

<<<<<<< HEAD
def count_vehicles(bbox1_idx,counter,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,vehicle,frame,name):
=======
def count_vehicles(bbox1_idx,counter,name,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,vehicle):
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c
    # print("Vehicle: ", vehicle)
    # print("I m in count_vehicles")
    # print("bbox1_idx: ", bbox1_idx)
    # iterating through the list of bounding boxes of motorcycle
    for bbox1 in bbox1_idx:
        # print("I m in for loop")
        for i in vehicle:
            # print("I m in second for loop")
            # print("motorcycle: ", vehicle)
            x3,y3,x4,y4,id1=bbox1
            print("X3,Y3,X4,Y4,ID1")
            print(bbox1)
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2

<<<<<<< HEAD
                

=======
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c
            # Checking vehicles in down direction
            if cym > first_thres_dn and cxm > 96 and cxm < 867 and cym < final_thres_dn and id1 not in vh_dn and id1 not in vh_east and id1 not in vh_west:
                print("ID",id1,"Crossed the DOWN line")
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),4)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                cv2.putText(frame, f'{name}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                vh_dn.add(id1)
            # if id1 exist in vh_dn set to check if that object is going down or up
            if id1 in vh_dn:
                print("ID",id1,"Crossed the DOWN line in the past")
                # to check if that vehicle actually went in the down direction and not returned again
                if cym > final_thres_dn :
                    print("ID",id1,"GOING DOWN CONFIRMED")
                    cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
                    if id1 not in counter:
                        counter.append(id1)
                # Checking vehicles if it again went in up direction
<<<<<<< HEAD
                # elif cym < check_thres_dn:
                #     print("ID",id1,"DOWN GOING vehicle UP CONFIRMED")
                #     cvzone.putTextRect(frame,"DOWN GOING vehicle UP CONFIRMED",(250,240),2,1)
                #     cv2.putText(frame, f'{name}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                #     cv2.circle(frame,(cxm,cym),10,(0,255,0),-1)
                #     if id1 not in counter:
                #         counter.append(id1)
                # else:
                #     vh_dn.pop()

            # # Checking vehicles in east direction
            # if cxm > first_thres_east and cxm < final_thres_east and id1 not in vh_up and id1 not in vh_dn and id1 not in vh_west:
            #     print("ID",id1,"Crossed the EAST line")
            #     cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),4)
            #     cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            #     cv2.putText(frame, f'{name}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            #     vh_east.add(id1)
            # # if id1 exist in vh_east set to check if that object is going east or west
            # if id1 in vh_east:
            #     print("ID",id1,"Crossed the EAST line in the past")
            #     # to check if that vehicle actually went in the east direction and not returned again
            #     if cxm > final_thres_east :
            #         print("ID",id1,"GOING EAST CONFIRMED")
            #         cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
            #         if id1 not in counter:
            #             counter.append(id1)
            #         # vh_east.pop()
            
            # # Checking vehicles in west direction
            # if cxm < first_thres_west and cxm > final_thres_west and id1 not in vh_up and id1 not in vh_dn and id1 not in vh_east:
            #     print("ID",id1,"Crossed the WEST line")
            #     cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),4)
            #     cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            #     cv2.putText(frame, f'{name}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            #     vh_west.add(id1)
            # # if id1 exist in vh_west set to check if that object is going east or west
            # if id1 in vh_west:
            #     print("ID",id1,"Crossed the WEST line in the past")
            #     # to check if that vehicle actually went in the west direction and not returned again
            #     if cxm <  final_thres_west :
            #         print("ID",id1,"GOING WEST CONFIRMED")
            #         cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
            #         if id1 not in counter:
            #             counter.append(id1)
            #     # else:
            #     #     vh_west.pop()
=======
                if cym < check_thres_dn:
                    print("ID",id1,"DOWN GOING vehicle UP CONFIRMED")
                    cvzone.putTextRect(frame,"DOWN GOING vehicle UP CONFIRMED",(250,240),2,1)
                    cv2.circle(frame,(cxm,cym),10,(0,255,0),-1)
                    if id1 not in counter:
                        counter.append(id1)

            # Checking vehicles in east direction
            if cxm > first_thres_east and cxm < final_thres_east and id1 not in vh_up and id1 not in vh_dn and id1 not in vh_west:
                print("ID",id1,"Crossed the EAST line")
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),4)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                vh_east.add(id1)
            # if id1 exist in vh_east set to check if that object is going east or west
            if id1 in vh_east:
                print("ID",id1,"Crossed the EAST line in the past")
                # to check if that vehicle actually went in the east direction and not returned again
                if cxm > final_thres_east :
                    print("ID",id1,"GOING EAST CONFIRMED")
                    cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
                    if id1 not in counter:
                        counter.append(id1)
            
            # Checking vehicles in west direction
            if cxm < first_thres_west and cxm > final_thres_west and id1 not in vh_up and id1 not in vh_dn and id1 not in vh_east:
                print("ID",id1,"Crossed the WEST line")
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),4)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
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
    return counter
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c

# Load the YOLOv8n model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use GPU 0
model = YOLO('yolov8n.pt')
model.to(device)


# creating wndiow and reading videos   
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
<<<<<<< HEAD
cap=cv2.VideoCapture("G:\\NIT STUDENT PROJECT\\Monday Clip\\192.168.1.100_ch8_20230807120002_20230807160003.asf")
=======
cap=cv2.VideoCapture("F:\\NIT STUDENT PROJECT\\Monday Clip\\192.168.1.100_ch8_20230807120002_20230807160003.asf")
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c

# reading coco file
my_file = open("C:\\Users\\Dastaan DZ\\Documents\\Projects\\yolo\\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count_motorcycle=0
count_car=0
count_truck=0
count_bus=0

passed_motorcycle={}
passed_car={}
passed_truck={}
passed_bus={}

# initializing thres hold pixel for direction detection
first_thres_dn=260
final_thres_dn=300
check_thres_dn=247
first_thres_east = 142
final_thres_east = 172
first_thres_west = 872
final_thres_west = 842

# initializing set to store the id of vehicles in different direction
vh_dn = set()
vh_up = set()
vh_east = set()
vh_west = set()

# initializing tracker object for different vehicles
tracker_motorcycle=Tracker()
tracker_car=Tracker()
tracker_truck=Tracker()
tracker_bus=Tracker()


# counter to count the number of vehicles of particular type
counter1=[]
counter2=[]
counter3=[]
counter4=[]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)
    print('\nResults:')
    print(results[0])  # Print results

    a=results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    # making list for different vehicles ids
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    motorcycle=[]
    car=[]
    truck=[]
    bus=[]

    for index,row in px.iterrows():
<<<<<<< HEAD
        print("+++++++PRINTING INDEX,ROW++++++++")
        print(index,row,class_list[int(row[6])])
=======
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c
        # extracting out the coordinates of the bounding box
        # print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        confidence_score=row[5]
        name=class_list[int(row[6])]
        # print('\n')
        # print("NAME:  ", name)
        # checking motorcycle/ car/ truck/ bus with confidence score > 0.7
<<<<<<< HEAD
        # if name == 'motorcycle'and confidence_score > 0.4:
        #     list1.append([x1,y1,x2,y2])
        #     motorcycle.append(name)
        # if name == 'car' and confidence_score > 0.6:
        #     list2.append([x1,y1,x2,y2])
        #     car.append(name)
        # if name == 'truck' and confidence_score > 0.6:
        #     list3.append([x1,y1,x2,y2])
        #     truck.append(name)
        if name == 'bus' and confidence_score > 0.8:
            list4.append([x1,y1,x2,y2])
            bus.append(name)
    # getting custom ids for different vehicles
    # bbox1_idx=tracker_motorcycle.update(list1)
    # bbox2_idx=tracker_car.update(list2)
    # bbox3_idx=tracker_truck.update(list3)
    bbox4_idx=tracker_bus.update(list4)

    # getting the count of different vehicles
    count_vehicles(bbox1_idx,counter1,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,motorcycle,frame,'motorcycle')
    count_vehicles(bbox2_idx,counter2,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,car,frame,'car')
    # print("Counter2: ", counter2)
    count_vehicles(bbox3_idx,counter3,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,truck,frame,'truck')
    count_vehicles(bbox4_idx,counter4,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,bus,frame,'bus')
=======
        if name == 'motorcycle'and confidence_score > 0.5:
            list1.append([x1,y1,x2,y2])
            motorcycle.append(name)
        if name == 'car' and confidence_score > 0.5:
            list2.append([x1,y1,x2,y2])
            car.append(name)
        if name == 'truck' and confidence_score > 0.5:
            list3.append([x1,y1,x2,y2])
            truck.append(name)
        if name == 'bus' and confidence_score > 0.7:
            list4.append([x1,y1,x2,y2])
            bus.append(name)
    # getting custom ids for different vehicles
    bbox1_idx=tracker_motorcycle.update(list1)
    bbox2_idx=tracker_car.update(list2)
    bbox3_idx=tracker_truck.update(list3)
    bbox4_idx=tracker_bus.update(list4)

    # getting the count of different vehicles
    # counter1 = count_vehicles(bbox1_idx,counter1,name,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,motorcycle)
    # counter2 = count_vehicles(bbox2_idx,counter2,name,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,car)
    # print("Counter2: ", counter2)
    counter3 = count_vehicles(bbox3_idx,counter3,name,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,truck)
    counter4 = count_vehicles(bbox4_idx,counter4,name,first_thres_dn,final_thres_dn,check_thres_dn,first_thres_east,final_thres_east,first_thres_west,final_thres_west,bus)
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c

    # Threshold for covering the vehicle in down direction
    cv2.line(frame, (96, 260), (867, 290), (0, 0, 255), 2)
    cv2.line(frame, (0, 300), (966, 320), (0, 0, 255), 2)
    cv2.line(frame, (267, 247), (505, 257), (0,0,255), 2)
    # Threshold for covering vehicle in east direction 
    cv2.line(frame, (142, 254), (142,497), (0, 255,0), 2)
    cv2.line(frame, (172, 254), (172,497), (0, 255,0), 2)
    # Threshold for covering vehicle in west direction
    cv2.line(frame, (872 ,283), (872, 499), (255, 0,0), 2)
    cv2.line(frame, (842 ,283), (842, 499), (255, 0,0), 2)

    # getting the count of different vehicles
    motorcyclec=(len(counter1))
    carc=(len(counter2))
    truckc=(len(counter3))
    busc=(len(counter4))
    # showing values on screen
    # cvzone.putTextRect(frame,f'motorcyclec:-{motorcyclec}',(19,30),2,1)
    # cvzone.putTextRect(frame,f'carc:-{carc}',(19,60),2,1)
<<<<<<< HEAD
    # cvzone.putTextRect(frame,f'truckc:-{truckc}',(19,90),2,1)
=======
    cvzone.putTextRect(frame,f'truckc:-{truckc}',(19,90),2,1)
>>>>>>> 01711d2af4d3336a4e0578df60d71bca03772c5c
    cvzone.putTextRect(frame,f'busc:-{busc}',(19,120),2,1)

    # cvzone.putTextRect(frame,f'motorcyclec:-{count_motorcycle}',(19,30),2,1)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
