from array import *
from ctypes import *
import os
import cv2
import darknet
import glob
import time
import tkinter as tk
import numpy as np
import time


def convertBack(x, y, w, h):  # Convert from center coordinates to bounding box coordinates
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def drawBox(img, point1, point2, last_id_block):  # Drawbox and id
    cv2.rectangle(img, point1, point2, (0, 255, 0), 1)  # Draw our rectangles

    cv2.putText(img, str(last_id_block),
        (round(point1[0]+(point2[0]-point1[0])/2)-10*len(str(last_id_block)), round(point1[1]+(point2[1]-point1[1])/2)+5 ), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #(point1[0], point1[1]-5 ), cv2.FONT_HERSHEY_SIMPLEX, 1,
        [0, 255, 0], 2)
    # Draw our rectangles
    a=2
    
    return 


def cvDrawBoxes(detections, img, time_start_between, time_width_element, time_space, detectionsBlock, last_id_block, rectangleblock):
    # ================================================================
    # 1. Purpose : Vehicle Counting
    # ================================================================
    width, height = img.shape[1], img.shape[0]
    levelUp = 125; #detection line top
    levelDown = height-720; #detection line bottom
    levelLeft=220
    levelRight=950

    n = 0
    threshold = 50
        
        
    cv2.rectangle(img, (0,height - 765), (levelLeft-1, height - 725), (0, 200, 0), thickness=cv2.FILLED)
    cv2.rectangle(img, (levelLeft,levelDown-10), (levelRight,levelDown), (120, 50, 110), thickness=cv2.FILLED)
    cv2.putText(img,
                str(last_id_block) + " ks.", (80, height - 735),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                [255, 255, 255], 2)
    cv2.putText(img,
                "Kvantita:", (5, height - 735),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                [255, 255, 255], 1)                
    speed=0.100    
    cv2.rectangle(img, (0,height - 725), (levelLeft-1, height - 685), (30, 75, 120), thickness=cv2.FILLED)
    cv2.putText(img,
                str(round(time_width_element*speed,2)) + " m.", (80, height - 695),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                [255, 255, 255], 1)
    cv2.putText(img,
                "Delka:", (26, height - 695),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                [255, 255, 255], 1)

    cv2.rectangle(img, (0,height - 685), (levelLeft-1, height - 645), (70, 25, 120), thickness=cv2.FILLED)
    cv2.putText(img,
                str(round(time_space*speed,2)) + " m.", (80, height - 655),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                [255, 255, 255], 1)
    cv2.putText(img,
                "Mezera:", (13, height - 655),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                [255, 255, 255], 1)


    #cv2.line(img, (0, levelUp), (width, levelUp), (50, 50, 50), thickness=2, lineType=8, shift=0)



    if len(detections) > 0:  # If there are any detections

        for detection in detections:  # For each detection
            name_tag = detection[0].decode()  # Decode list of classes
            if name_tag == 'glass':  # Filter detections for car class
                x, y, w, h = [detection[2][0], \
                              detection[2][1], \
                              detection[2][2], \
                              detection[2][3]]  # Obtain the detection coordinates
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))  # Convert to bounding box coordinates

                #if ymin<levelDown and ymax<levelDown:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                #установка времени конца длины элемента 
                if ymax<levelDown:
                    ind_found=-1 # indikator of found next rectangle in exist array
                    for i in range(len(rectangleblock)):
                                        
                                rectangleblock_xmin = rectangleblock[i][0] - threshold, rectangleblock[i][0] + threshold
                                rectangleblock_ymin = rectangleblock[i][1] - threshold, rectangleblock[i][1] + threshold
                                rectangleblock_xmax = rectangleblock[i][2] - threshold, rectangleblock[i][2] + threshold
                                rectangleblock_ymax = rectangleblock[i][3] - threshold, rectangleblock[i][3] + threshold
                                        
                                if (rectangleblock_xmin[0] < xmin < rectangleblock_xmin[1] and
                                rectangleblock_ymin[0] < ymin < rectangleblock_ymin[1] and
                                rectangleblock_xmax[0] < xmax < rectangleblock_xmax[1] and
                                rectangleblock_ymax[0] < ymax < rectangleblock_ymax[1]):
                                    ind_found=i
                                            
                    if ind_found>-1 and rectangleblock[ind_found][5]!=0: #если чтото найдено и время начала не равно нулю/был уже зафиксирован конец/
                        temp=rectangleblock[ind_found][4]
                        time_width_element=time.time()-rectangleblock[ind_found][5]#фиксируем время проезда длины элемента
                        rectangleblock[ind_found]=[xmin, ymin, xmax, ymax, temp, 0]#стираем время начала замера элемента для неопределения в дальнейшем
                        time_start_between=time.time()#фиксируем время начала замера пробела между элементами
                #усли верх элемента зашел за нижнюю линию 
                if ymin<levelDown and ymax>levelUp:
                    #если в массиве еще вообще пусто    
                    if not rectangleblock:
                        if levelLeft<xmin<levelRight and levelLeft<xmax<levelRight:
                            last_id_block+=1
                            rectangleblock.append([xmin, ymin, xmax, ymax, last_id_block, time.time()])
                            cv2.rectangle(img, (levelLeft,levelDown-10), (levelRight,levelDown), (0, 255, 0), thickness=cv2.FILLED)
                    else:
                        

                           
                        ind_found=0 # indikator of found next rectangle in exist array
                        for i in range(len(rectangleblock)):
                                
                            rectangleblock_xmin = rectangleblock[i][0] - threshold, rectangleblock[i][0] + threshold
                            rectangleblock_ymin = rectangleblock[i][1] - threshold, rectangleblock[i][1] + threshold
                            rectangleblock_xmax = rectangleblock[i][2] - threshold, rectangleblock[i][2] + threshold
                            rectangleblock_ymax = rectangleblock[i][3] - threshold, rectangleblock[i][3] + threshold
                            '''
                            cv2.rectangle(img, (rectangleblock_xmin[0],rectangleblock_ymin[0]), (rectangleblock_xmin[1], rectangleblock_ymin[1]), (255, 255, 255), 1)
                            cv2.rectangle(img, (rectangleblock_xmax[0],rectangleblock_ymax[0]), (rectangleblock_xmax[1], rectangleblock_ymax[1]), (255, 255, 255), 1)

                            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0), 1)
                            cv2.rectangle(img, (rectangleblock[i][0],rectangleblock[i][1]), (rectangleblock[i][2],rectangleblock[i][3]), (0, 0, 255), 1)
                            '''
                            amount=0 
                            if rectangleblock_xmin[0] < xmin < rectangleblock_xmin[1]:
                                amount+=25 
                            if rectangleblock_ymin[0] < ymin < rectangleblock_ymin[1]: 
                                amount+=25
                            if rectangleblock_xmax[0] < xmax < rectangleblock_xmax[1]:
                                amount+=25
                            if rectangleblock_ymax[0] < ymax < rectangleblock_ymax[1]:
                                amount+=25
                            if amount>=75:
                                ind_found=i+1
                        #если найден элемент в массиве обновляем его координаты                                                
                        if ind_found>0:
                            temp=rectangleblock[ind_found-1][5] #временное сохраненик ячейки начального времени верх грань прошла через нижнюю линию
                            rectangleblock[ind_found-1] = xmin, ymin, xmax, ymax, rectangleblock[ind_found-1][4],temp
                        #если ненайден элемент в массиве добавляем его в массив
                        elif ymin>levelUp:
                            if levelLeft<xmin<levelRight and levelLeft<xmax<levelRight: 
                                last_id_block+=1
                                rectangleblock.append([xmin, ymin, xmax, ymax, last_id_block, time.time()])
                                if time_start_between !=0:
                                    time_space=time.time()-time_start_between
                                cv2.rectangle(img, (levelLeft,levelDown-10), (levelRight,levelDown), (0, 255, 0), thickness=cv2.FILLED)#фото-финиш зеленая линия
                                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0), 1)
        #deleting elements upper up line from array
            if ymax<levelUp:
                ind_del=999
                for i in range(len(rectangleblock)):
                    rectangleblock_xmin = rectangleblock[i][0] - threshold, rectangleblock[i][0] + threshold
                    rectangleblock_ymin = rectangleblock[i][1] - threshold, rectangleblock[i][1] + threshold
                    rectangleblock_xmax = rectangleblock[i][2] - threshold, rectangleblock[i][2] + threshold
                    rectangleblock_ymax = rectangleblock[i][3] - threshold, rectangleblock[i][3] + threshold
                                        
                    if (rectangleblock_xmin[0] < xmin < rectangleblock_xmin[1] and
                    rectangleblock_ymin[0] < ymin < rectangleblock_ymin[1] and
                    rectangleblock_xmax[0] < xmax < rectangleblock_xmax[1] and
                    rectangleblock_ymax[0] < ymax < rectangleblock_ymax[1]):
                        ind_del=i
                if ind_del<999:
                    del rectangleblock[ind_del]
                    a=2
 
        #drawing rectangles over all detections
        if rectangleblock:
            #rectangleblock.sort(key = lambda x: x[4])
            #rectangleblock = sorted(rectangleblock, key=rectangleblock[4])

            for i in range(len(rectangleblock)):
                point1=rectangleblock[i][0],rectangleblock[i][1]
                point2=rectangleblock[i][2],rectangleblock[i][3] 
                drawBox(img, point1, point2, rectangleblock[i][4])

 

    return img, time_start_between, time_width_element, time_space, detectionsBlock, last_id_block, rectangleblock  # Return Image with detections
    # =================================================================#


netMain = None
metaMain = None
altNames = None


def YOLO():
    global metaMain, netMain, altNames
    configPath = "./backup/GLASS_ACURACY/yolo-obj.cfg"
    weightPath = "./backup/GLASS_ACURACY/yolo-obj_last_13102020.weights"
    metaPath = "./backup/GLASS/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # cap = cv2.VideoCapture(0)
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    cap = cv2.VideoCapture("./testg9.mp4")
    #cap = cv2.VideoCapture(filedialog.askopenfilename())

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # new_height, new_width = frame_height //2, frame_width//2
    new_height, new_width = frame_height, frame_width

    out = cv2.VideoWriter(
        "./test5_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (new_width, new_height))

    darknet_image = darknet.make_image(new_width, new_height, 3)

    time_start_between=0
    time_width_element = 0
    time_space=0
    detectionsBlock = []
    rectangleBlock = [] 
    #new_height, new_width = 200, 1080
    last_id_block=0 
    i = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        #frame_read = frame_read[350:550, 100:1180]

        # Create an image we reuse for each detect

        image_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb,
                               (new_width, new_height),
                               interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        image, time_start_between, time_width_element, time_space, detectionsBlock, last_id_block, rectangleBlock = cvDrawBoxes(detections, image_rgb, time_start_between,
                                                                               time_width_element, time_space, detectionsBlock, last_id_block, rectangleBlock)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fps = 1 / (time.time() - prev_time)
        print(fps)
        cv2.imshow('Output', image)
        cv2.waitKey(3)
        #out.write(image)

        i += 1
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO()
