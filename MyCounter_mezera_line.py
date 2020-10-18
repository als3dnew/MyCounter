# ================================================================
#  To learn how to Develop Advance YOLOv4 Apps - Then check out:
#  https://augmentedstartups.info/yolov4release
# ================================================================
from array import *
from ctypes import *
import os
import cv2
import darknet
import glob
import time
import tkinter as tk
import numpy as np
import math
from deep_sort.tracker import Tracker


def convertBack(x, y, w, h):  # Convert from center coordinates to bounding box coordinates
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def drawBox(img, point1, point2, last_id_block):  # Drawbox and id
    cv2.rectangle(img, point1, point2, (0, 255, 0), 1)  # Draw our rectangles
    cv2.putText(img, str(last_id_block),
        (round(point1[0]+(point2[0]-point1[0])/2)-5, round(point1[1]+(point2[1]-point1[1])/2)+5 ), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #(point1[0], point1[1]-5 ), cv2.FONT_HERSHEY_SIMPLEX, 1,
        [0, 255, 0], 2)
    return 


def cvDrawBoxes(detections, img, amount_glases, left_corner_indif, detectionsBlock, last_id_block, rectangleblock):
    # ================================================================
    # 1. Purpose : Vehicle Counting
    # ================================================================

    if len(detections) > 0:  # If there are any detections
        
        n = 0
        threshold = 50
        width, height = img.shape[1], img.shape[0]
                        
        cv2.putText(img,
                    str(last_id_block) + " ks.", (25, height - 730),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [255, 0, 0], 2)
        # setting levels
        levelUp = 150; #detection line top
        levelDown = height-150; #detection line bottom
        levelLeft=0
        levelRight=width
        cv2.line(img, (0, levelUp), (width, levelUp), (120, 50, 110), thickness=7, lineType=7, shift=0)
        cv2.line(img, (levelLeft, levelDown), (levelRight, levelDown), (120, 50, 110), thickness=7, lineType=8, shift=0)


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
                if ymin<levelDown and ymax>levelUp:

                    if not rectangleblock:
                        if levelLeft<xmin<levelRight and levelLeft<xmax<levelRight:
                            last_id_block+=1
                            rectangleblock.append([xmin, ymin, xmax, ymax, last_id_block])
                            cv2.line(img, (0, levelDown), (width, levelDown), (0, 255, 0), thickness=7, lineType=8, shift=0)
                        

                            
                    else:
                        


                        ind_found=0 # indikator of found next rectangle in exist array
                        for i in range(len(rectangleblock)):
                                        
                                    rectangleblock_xmin = rectangleblock[i][0] - threshold, rectangleblock[i][0] + threshold
                                    rectangleblock_ymin = rectangleblock[i][1] - threshold, rectangleblock[i][1] + threshold
                                    rectangleblock_xmax = rectangleblock[i][2] - threshold, rectangleblock[i][2] + threshold
                                    rectangleblock_ymax = rectangleblock[i][3] - threshold, rectangleblock[i][3] + threshold
                                        
                                    if (rectangleblock_xmin[0] < xmin < rectangleblock_xmin[1] and
                                    rectangleblock_ymin[0] < ymin < rectangleblock_ymin[1] and
                                    rectangleblock_xmax[0] < xmax < rectangleblock_xmax[1] and
                                    rectangleblock_ymax[0] < ymax < rectangleblock_ymax[1]):
                                        ind_found=i+1
                                            
                        if ind_found>0:
                            rectangleblock[ind_found-1] = xmin, ymin, xmax, ymax, rectangleblock[ind_found-1][4]
                        elif ymin>levelUp:
                            if levelLeft<xmin<levelRight and levelLeft<xmax<levelRight: 
                                last_id_block+=1
                                rectangleblock.append([xmin, ymin, xmax, ymax, last_id_block])
                                cv2.line(img, (0, levelDown), (width, levelDown), (0, 255, 0), thickness=7, lineType=8, shift=0)

                

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
 
        #drawing rectangles over all detections
        if rectangleblock:
            #rectangleblock.sort(key = lambda x: x[4])
            #rectangleblock = sorted(rectangleblock, key=rectangleblock[4])
            for i in range(len(rectangleblock)):
                point1=rectangleblock[i][0],rectangleblock[i][1]
                point2=rectangleblock[i][2],rectangleblock[i][3] 
                drawBox(img, point1, point2, rectangleblock[i][4])


        #mezera
        for i in range(len(rectangleblock)):
            upperNeighbor = rectangleblock[i][0] - threshold*4, rectangleblock[i][0] + threshold*4
            ind_founded=0
            for j in range(len(rectangleblock)): 
                if  rectangleblock[j][1] > rectangleblock[i][1]+threshold*2 and upperNeighbor[0]<rectangleblock[j][0]<upperNeighbor[1] and j!=i and ind_founded==0:
                    pt1=round((rectangleblock[i][2]-rectangleblock[i][0])/2)+rectangleblock[i][0],rectangleblock[i][3]
                    pt2=round((rectangleblock[j][2]-rectangleblock[j][0])/2)+rectangleblock[j][0],rectangleblock[j][1]
                    delka=round(math.sqrt((pt2[1]-pt2[0])*(pt2[1]-pt2[0])+(pt1[1]-pt1[0])*(pt1[1]-pt1[0])))
                    ind_founded=1
                    cv2.line(img, pt1, pt2, (0, 255, 255), thickness=2, lineType=8, shift=0)
                    cv2.putText(img, str(delka),
                                pt1, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #(point1[0], point1[1]-5 ), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                [0, 255, 0], 2)
                    



    return img, amount_glases, left_corner_indif, detectionsBlock, last_id_block, rectangleblock  # Return Image with detections
    # =================================================================#


netMain = None
metaMain = None
altNames = None


def YOLO():
    global metaMain, netMain, altNames
    configPath = "./backup/GLASS/yolo-obj_detect.cfg"
    weightPath = "./backup/GLASS/yolo-obj_last.weights"
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

    amount_glases = 0
    left_corner_indif = 0
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

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.95)
        image, amount_glases, left_corner_indif, detectionsBlock, last_id_block, rectangleBlock = cvDrawBoxes(detections, image_rgb, amount_glases,
                                                                               left_corner_indif, detectionsBlock, last_id_block, rectangleBlock)

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
