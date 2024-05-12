from PIL import Image
import os
import cv2
import numpy as np
import time
from skimage import io, color
from sklearn.cluster import DBSCAN
import random
import matplotlib.pyplot as plt
from webcolors import CSS3_NAMES_TO_HEX
from collections import Counter


dest="C:/Users/Rwitobroto/Desktop/Test/"
os.chdir("C:/Users/Rwitobroto/Desktop/Test")
def binarize(img, threshold):
    img=img.convert('L')
    img_array=np.array(img)
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i][j] > threshold:
                img_array[i][j] = 255
            else:   
                img_array[i][j] = 0
    return img_array
def count_filled_circles(labeled_image,num_labels):
    counts_per_region = {}
    for label in range(2, num_labels):
        mask = np.uint8(labeled_image == label) * 255
        area1=np.count_nonzero(mask>=255-9)
        if area1>100:
            filled_circle_count = -2
            mask=cv2.threshold(mask,145,255,cv2.THRESH_BINARY_INV)[1]
            numlbl,lbl=cv2.connectedComponents(mask)
            for lbls in range(0,numlbl):
                mask1=np.uint8(lbl==lbls)*255
                area=np.count_nonzero(mask1>=(255-9))
                #print(f"Area {lbls} of label {label} is ",area)
                per_area=area/mask1.size
                #print(per_area)
                if per_area>=0.001:
                    filled_circle_count+=1
            counts_per_region[label] = filled_circle_count
    return counts_per_region
def tuplecreate(img):
    inf=[]
    num_labels, labels=cv2.connectedComponents(img)
    filled_circle_counts = count_filled_circles(labels,num_labels)
    for region, count in filled_circle_counts.items():
        if count>=0:
            inf.append(count)
    return inf
def clustering(frame1):
    frame=np.array(frame1)
    climg={}
    data = []
    for i in range(1,len(frame)-1):
        for j in range(1,len(frame[0])-1):
            r=frame[i][j]
            if r <= 8:
                data.append([i,j])            
    data = np.array(data)
    print(len(data))
    clustering = DBSCAN(eps=7.5, min_samples=1).fit(data)
    n = len(list(set(clustering.labels_)))
    clusters = [[] for i in range(n)]
    for i in range(len(data)):
        clusters[clustering.labels_[i]].append(i)
    X = [[data[i][1] for i in j] for j in clusters]
    Y = [[data[i][0] for i in j] for j in clusters]
    for i in range(len(clusters)):
        im1 = frame1.crop((min(X[i])-7,min(Y[i])-7,max(X[i])+7,max(Y[i])+7))
        im1.save(dest+"letter"+str(i+1)+".jpg")
        climg[i]=cv2.imread(dest+"letter"+str(i+1)+".jpg")
    return climg
def to_letter(ls):
    if ls==[0]:
        return chr(32)
    n=len(ls)
    no=2**(n)-1
    temp=n-1
    for i in range (len(ls)):
        no=no+ls[i]*(2**temp)
        temp=temp-1
    if no<=32:
        no=no-1
    if no<=126:
        return chr(no)
    else:
        return 'null'
def process_images(st):
    temp=Image.open(st)
    temp=binarize(temp,145)
    temp=Image.fromarray(temp)
    temp.save(dest+"Binarised.jpg")
    frame1=Image.open(dest+"Binarised.jpg")
    climg=clustering(frame1)
    infdict={}
    str=""
    for i in range (len(climg)):
        climg[i]=cv2.cvtColor(climg[i],cv2.COLOR_BGR2GRAY)
        climg[i]=cv2.threshold(climg[i],145,255,cv2.THRESH_BINARY)[1]
        lst=tuplecreate(climg[i])
        infdict[i]=lst
        if len(lst)!=0:
            s=to_letter(infdict[i])
            if (s != 'null'):
                str=str+""+s
    return str
def main():
    str=input("Enter the path of the input : ")
    st=process_images(str)
    print(st)
if __name__ == "__main__":
    main()
