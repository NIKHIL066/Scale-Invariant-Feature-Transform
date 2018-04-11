import numpy
import cv2
from pylab import *
import copy as cp
from scipy import ndimage
import math

scales = 5
octaves = 4
sigma = 1.7
k = 2**0.5



def ScaleSpace(image1,n):
    sigma = 1.7
    k = 2**0.5
    scales = 5
    octaves = 4
    base_image = np.zeros((shape(image1)))
    base_image[:]= image1
    image_octaveList=[]
    image_baseList = []
    for i in range(octaves):
        image_scaleList=[]
        for j in range(scales):
            
            if i==0 and j==0:
                temp1=cp.deepcopy(base_image)
                image_scaleList.append(temp1)
            elif i>0 and j==0:
                temp2=ndimage.zoom(image_baseList[i-1][0],0.5, order =1)
                temp3=cp.deepcopy(temp2)
                image_scaleList.append(temp3)
        
        image_baseList.append(image_scaleList)
     
      
    for i in range(octaves):
        image_scaleList=[]
        for j in range(scales):
            
            if j==0:
                temp1 =np.zeros(np.shape(image_baseList[i][0]))
                temp1[:]=image_baseList[i][0]
            sigma=math.pow(k,j)*1.7
            histogram_size= int(math.ceil(7*sigma))
            histogram_size= 2*histogram_size+1
            
            temp2=temp3=np.zeros(np.shape(temp1))
            temp2=cv2.GaussianBlur(temp1,(histogram_size,histogram_size),sigma,sigma)
            image_scaleList.append(temp2)
       
        image_octaveList.append(image_scaleList)  

    return image_octaveList

def DoG_Space(image_octaveList):
    DoG_List=[]
    for i in range(octaves):
        image_scaleList=[]
        for j in range(1,scales):
            
            difference = np.zeros(np.shape(image_octaveList[i][0]))
            difference[:]= np.subtract(image_octaveList[i][j],image_octaveList[i][j-1])
            image_scaleList.append(difference)
            
        DoG_List.append(image_scaleList)    
    return image_scaleList,DoG_List

     
def Local_Extrema(DoG_List):
    c1=0 
    image_extremumList=[]
    for i in range(octaves):
        image_scaleList=[]
        for j in range(1, scales-2):
            image_extremum=np.zeros(DoG_List[i][j].shape,dtype=np.float64)
         
            for l in range(1, DoG_List[i][j].shape[0]):
                for m in range(1, DoG_List[i][j].shape[1]):
                    ext_points= DoG_List[i][j][l][m]
                    if ext_points == max(DoG_List[i][j][l-1:l+2, m-1:m+2].max(), DoG_List[i][j-1][l-1:l, m-1:m+2].max(), DoG_List[i][j+1][l-1:l+2, m-1:m+2].max()):
                        image_extremum[l][m]= ext_points
                        c1+=1
                    elif ext_points== min(DoG_List[i][j][l-1:l+2, m-1:m+2].min(), DoG_List[i][j-1][l-1:l+2, m-1:m+2].min(), DoG_List[i][j+1][l-1:l+2, m-1:m+2].min()):
                        image_extremum[l][m]= ext_points
                        c1+=1
            image_scaleList.append(image_extremum)
        image_extremumList.append(image_scaleList)
    return image_scaleList,image_extremumList

def Non_Zero_Extrema(image_extremumList,image1,n):
    key_points=0
    sigma_nonzero=[]
    extremum_nonzero=[]
    for i in range(octaves):
        image_sigmaList=[]
        image_scaleList=[]
        for j in range(scales-3):
            temp4=[]
            temp4[:] = np.transpose(image_extremumList[i][j].nonzero())
            key_points+=len(temp4)
            image_scaleList.append(temp4)
            image_sigmaList.append(math.pow(k,j)*1.6)
        extremum_nonzero.append(image_scaleList)
        sigma_nonzero.append(image_sigmaList)
    plt.gray()
    plt.figure(n+1)
    image2=np.zeros(np.shape(image1))
    plt.imshow(image2)
    for i in range(octaves):
        for j in range(0,2):
            for l in range(len(extremum_nonzero[i][j])):
                x=math.pow(2,i)*extremum_nonzero[i][j][l][0]
                y=math.pow(2,i)*extremum_nonzero[i][j][l][1]
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'ro')
    plt.title('Non-Zero Extremum Points')
    
    return sigma_nonzero,extremum_nonzero

def Accurate_Extrema(sigma_nonzero,extremum_nonzero,
    image_scaleList,DoG_List,image1,n,image_octaveList):
    c2=1 
    c3=0 
    extremum_points= []
    for i in range(octaves):
        image_scaleList=[]
        for j in range(2):
            c2=1
            keyPointsPerScale =[]
            
            for l in range(len(extremum_nonzero[i][j])):
                matrix_A = np.zeros((3,3))
                matrix_B = np.zeros((3,1))
                x_coord= extremum_nonzero[i][j][l][0]
                y_coord= extremum_nonzero[i][j][l][1]
                sigma_current = sigma_nonzero[i][j] 
                
                if(x_coord+1 < DoG_List[i][0].shape[0] and y_coord+1 < DoG_List[i][0].shape[1] and x_coord-1 >-1 and y_coord-1 >-1):
                    x_newcoord=x_coord
                    y_newcoord=y_coord
                    xnew=np.zeros((3,1))
                    sigma_new = sigma_current
                    
                    matrix_A[0][0] = DoG_List[i][j][x_coord][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+2][x_coord][y_coord]
                    matrix_A[0][1] = DoG_List[i][j+2][x_coord+1][y_coord] -DoG_List[i][j+2][x_coord-1][y_coord] - DoG_List[i][j][x_coord+1][y_coord] + DoG_List[i][j][x_coord-1][y_coord]
                    matrix_A[0][2] = DoG_List[i][j+2][x_coord][y_coord+1] -DoG_List[i][j+2][x_coord][y_coord-1] - DoG_List[i][j][x_coord][y_coord+1] + DoG_List[i][j-2][x_coord][y_coord-1]
                   
                    matrix_A[1][0] = matrix_A[0][2]
                    matrix_A[1][1] = DoG_List[i][j+1][x_coord+1][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord-1][y_coord]
                    matrix_A[1][2] = DoG_List[i][j+1][x_coord-1][y_coord-1] - DoG_List[i][j+1][x_coord+1][y_coord-1]  - DoG_List[i][j+1][x_coord-1][y_coord+1] + DoG_List[i][j+1][x_coord+1][y_coord+1]
                   
                    matrix_A[2][0] = matrix_A[0][2]
                    matrix_A[2][1] = matrix_A[1][2]
                    matrix_A[2][2] = DoG_List[i][j+1][x_coord][y_coord+1] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord][y_coord-1]

                    matrix_B[0][0] =  DoG_List[i][j+2][x_coord][y_coord] - DoG_List[i][j][x_coord][y_coord]
                    matrix_B[1][0] =  DoG_List[i][j+1][x_coord+1][y_coord]- DoG_List[i][j+1][x_coord-1][y_coord]
                    matrix_B[2][0] =  DoG_List[i][j+1][x_coord][y_coord+1]- DoG_List[i][j+1][x_coord][y_coord-1]
                    
                    xdash=np.dot(np.linalg.pinv(matrix_A),matrix_B)
                    xnew[:] = xdash
                    
                    skipPoint=0
                    if abs(xdash[0][0])>0.5 or abs(xdash[1][0])>0.5 or abs(xdash[2][0])>0.5:
                        skipPoint=1      
                        if abs(xdash[1][0])>0.5 :
                            x_newcoord = x_coord + round(xdash[1][0])
                            xnew[1][0] = xdash[1][0]- round(xdash[1][0])
                            if (x_newcoord > image_octaveList[i][0].shape[0]-1) or x_newcoord <0:
                                skipPoint =1
                                
                        if abs(xdash[2][0])>0.5:
                            y_newcoord= y_coord + round(xdash[2][0])
                            xnew[2][0] = xdash[2][0] - round(xdash[2][0])
                            if (y_newcoord > image_octaveList[i][0].shape[1]-1) or y_newcoord<0:
                                skipPoint =1
                        
                        if abs(xdash[0][0])>0.5:
                            if xdash[0][0]> 0 :
                                sigma_new = math.pow(k, (j+1))*1.6
                                xnew[0][0] = (sigma_new - math.pow(k,j)*1.6) - xdash[0][0]
                            else:
                                sigma_new = math.pow(k,(j-1))*1.6
                                xnew[0][0] = (math.pow(k,j)*1.6 - sigma_new) + xdash[0][0]
    
                    if (skipPoint==0):
                        contrast_keypoint = DoG_List[i][j+1][x_newcoord][y_newcoord] + 0.6 * matrix_B[1][0] *xnew[2][0] + matrix_B[2][0]*xnew[2][0] + matrix_B[0][0] * xnew[0][0]
                        if abs(contrast_keypoint)>0.03:
                            diff_xx = DoG_List[i][j+1][x_coord+1][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord-1][y_coord]
                            diff_xy = DoG_List[i][j+1][x_coord-1][y_coord-1] - DoG_List[i][j+1][x_coord+1][y_coord-1] + DoG_List[i][j+1][x_coord-1][y_coord+1] +DoG_List[i][j+1][x_coord+1][y_coord+1]
                            diff_yy = DoG_List[i][j+1][x_coord][y_coord+1] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord][y_coord-1]
                           
                            trace_H = diff_xx + diff_yy
                            determinant_H = diff_xx * diff_yy - diff_xy**2
                            curvature_ratio = (trace_H*trace_H)/determinant_H
                            if abs(curvature_ratio)<10.0:
                                key_attributePoints = []
                               
                                key_attributePoints.append(c2)
                                key_attributePoints.append(x_newcoord)
                                key_attributePoints.append(y_newcoord)
                                
                                key_attributePoints.append(sigma_new)
                                
                                key_attributePoints.append(xnew[0][0])
                                key_attributePoints.append(xnew[1][0])
                                key_attributePoints.append(xnew[2][0])
                                
                                key_attributePoints.append(x_coord)
                                key_attributePoints.append(y_coord)
                                
                                key_attributePoints.append(sigma_current)
                                key_attributePoints.append(j+1)
                               
                                c2= c2+1
                                keyPointsPerScale.append(key_attributePoints)
                                c3 +=1
            
            image_scaleList.append(keyPointsPerScale)
        extremum_points.append(image_scaleList)
    plt.gray()
    plt.figure(n+2)
    image2=np.zeros(np.shape(image1))
    plt.imshow(image2)
    
    for i in range(octaves):
        for j in range(2):
            for l in range(len(extremum_points[i][j])):
                x=math.pow(2,i)*extremum_points[i][j][l][1]
                y=math.pow(2,i)*extremum_points[i][j][l][2]
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'ro') 
    plt.title('Key Points')            
    return image_scaleList,extremum_points

def Orientation_Assigner(image_scaleList,extremum_points,image_octaveList):
    c4 = []
    c5 = 0
    for i in  range (octaves):
        image_scaleList = []
        for j in range(scales-3):
            c2 = 1

            keyPointsPerScale = []
            for p in  range(len(extremum_points[i][j])):
                x_coord = extremum_points[i][j][p][1]
                y_coord = extremum_points[i][j][p][2]
                
                sig = extremum_points[i][j][p][3]
                IOr = np.zeros(image_octaveList[i][j].shape)
                IOr = image_octaveList[i][j]
                
                histogram_size = int(math.ceil(7*sig))
              
                Iblur = np.zeros(IOr.shape)
               
                H = cv2.getGaussianKernel(histogram_size,int(sig));
                Iblur[:                                                                                                                                                                                                                                                                                                                                                                                                                                                     ] = cv2.filter2D(IOr,-1,H);

                bins = np.zeros((1,36));
                for s in range(-histogram_size,histogram_size+1):
                    for t in range(-histogram_size,histogram_size+1):
                        if (((x_coord + s)>0) and ((x_coord + s)<(Iblur.shape[0]-1)) and ((y_coord+ t)>0) and ((y_coord + t)<(Iblur.shape[1]-1))):
                            xmag1 = Iblur[x_coord+s+1][y_coord+t]
                            xmag2 = Iblur[x_coord+s-1][y_coord+t]
                            
                            ymag1 = Iblur[x_coord+s][y_coord+t+1]
                            ymag2 = Iblur[x_coord+s][y_coord+t-1]
                            m = math.sqrt(math.pow((xmag1-xmag2),2) + math.pow((ymag1-ymag2),2))
                            den = xmag2-xmag1
                            if den==0:
                               den = 5
                            theta = math.degrees(math.atan((ymag2-ymag1)/(den)))
                            
                            if(theta<0):
                                theta = 360 + theta                           
                            binary = (int)((theta/360)*36)%36
                            
                           
                            if binary ==36:
                                binary = 35
                            bins[0][binary] = bins[0][binary] + m

                maxBinNo = np.argmax(bins)
                maxtheta = maxBinNo*10
                maxmag = bins[0][maxBinNo]
                
                extremum_points[i][j][p].append(maxtheta)
                extremum_points[i][j][p].append(maxmag)


                nbins = 36
                threshold = 0.8
                o = 0
                for y in range(0,36):
                    orientation = 0
                    y_prev = (y-1+nbins)%nbins
                    y_next = (y+1)%nbins
                    
                    if bins[0][y] > threshold*maxtheta and bins[0][y] > bins[0][y_prev] and  bins[0][y]> bins[0][y_next]:
                        offset = (bins[0][y_prev] - bins[0][y_next])/(2*(bins[0][y_prev]+bins[0][y_next]-2*bins[0][y]))
                        exact_bin = y + offset
                        orientation = exact_bin*360/float(36)
                        
                        if orientation>360:
                            orientation-=360
                       
                        o+=1
                        extPtskey_attributePoints = []
                        extPtskey_attributePoints[:] = extremum_points[i][j][p]
                        extPtskey_attributePoints[11] = orientation
                        keyPointsPerScale.append(extPtskey_attributePoints)
            c5 +=len(keyPointsPerScale)
            image_scaleList.append(keyPointsPerScale)
        c4.append(image_scaleList)
    return image_scaleList

def Image_Descriptor(image_octaveList,extremum_points,image1,n):
    dx_list  = []
    dy_list  = []
    for i in range(len(image_octaveList)):
        image_scaleList1 = []
        image_scaleList2 = []
        for j in range(scales):
            dx,dy = np.gradient(image_octaveList[i][j])
            image_scaleList1.append(dx)
            image_scaleList2.append(dy)
        dx_list.append(image_scaleList1)
        dy_list.append(image_scaleList2)

    const = 3
    plt.gray()
    plt.figure(n+3)
    image2=np.zeros(np.shape(image1))
    plt.imshow(image2)
            
    for i in range(octaves):
        for j in range(2):
            for l in range(len(extremum_points[i][j])):
                x=math.pow(2,i)*extremum_points[i][j][l][1]
                y=math.pow(2,i)*extremum_points[i][j][l][2]
                dx =  const*extremum_points[i][j][l][3] * math.degrees(math.cos(extremum_points[i][j][l][10]))
                dy =  const*extremum_points[i][j][l][3] * math.degrees(math.sin(extremum_points[i][j][l][10]))
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'ro')
    plt.title('Image Descriptor Key Points')
    plt.figure(n+4)
    plt.imshow(image1)
    plt.title('Original Image')
    plt.show()


def SIFT(image1,n):    
    
    image_octaveList=ScaleSpace(image1,n)
    
    
    
    image_scaleList,DoG_List=DoG_Space(image_octaveList)
        
    image_scaleList,image_extremumList=Local_Extrema(DoG_List)        
                
                    
    sigma_nonzero,extremum_nonzero=Non_Zero_Extrema(image_extremumList,image1,n)                    
                        
                                
    image_scaleList,extremum_points=Accurate_Extrema(sigma_nonzero,extremum_nonzero,
    image_scaleList,DoG_List,image1,n,image_octaveList)  

    Image_Descriptor(image_octaveList,extremum_points,image1,n)
   

   
      
      
      
            

input_image1= cv2.imread('hot_air_balloon.jpg',0)
input_image2 = cv2.imread('hot_air_balloon.jpg',0)

SIFT(input_image1,0)
SIFT(input_image2,3)
