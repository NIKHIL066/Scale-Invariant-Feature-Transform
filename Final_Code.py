import numpy
import cv2
import pylab as plt
import math
from tkinter import *
from tkinter import filedialog
import cv2

#Four octaves
octave1=[]
octave2=[]
octave3=[]
octave4=[]

#Four Difference of Gaussians
DoG1=[]
DoG2=[]
DoG3=[]
DoG4=[]

MAXSIZE=5
iname="dummy"
#Function to compute gaussian blur
def gaussianblur(img,sigma):


      if(sigma<0):
        print("SIGMA SHOULD BE POSITIVE")
        return;
      
      deno=(((math.sqrt(2*3.142*sigma*sigma))))
      k=[0,0,0,0,0]
      sum=0
      
      for x in range(-2,3):
       numo=math.exp(-((x*x)/(2*(sigma*sigma))))
       k[x+2]=numo/deno
       sum=sum+k[x+2]
       
  

      for x in range(0,5):
        k[x]=(k[x]/sum)
       
    

      empty_img = numpy.zeros((img.shape[0],img.shape[1],3), numpy.uint8)

      for i in range(0,img.shape[0]):
        for j in range(2,img.shape[1]-2):
            empty_img[i,j]=((img[i,j-2]*k[0])+(img[i,j-1]*k[1])+(img[i,j]*k[2])+(img[i,j+1]*k[3])+(img[i,j+2]*k[4]))

           
      return empty_img;








#function to blurr the images using Gassian Blurr API
def module1 (img,octave) :
		octave.append(img)
		for i in range(MAXSIZE) :
			blurred_image=gaussianblur(octave[i],1)
			octave.append(blurred_image)

#function to display octaves
def module2 (image,size):
		for i in range(size):
			plt.subplot(2,3,i+1),plt.imshow(image[i],'gray')
			name="image"+str(i)
			plt.title(name)
			plt.xticks([]),plt.yticks([])
		plt.show()


def module7(image):
      plt.subplot(2,3,1),plt.imshow(image,'gray')
      plt.xticks([]),plt.yticks([])
      plt.show()

#function to reduce image size by half
def module3 (octave):
		height=octave.shape[0]
		width=octave.shape[1]
		height=int(height/2)
		width=int(width/2)
		new_image=cv2.resize(octave,(width,height))
		#new_image=cv2.resize(octave,(width,height),fx=0.5,fy=0.5)
		return new_image
		
#function to find differnce of Gaussians
def module4 (image1,image2):
   dog = numpy.zeros((image1.shape[0],image1.shape[1],3), numpy.uint8)
   for i in range(0,image1.shape[0]):
       for j in range(0,image1.shape[1]):
             dog[i,j]=abs(image1[i,j]-image2[i,j])
      
             
   return dog;


def module5 (image,m,x,y,z):
    #max_val=image[x][x][y][z]
    max_list=[]
    for i in range(y,y+3):
          for j in range(z,z+3):
                max_list.append(image[m][x][y][z])
    return max((max_list[0]+max_list[1]+max_list[2])/3)
    
def module6 (image,m,x,y,z):
    min_list=[]
    for i in range(y,y+3):
          for j in range(z,z+3):
                min_list.append(image[m][x][y][z])
    return min((min_list[0]+min_list[1]+min_list[2])/3)



class window(Frame):
    def _init_(self):
        Frame._init_(self,master)
        self.master=master
        self.init_window()
    
root=Tk()
root.title('OBJECT RECOGNITION IN IMAGE')
root.geometry("1350x700")

def load_file():
    fname = filedialog.askopenfilename(filetypes=(("All files", "*.*"),
                                           ("jpg files", "*.jpg*"),
                                           ("png files", "*.png*") ))
    url.set(fname)
    iname=fname
def process():

    if len(url.get()) == 0:
          print("Please give the file name");
          exit()
    root.destroy()
    img = cv2.imread(url.get(),cv2.IMREAD_GRAYSCALE)
    module1(img,octave1)


    resized_image=module3(octave1[0])


    module1(resized_image,octave2)


    resized_image=module3(octave2[0])

    module1(resized_image,octave3)


    resized_image=module3(octave3[0])

    module1(resized_image,octave4)


    #Steps to display four octaves
    module2(octave1,MAXSIZE)
    module2(octave2,MAXSIZE)
    module2(octave3,MAXSIZE)
    module2(octave4,MAXSIZE)

    for index in range(4):
      DoG1.append(module4(octave1[index],octave1[index+1]))
    module2(DoG1,MAXSIZE-1)	
    h,w=octave1[0].shape
    DoG_List=[]
    DoG_List.append(DoG1)

    for index in range(4):
      DoG2.append(module4(octave2[index],octave2[index+1]))
    module2(DoG2,MAXSIZE-1)
    DoG_List.append(DoG2)

    for index in range(4):
      DoG3.append(module4(octave3[index],octave3[index+1]))
    module2(DoG3,MAXSIZE-1)
    DoG_List.append(DoG3)

    for index in range(4):
      DoG4.append(module4(octave4[index],octave4[index+1]))
    module2(DoG4,MAXSIZE-1)
    DoG_List.append(DoG4)

    img_extrem_List=[]
    img_List=[]
    for i  in range(4):
      image_scaleList=[]
      for j in range(1,3):
        image_extreme=numpy.zeros(DoG_List[i][j].shape,dtype=numpy.uint8)
        for l in range(DoG_List[i][j].shape[0]):
          for m in range(DoG_List[i][j].shape[1]):
            ext_points=(DoG_List[i][j][l][m][0]+DoG_List[i][j][l][m][1]+DoG_List[i][j][l][m][2])/3
            if (ext_points==max(module5(DoG_List,i,j,l-1,m-1),module5(DoG_List,i,j-1,l-1,m-1),module5(DoG_List,i,j+1,l-1,m-1))):
              image_extreme[l][m]=DoG_List[i][j][l][m][0],DoG_List[i][j][l][m][1],DoG_List[i][j][l][m][2]
            elif (ext_points==min(module6(DoG_List,i,j,l-1,m-1),module6(DoG_List,i,j-1,l-1,m-1),module6(DoG_List,i,j+1,l-1,m-1))):
              image_extreme[l][m]=DoG_List[i][j][l][m][0],DoG_List[i][j][l][m][1],DoG_List[i][j][l][m][2]
                    #print (ext_points)

      image_scaleList.append(image_extreme)
      module7(image_extreme)
    img_extrem_List.append(image_scaleList)
    
def comeout():
    exit()
    
#code for initializing GUI and its components
C = Canvas(root, bg="powder blue", height=800, width=1350)
C.pack()

l1=Label(root,text="OBJECT RECOGNITION",font=50)
l1.pack()
l1.place(x=600,y=100)

url=StringVar()
t=Entry(root,textvariable=url)
t.pack()
t.place(x=550,y=200)

b1=Button(root,text="browse", height = 1, width = 20,command=load_file)
b1.pack()
b1.place(x=750,y=200)

b2=Button(root,text="exit",width=20,command=comeout)
b2.pack()
b2.place(x=550,y=250)

b2=Button(root,text="submit", width=20,command=process)
b2.pack()
b2.place(x=750,y=250)




root.mainloop()

        
