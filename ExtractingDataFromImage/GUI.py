import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
import time
from tkinter import messagebox
import threading
import ImageExtract
from time import sleep
class MyVideoCapture:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, 1280)
        self.vid.set(4, 720)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source",video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame,(1280,720))
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret,None)
class MY_GUI:
    def __init__(self, Window):
        self.root = Window
        self.main_frame = tk.Frame(self.root)
        # Code for introduction page 
        self.page_1 = tk.Frame(self.main_frame,width= 500,height =630,bg = 'white')

        logoHCMUTEImage = Image.open("logo-hcmute-inkythuatso.jpg")
        logoHCMUTEImage = logoHCMUTEImage.resize((86,109),Image.LANCZOS)
        logoHCMUTE = ImageTk.PhotoImage(logoHCMUTEImage)
        self.page_1_LB1 = tk.Label(self.page_1,image=logoHCMUTE, borderwidth=0)
        self.page_1_LB1.image = logoHCMUTE
        self.page_1_LB1.place(x=31,y=46)


        self.page_1_LB2 = tk.Label(self.page_1, text='UNIVERSITY OF TECHNOLOGY AND EDUCATION', font=('Inner', 11),bg='white',borderwidth=0,fg='red')
        self.page_1_LB2.place(x=142,y=76)
        
        self.page_1_LB3 = tk.Label(self.page_1, text='FACULTY OF MANUFACTURING AND ENGINEERING', font=('Inner', 10),bg='white',borderwidth=0,fg='blue')
        self.page_1_LB3.place(x=142,y=107)

        self.page_1_LB4 = tk.Label(self.page_1, text='ROBOTICS AND ARTIFICIAL INTELLIGENT BRANCH', font=('Inner', 13),bg='white',borderwidth=0,fg='#00D1FF')
        self.page_1_LB4.place(x=51,y=211)

        self.page_1_LB5 = tk.Label(self.page_1, text='EXTRACTING INFORMATION FROM \n MEDICAL LABELS', font=('Inner', 15),width = 35, height =2,bg='white',borderwidth=0,fg='red')
        self.page_1_LB5.place(x=49,y=262)

        self.page_1_LB6 = tk.Label(self.page_1, text='Lecturer: Nguyen Van Thai', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_1_LB6.place(x=250,y=412)

        self.page_1_LB7 = tk.Label(self.page_1, text='Students:', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_1_LB7.place(x=250,y=449)

        self.page_1_LB8 = tk.Label(self.page_1, text='Lai The Trung              21151405', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_1_LB8.place(x=250,y=472)

        self.page_1_LB9 = tk.Label(self.page_1, text='Tran Nhat Hoang         21134008', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_1_LB9.place(x=250,y=495)

        self.page_1_LB10 = tk.Label(self.page_1, text='Vu Chi Dat                  21134004', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_1_LB10.place(x=250,y=518)

        medicBottleImage = Image.open("image_1.jpg")
        medicBottleImage = medicBottleImage.resize((175,175),Image.LANCZOS)
        medicBottle = ImageTk.PhotoImage(medicBottleImage)
        self.page_1_LB1 = tk.Label(self.page_1,image=medicBottle, borderwidth=0)
        self.page_1_LB1.image = medicBottle
        self.page_1_LB1.place(x=36,y=366)

        self.page_1.pack(fill=tk.BOTH, expand=True)

        # Code for application page
        self.page_2 = tk.Frame(self.main_frame,width= 500,height =630,bg = 'white')
        self.video_source = 0
        self.fpsLimit = 30
        self.vid = MyVideoCapture(self.video_source)
        self.listImage = []    
        self.lenListImage = 0
        self.maxImage = 0
        self.isCylinder = False
        self.grayImage = Image.fromarray(np.ones((100,400,3),dtype=np.uint8)*125)
        self.start_Working = True
        self.on_Working = False


        self.page_2_canvas = tk.Canvas(self.page_2, width = 355, height = 200)
        self.page_2_canvas.place(x=21,y=15)

        self.gray = ImageTk.PhotoImage(self.grayImage)
        self.page_2_IMG1 = tk.Label(self.page_2,image=self.gray, borderwidth=0)
        self.page_2_IMG1.image = self.gray
        self.page_2_IMG1.place(x=53,y=321)

        self.page_2_IMG2 = tk.Label(self.page_2,image=self.gray, borderwidth=0)
        self.page_2_IMG2.image = self.gray
        self.page_2_IMG2.place(x=53,y=475)        

        self.page_2_B1 = tk.Button(master=self.page_2, text='CAPTURE', font=('Inner', 10), width=15,height=2, borderwidth= 0, bg = "#8FFF00", command=self.capture)
        self.page_2_B1.place(x=38, y=229)        

        self.page_2_B2 = tk.Button(master=self.page_2, text='GET NEW', font=('Inner', 10), width=15,height=2, borderwidth= 0, bg = "#8FFF00", command=self.getNewData)
        self.page_2_B2.place(x=191, y=229)  

        self.page_2_B3 = tk.Button(master=self.page_2, text='EXTRACT', font=('Inner', 10), width=15,height=2, borderwidth= 0, bg = "#8FFF00",command= lambda: threading.Thread(target=self.extracting).start())
        self.page_2_B3.place(x=342, y=229)  

        self.page_2_B4 = tk.Button(master=self.page_2, text='Cylinder', font=('Inner', 10), width=10,height=2, borderwidth= 0, bg = "#00D1FF", command=self.cylinder)
        self.page_2_B4.place(x=393, y=60) 

        self.page_2_B5 = tk.Button(master=self.page_2, text='Box', font=('Inner', 10), width=10,height=2, borderwidth= 0, bg = "#00D1FF", command=self.box)
        self.page_2_B5.place(x=393, y=124) 

        self.page_2_LB1 = tk.Label(master=self.page_2, text='Raw Image', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_2_LB1.place(x=50,y=286)
        
        self.page_2_LB2 = tk.Label(master=self.page_2, text='Extract Image', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_2_LB2.place(x=50,y=440)

        self.page_2_LB3 = tk.Label(master=self.page_2, text='Shape', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_2_LB3.place(x=417,y=30)

        self.page_2_LB4 = tk.Label(master=self.page_2, text='Shape', font=('Inner', 10),bg='white',borderwidth=0)
        self.page_2_LB4.place(x=417,y=30)

        self.delay = 1
        self.update()



        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.pages = [self.page_1, self.page_2]
        self.count = 0
        #-------------BOTTOM FRAME FOR CHANGE THE SCREEN----------------------
        self.bottom_frame = tk.Frame(self.root)
        # back button UI,  command is using the move back page function
        self.back_btn = tk.Button(self.bottom_frame, text='Back',
                            font=('Bold', 12),
                            bg='#1877f2', fg='white', width=8,
                            command=self.move_back_page)
        self.back_btn.pack(side=tk.LEFT, padx=10)
        # next button UI, command is using the move next page function
        self.next_btn = tk.Button(self.bottom_frame, text='Next',
                            font=('Bold', 12),
                            bg='#1877f2', fg='white', width=8,
                            command=self.move_next_page)
        self.next_btn.pack(side=tk.RIGHT, padx=10)

        self.bottom_frame.pack(side=tk.BOTTOM, pady=10)
    def move_next_page(self):

        if not( self.count > (len(self.pages) - 2)):
            for p in self.pages:
                p.pack_forget()

        self.count += 1
        # counts how many pages
        page = self.pages[self.count]
        page.pack()
    def move_back_page(self):

        if not (self.count == 0):
            for p in self.pages:
                p.pack_forget()

        self.count -= 1
        # counts how many pages
        page = self.pages[self.count]
        page.pack()
    def update(self):
        ret, self.frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.frame).resize((355,200),Image.LANCZOS))
            self.page_2_canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.page_2.after(self.delay, self.update)

    def capture(self):
        if self.maxImage == 0:
            messagebox.showinfo('attention', 'Please chose the shape first')
            return
        if len(self.listImage)<self.maxImage:
            try:
                img = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
                extracter = ImageExtract.ImageExtracter(self.frame)
                _ = extracter.FindLabelRegion()
                extracter.ProcessingImage()
                self.listImage.append(extracter.ProcessedOutput)
                # self.listImage.append(img)
                print(len(self.listImage))
            except Exception as e:
                messagebox.showinfo('attention', f"{e}")
        
        else:
            messagebox.showinfo('attention', 'You reach the maxium number of captured images')
        
    def getNewData(self):
        self.page_2_IMG1.config(image=self.gray)
        self.page_2_IMG1.image = self.gray
        self.page_2_IMG1.config(image=self.gray)
        self.page_2_IMG1.image = self.gray        
        self.listImage = []

    def extracting(self):
        # print('order working')
        # if (self.start_Working == False):
        #     self.thread.join()
        # self.start_Working = True
        # print(self.start_Working) 
        # self.thread.start()


        print('i am fucking working')
        self.on_Working = True
        print(self.start_Working)
        try:
            listname = ['imgCy_1.jpg','imgCy_2.jpg','imgCy_3.jpg','imgCy_4.jpg','imgCy_5.jpg','imgCy_6.jpg','imgCy_7.jpg','imgCy_8.jpg','imgCy_9.jpg']
            # listname = ['imgRec_1.jpg','imgRec_2.jpg','imgRec_3.jpg','imgRec_4.jpg']
            imgs = []
            for name in listname:
                img = cv2.imread(name)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                extracter = ImageExtract.ImageExtracter(img)
                _ = extracter.FindLabelRegion()
                extracter.ProcessingImage()
                self.listImage.append(extracter.ProcessedOutput)
            rawImage = ImageExtract.ConnectData(self.listImage,self.isCylinder)
            Text_extracter = ImageExtract.OCR_extracter(rawImage)
            mask = Text_extracter.TextRecognition()
            Text_extracter.ExtractToCSVfile()
            extractedImage = Text_extracter.TextDetection()
            
            rawphoto = ImageTk.PhotoImage(image = Image.fromarray(rawImage).resize((400,100),Image.LANCZOS))
            self.page_2_IMG1.config(image=rawphoto)
            self.page_2_IMG1.image = rawphoto 
            
            extractedphoto = ImageTk.PhotoImage(image = Image.fromarray(extractedImage).resize((400,100),Image.LANCZOS))
            self.page_2_IMG2.config(image=extractedphoto)
            self.page_2_IMG2.image = extractedphoto 
        except Exception as e: 
            messagebox.showinfo('attention', f"{e}")


    def cylinder(self):
        self.isCylinder = True
        self.maxImage = 10
        print("is Cylinder:",self.isCylinder)
        print("maxImg:",self.maxImage)

    def box(self):
        self.isCylinder = False
        self.maxImage = 4
        print("is Cylinder:",self.isCylinder)
        print("maxImg:",self.maxImage)

root = tk.Tk()
root.geometry('500x700')
root.title('Tkinter Hub')
MY_GUI(root)
root.mainloop()
