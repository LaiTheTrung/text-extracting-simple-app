import tkinter
import cv2
import time
import PIL.Image, PIL.ImageTk
import tkinter.messagebox
import threading
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

class App:
    def __init__(self, window, window_title,video_source = 0, fpsLimit = 30):

        self.window = window
        self.window.title(window_title)

        self.video_source = video_source
        self.fpsLimit = fpsLimit

        self.vid = MyVideoCapture(video_source)    
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.btn_getvideo=tkinter.Button(window, text="getvideo", width=50, command=threading.Thread(target=self.getvideo).start)
        self.btn_getvideo.pack(anchor=tkinter.CENTER, expand=True)
        self.delay = 1
        self.update()

        self.window.mainloop()
    def getvideo(self):
        start_time = time.time()
        out = cv2.VideoWriter('output_'+time.strftime("%d-%m-%Y-%H-%M-%S")+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

        while int(time.time()-start_time) < self.fpsLimit:           
            ret, frame = self.vid.get_frame()
            if ret:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                break
        out.release()
        # tkinter.messagebox.showinfo(title="Notification", message="save video successful")
        print("success")
        cv2.destroyAllWindows()


    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)
    

App(tkinter.Tk(), "get_video",0,10)