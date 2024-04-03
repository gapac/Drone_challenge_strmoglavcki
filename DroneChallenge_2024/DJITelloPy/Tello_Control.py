from djitellopy import Tello
import cv2, math, time
import threading
import datetime
import os
from os import path
from cv2 import aruco
#import cv2.aruco as aruco
import matplotlib as mpl
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
from threading import Thread, Event
import matplotlib.pyplot as plt
import PID 

GorDol_save = []
NaprejNazaj_save = []
Yaw_save = []
LevoDesno_save = []

class TelloC:
    def __init__(self):  
        self.root = tki.Tk()
        self.root.title("TELLO Controller")

        # Create a label to display the video stream
        self.image_label = tki.Label(self.root)
        self.image_label.pack()

        self.arucoId = 2

        self.tello = Tello()
        self.error_sum = 0
        self.prev_error = 0

        
        self.frame = None  # frame read from h264decoder and used for pose recognition 
        self.frameCopy = None
        self.frameProc = None
        self.thread = None # thread of the Tkinter mainloop
        self.stopEvent = None 
        self.lastPrintTime = 0 
        
        # control variables
        self.distance = 0.2  # default distance for 'move' cmd
        self.degree = 30  # default degree for 'cw' or 'ccw' cmd

        self.waitSec = 0.1
        self.state = 1
        self.oldTime = 0
        self.TR = None
        self.Tvec = None
        self.Rvec = None

        fname = 'calib.txt'
        self.cameraMatrix = None
        self.distCoeffs = None
        self.numIter = 1
        self.specific_point_aruco_1 = np.array([[0], [0.2], [1]]) 
        self.specific_point_aruco_2 = np.array([[0], [0.2], [-0.5]]) 
        self.Step_1 = True
        self.prev_T1_filtered = None
        self.prev_T2_filtered = None
        self.prev_yaw_filtered = None
        #add T11-T24
        self.prev_T11 = None
        self.prev_T12 = None
        self.prev_T13 = None
        self.prev_T14 = None

        self.prev_T21 = None
        self.prev_T22 = None
        self.prev_T23 = None
        self.prev_T24 = None

        self.prev_yaw_1= None
        self.prev_yaw_2= None
        self.prev_yaw_3= None
        self.prev_yaw_4= None


        self.last_call_T1_filtered = None

        self.controlEnabled = True
        self.takeoffEnabled = True
        self.landEnabled = False

        self.cur_fps = 0 
        self.frame_count = 0
        self.last_fps_calculation = time.time()

        if path.exists(fname):
            self.cameraMatrix = np.loadtxt(fname, max_rows=3, dtype='float', unpack=False) 
            self.distCoeffs = np.loadtxt(fname, max_rows=1,skiprows=3, dtype='float', unpack=False)

        self.tello.connect()
        self.lock = threading.RLock()
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

        # Persistent thread control
        self.controlEvent = Event()
        self.controlArgs = None

        # Start the persistent control thread
        self.controlThread = Thread(target=self.persistentControlLoop)
        self.controlThread.daemon = True
        self.controlThread.start()

        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.mainloop() 

    def persistentControlLoop(self):
        while not self.stopEvent.is_set():
            # Wait for the signal to control
            self.controlEvent.wait()
            # Reset the event to wait for the next signal
            self.controlEvent.clear()
            
            # Safely retrieve arguments for controlAll
            with self.lock:
                if self.controlArgs:
                    T1, T2, yaw = self.controlArgs
                    if self.controlEnabled:
                        self.controlAll(T1, T2, yaw)
                    else:
                        time.sleep(0.01)
                    self.controlArgs = None  # Reset arguments
                else:
                    time.sleep(0.01)

    def startControlAllThread(self, T1, T2, yaw):
        # Set the arguments for controlAll
        with self.lock:
            self.controlArgs = (T1, T2, yaw)
        # Signal the persistent thread to run controlAll
        self.controlEvent.set()

    def on_key_press(self,event):
        key = event.keysym 
        print("Key: ",key)
        if key == 'w':
            self.tello.move_forward(30)
        elif key == 's':
            self.tello.move_back(30)        
        elif key == 'a':
            self.tello.move_left(30)
        elif key == 'd':
            self.tello.move_right(30)
        elif key == 'e':
            self.tello.rotate_clockwise(30)
        elif key == 'q':
            self.tello.rotate_counter_clockwise(30)
        elif key == 'r':
            self.tello.move_up(30)
        elif key == 'f':
            self.tello.move_down('f')
        elif key == '1':
            self.tello.flip('f')
            #self.tello.move_down(30)
        elif key == 'o':
            self.tello.takeoff()
        elif key == 'x':
            self.tello.emergency()
        elif key == 'p':
            self.tello.land()
            self.tello.end
            # Plotting GorDol
            plt.subplot(4, 1, 1)
            plt.plot(GorDol_save)
            plt.title('GorDol')
            plt.xlabel('Time')
            plt.ylabel('Value')

            # Plotting NaprejNazaj
            plt.subplot(4, 1, 2)
            plt.plot(NaprejNazaj_save)
            plt.title('NaprejNazaj')
            plt.xlabel('Time')
            plt.ylabel('Value')

            # Plotting Levodesno
            plt.subplot(4, 1, 3)
            plt.plot(LevoDesno_save)
            plt.title('Levodesno')
            plt.xlabel('Time')
            plt.ylabel('Value')

            # Plotting Jo
            plt.subplot(4, 1, 4)
            plt.plot(Yaw_save)
            plt.title('Jo')
            plt.xlabel('Time')
            plt.ylabel('Value')

            #plt.tight_layout()
            plt.show()
            
        

    def videoLoop(self):
        try:
            self.tello.streamoff()
            self.tello.streamon()

            self.frame_read = self.tello.get_frame_read()
            time.sleep(0.5)  # Give some time for the stream to start
            
            # Variables to control the FPS
            fps_limit = 20
            time_per_frame = 1.0 / fps_limit
            last_time = time.time()
            start_time = last_time

            while not self.stopEvent.is_set():   
                current_time = time.time()
                elapsed_time = current_time - last_time             
                if elapsed_time > time_per_frame:
                    self.frame = self.frame_read.frame
                    if self.frame is not None:

                        self.frameCopy = self.frame.copy()
                        self.frameProc = self.frame.copy()
                        
                        self.Rvec, self.Tvec = self.detectAruco(self.arucoId)

                        if self.Rvec is not None and self.Tvec is not None:
                            T1, T2, yaw = self.transformArucoToDroneCoordinates(self.Rvec, self.Tvec)
                            if T1 is not None and T2 is not None and yaw is not None:
                                self.startControlAllThread(T1, T2, yaw)
                            else:
                                self.startControlAllThread(None, None, None)
                        else:
                            self.startControlAllThread(None, None, None)

                        pil_image = Image.fromarray(self.frameCopy) 
                        tk_image = ImageTk.PhotoImage(image=pil_image) 
                        self.image_label.configure(image=tk_image, width=960, height=720)
                        self.image_label.image = tk_image  

                        if self.numIter < 100:
                            self.numIter = self.numIter + 1
                        else:
                            end_time = time.time()
                            total_time = end_time - start_time
                            timePerIt = total_time / 100.0
                            print("Time per iteration: ",timePerIt)
                            self.numIter = 1
                            start_time = time.time()
                    last_time = current_time
                else:
                    time_to_sleep = time_per_frame - elapsed_time
                    time.sleep(time_to_sleep)                                                    
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError",e)


    def transformArucoToDroneCoordinates(self, rvec, tvec):
        if rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R_j, _ = cv2.Rodrigues(rvec)
            # Define the camera to drone rotation matrix
            R_cam_drone = np.array([[0.0000, -0.2079, 0.9781],
                                    [-1.0000, 0.0000, 0.0000],
                                    [0, -0.9781, -0.2079]])
            
            # Transform the point in Aruco coordinate system to drone coordinate system
            T_transformed = np.dot(R_cam_drone, tvec.reshape(3, 1)).flatten()
            R_aruco_to_drone = np.dot(R_cam_drone, R_j)

            # Calculate roll
            roll = np.arctan2(-R_aruco_to_drone[2, 0], np.sqrt(R_aruco_to_drone[2, 1]**2 + R_aruco_to_drone[2, 2]**2))
            roll_degrees = np.degrees(roll)
            #print("Roll:", roll_degrees)

            # Calculate pitch
            pitch = np.arctan2(R_aruco_to_drone[2, 1], R_aruco_to_drone[2, 2])
            pitch_degrees = np.degrees(pitch)
            #print("Pitch:", pitch_degrees)
            
            # Calculate yaw from rotation matrix, assuming R_j is the rotation matrix from Aruco to camera            
            yaw = np.arctan2(R_aruco_to_drone[1, 0], R_aruco_to_drone[0, 0]) + np.pi/2
            yaw_degrees = np.degrees(yaw)

            # Transform the specific points in Aruco coordinate system to drone coordinate system
            # Apply rotation
            specific_point_transformed_1 = np.dot(R_aruco_to_drone, self.specific_point_aruco_1)
            specific_point_transformed_2 = np.dot(R_aruco_to_drone, self.specific_point_aruco_2)
            # Apply translation
            specific_point_transformed_1 += T_transformed.reshape(3, 1)
            specific_point_transformed_2 += T_transformed.reshape(3, 1)

            return specific_point_transformed_1.flatten(), specific_point_transformed_2.flatten(), yaw_degrees

        return None, None


    def detectAruco(self, arucoId):
        self.gray = cv2.cvtColor(self.frameProc, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        parameters =  aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(self.gray) 
        frame_markers = aruco.drawDetectedMarkers(self.frameCopy, corners, ids)

        if ids is not None:
            for i in range(len(ids)):
                if ids[i] == arucoId:
                    c = corners[i]
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c, 0.10, self.cameraMatrix, self.distCoeffs)
                    
                    if rvec is not None and tvec is not None:
                        cv2.drawFrameAxes(self.frameCopy, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.20)  
                        """
                        # Convert to Euler
                        R_mat = np.matrix(cv2.Rodrigues(rvec)[0])
                        roll, pitch, yaw = self.rotationMatrixToEulerAngles(R_mat)
                        rollD = math.degrees(roll)
                        pitchD = math.degrees(pitch) 
                        yawD = math.degrees(yaw)
                           
                        rot = np.array([rollD, pitchD, yawD])
                        print("RPY: ",rot)
                        #return np.array([tvec[0][0], rot])"""
                        return rvec, tvec[0][0]
        return None, None

    def rotationMatrixToEulerAngles(self,R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    
    def controlAll(self, T1, T2, yaw):
        self.controlEnabled = False
        currTime = time.time()
        
        if self.takeoffEnabled:
            self.takeoffEnabled = False
            self.tello.takeoff()

        # Increment the frame count
        self.frame_count += 1
        # Calculate and print FPS every 2 seconds
        if time.time() - self.last_fps_calculation >= 2:
            self.cur_fps = self.frame_count / (time.time() - self.last_fps_calculation)
            #print(f"FPS: {self.cur_fps:.2f}")

            # Reset the frame count and last FPS calculation time
            self.frame_count = 0
            self.last_fps_calculation = time.time()    

            #print('T1: ', T1)
            #print('T2: ', T2)
            #print('yaw: ', yaw)
            #self.tello.is_flying and

        if T1 is not None and T2 is not None and yaw is not None and self.cur_fps > 10:
            print(f"FPS: {self.cur_fps:.2f}")
            #self.on_key_press('w' or 'a' or 's' or 'd' or 'e' or 'q' or 'r' or 'f' or 'o' or 'p')
            if self.prev_T1_filtered is None:
                self.prev_T1_filtered = T1
            if self.prev_T2_filtered is None:
                self.prev_T2_filtered = T2
            if self.prev_yaw_filtered is None:
                self.prev_yaw_filtered = yaw
            if self.prev_T11 is None:   
                self.prev_T11 = T1
            if self.prev_T12 is None:
                self.prev_T12 = T1
            if self.prev_T13 is None:
                self.prev_T13 = T1
            if self.prev_T14 is None:
                self.prev_T14 = T1
            if self.prev_T21 is None:
                self.prev_T21 = T2
            if self.prev_T22 is None:
                self.prev_T22 = T2
            if self.prev_T23 is None:
                self.prev_T23 = T2
            if self.prev_T24 is None:
                self.prev_T24 = T2
            if self.prev_yaw_1 is None:
                self.prev_yaw_1 = yaw
            if self.prev_yaw_2 is None:
                self.prev_yaw_2 = yaw
            if self.prev_yaw_3 is None:
                self.prev_yaw_3 = yaw
            if self.prev_yaw_4 is None:
                self.prev_yaw_4 = yaw

            

            # Apply low pass filter
            # TODO naret bolsi filter? - mogoc povprecit T1,T2?
            # moveing avarage filter with 5 trailing smaples
            #print("T1: ",T1)
            #print("T2: ", T2)
            #if ((T1.any() == None) or (T2.any() == None) or (yaw == None)):
             #   T1, T2 = [0, 0, 0]
             #   yaw = 0
            

            T1_filtered = 0.5*T1 + 0.3*self.prev_T11 + 0.15*self.prev_T12 + 0.045*self.prev_T13 + 0.005*self.prev_T14
            T2_filtered = 0.2*T2 + 0.2*self.prev_T21 + 0.2*self.prev_T22 + 0.2*self.prev_T23 + 0.2*self.prev_T24
            yaw_filtered = 0.2*yaw + 0.2*self.prev_yaw_1 + 0.2*self.prev_yaw_2 + 0.2*self.prev_yaw_3 + 0.2*self.prev_yaw_4
            
            #yaw_filtered = 0.2*yaw + 0.8*self.prev_yaw_filtered
            #T1_filtered = 0.1 * T1 + 0.9 * self.prev_T1_filtered
            #T2_filtered = 0.1 * T2 + 0.9 * self.prev_T2_filtered
            #aw_filtered = 0.1 * yaw + 0.9 * self.prev_yaw_filtered

            # Update the previous filtered values for the next call
            self.prev_T1_filtered = T1_filtered
            self.prev_T2_filtered = T2_filtered
            self.prev_yaw_filtered = yaw_filtered

            self.prev_T11 = T1
            self.prev_T12 = self.prev_T11
            self.prev_T13 = self.prev_T12
            self.prev_T14 = self.prev_T13
            

            self.prev_T21 = T2
            self.prev_T22 = self.prev_T21
            self.prev_T23 = self.prev_T22
            self.prev_T24 = self.prev_T23

            self.prev_yaw_1 = yaw
            self.prev_yaw_2 = self.prev_yaw_1
            self.prev_yaw_3 = self.prev_yaw_2
            self.prev_yaw_4 = self.prev_yaw_4

            oddaljenostOdTarce = 1.25#1.25
            #Best regulator ever
            if (currTime - self.oldTime) > 0.0333:
                freq = 1/(currTime - self.oldTime)
                #print("frek: ",freq)

                # if(T2_filtered==None):
                #     self.tello.move_back(21)

                if(T2_filtered is not None and T2_filtered[0] > oddaljenostOdTarce and self.arucoId!=0):
                    
                    print("T2...: ",T2_filtered)
                    pidx = PID.PIDRegulator(20, 0.3, 0.3)
                    pidy = PID.PIDRegulator(20, 0.3, 0.3)    #ta del je treba naret sam enktrat, ce ne nedela I, D!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    pidz = PID.PIDRegulator(20, 0.3, 0.3)
                    pidyaw = PID.PIDRegulator(0.5, 0.1, 0.1)

                    NaprejNazaj = pidx.calculate(-1, -T2_filtered[0])
                    LevoDesno = pidy.calculate(0.2, T2_filtered[1])
                    GorDol = pidz.calculate(0.2, -T2_filtered[2])
                    jo = pidyaw.calculate(0, yaw_filtered)

                    # NaprejNazaj = (T2_filtered[0]-0.6)*10 +  ##-oddaljenostOdTarce
                    
                    # #print("NaprejNazaj: ", NaprejNazaj)
                    # if NaprejNazaj > 20:
                    #     NaprejNazaj = 20
                    # #print("GorDol: ", NaprejNazaj)

                    # LevoDesno = (-T2_filtered[1])*70 #spodaj v funkciji 0 #+0.2
                    # #print("LevoDesno: ", LevoDesno)

                    # GorDol = (T2_filtered[2]+0.2)*20 #+0.55
                    # #if GorDol > 25:
                    # #    GorDol = 25
                    # #print("GorDol: ", GorDol)

                    # #print("Yaw: ",yaw_filtered)
                    # #print("T2_filtered: ",T2_filtered)
                    # jo = -yaw_filtered*0.5

                    GorDol_save.append(GorDol)
                    NaprejNazaj_save.append(NaprejNazaj)
                    Yaw_save.append(yaw_filtered)
                    LevoDesno_save.append(LevoDesno)#LevoDesno

                    self.tello.send_rc_control(int(LevoDesno), int(NaprejNazaj), int(GorDol), int(jo))



                elif(T2_filtered is not None and T2_filtered[0] <  oddaljenostOdTarce and T2_filtered[1] < 0.09 and T2_filtered[2] < 0.09 and yaw_filtered < 3 and self.arucoId!=0 ):#TODOlahk dodas pogoje da gre naprej samo ko je cist poravnan
                    self.tello.send_rc_control(int(0), int(0), int(10), int(0))
                    self.prev_T11 = None
                    self.prev_T12 = None
                    self.prev_T13 = None
                    self.prev_T14 = None

                    self.prev_T21 = None
                    self.prev_T22 = None
                    self.prev_T23 = None
                    self.prev_T24 = None
                    #for i in  range(20):
                    #time.sleep(1)
                    self.tello.move_forward(150)
                    #self.tello.land()
                    self.arucoId = self.arucoId + 1 ###pazi plus 2
                    print("Menjava znacke - Aruco ID: ",self.arucoId)

                    if(self.arucoId == 3):
                        self.tello.flip_right()
                        self.tello.rotate_clockwise(180)##TODO odvisno na kiri progi si spremeni smer da ne zaznas nasprotnikove tarce
                        self.tello.move_left(150)
                    if(self.arucoId == 5):
                        #self.tello.move_forward(25)
                        self.tello.flip_forward()
                        self.tello.move_right(100)##TODO odvisno na kiri progi si spremeni smer da ne zaznas nasprotnikove tarce
                        self.arucoId = 0



                elif(T2_filtered is not None and self.arucoId==0):
                    print("T2...: ",T2_filtered)
                    NaprejNazaj = (T2_filtered[0])*10 
                    if NaprejNazaj > 15:
                        NaprejNazaj = 15

                    LevoDesno = (-T2_filtered[1]+0.2)*70 #spodaj v funkciji 0

                    GorDol = (T2_filtered[2])*20

                    jo = -yaw_filtered*0.5

                    GorDol_save.append(GorDol)
                    NaprejNazaj_save.append(NaprejNazaj)
                    Yaw_save.append(yaw_filtered)
                    LevoDesno_save.append(LevoDesno)#LevoDesno

                    self.tello.send_rc_control(int(LevoDesno), int(NaprejNazaj), int(GorDol), int(jo))

                    if abs(LevoDesno)<0.3 and abs(NaprejNazaj)<0.3:
                        self.tello.land()
                    


            self.oldTime = currTime

            self.last_call_T1_filtered = T1_filtered
        self.controlEnabled = True


    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()
    

