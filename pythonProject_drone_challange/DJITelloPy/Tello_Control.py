from djitellopy import Tello
import cv2, math, time
import threading
import datetime
import os
from os import path
from cv2 import aruco
import matplotlib as mpl
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
from threading import Thread, Event

class TelloC:
    def __init__(self):  
        self.root = tki.Tk()
        self.root.title("TELLO Controller")

        # Create a label to display the video stream
        self.image_label = tki.Label(self.root)
        self.image_label.pack()

        self.arucoId = 1     
        self.tello = Tello()
        
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
        self.last_call_T1_filtered = None

        self.controlEnabled = True
        self.takeoffEnabled = True
        self.landEnabled = True

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
            self.tello.move_down(30)
        elif key == 'o':
            self.tello.takeoff()
        elif key == 'p':
            self.tello.land()
            self.tello.end
        

    def videoLoop(self):
        try:
            self.tello.streamoff()
            self.tello.streamon()

            self.frame_read = self.tello.get_frame_read()
            time.sleep(0.5)  # Give some time for the stream to start
            
            # Variables to control the FPS
            fps_limit = 30
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
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(c, 0.10, self.cameraMatrix, self.distCoeffs)
                    
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
            print(f"FPS: {self.cur_fps:.2f}")

            # Reset the frame count and last FPS calculation time
            self.frame_count = 0
            self.last_fps_calculation = time.time()    

        if self.tello.is_flying and T1 is not None and T2 is not None and yaw is not None and self.cur_fps > 10:
            if (currTime - self.oldTime) > self.waitSec:
                if self.state==1:
                    self.heightC(T1,yaw)
                elif self.state==2:                  
                    self.distC(T1,yaw,0.5)
                elif self.state==3:               
                    self.lefrigC(T1,yaw)         
                elif self.state==4:   
                    self.yawC(T1,yaw)
                else:
                    self.tello.send_rc_control(0,0,0,0)
                    self.oldTime = currTime
            else:
                time.sleep(self.waitSec)
                self.tello.send_rc_control(0,0,0,0)
                self.oldTime = currTime

            if self.prev_T1_filtered is None:
                self.prev_T1_filtered = T1
            if self.prev_T2_filtered is None:
                self.prev_T2_filtered = T2
            
            # Apply low pass filter
            T1_filtered = 0.1 * T1 + 0.9 * self.prev_T1_filtered
            T2_filtered = 0.1 * T2 + 0.9 * self.prev_T2_filtered

            # Update the previous filtered values for the next call
            self.prev_T1_filtered = T1_filtered
            self.prev_T2_filtered = T2_filtered

            # Check if T1_filtered is less than 2 cm away from its last filtered value
            if self.last_call_T1_filtered is not None:
                distance = np.linalg.norm(T1_filtered - self.last_call_T1_filtered)
                if distance < 0.01:  # Less than 1 cm
                    if self.Step_1 and self.state==0:
                        self.Step_1 = False
                        T1_f_cm = [int(x * 100) for x in T1_filtered]
                        T2_f_cm = [int(x * 100) for x in T2_filtered]
                        print("T1_f_cm",T1_f_cm)
                        print("T2_f_cm",T2_f_cm)
                        self.tello.curve_xyz_speed(T1_f_cm[0], T1_f_cm[1], T1_f_cm[2], T2_f_cm[0], T2_f_cm[1], T2_f_cm[2], 10)

            # Update the last call filtered value of T1 for the next comparison
            self.last_call_T1_filtered = T1_filtered
        self.controlEnabled = True

    def heightC(self,T,yaw):
        currTime = time.time()
        error = T[2]
        premik = abs(error)
        if premik > 0.6:
            premik = 0.6

        # Premik gor
        if error >= 0.2:      
            resp1 = self.tello.move_up(int(premik*100))  
            if resp1:
                print('OK!')
                self.state = 2
            else:
                print('Ni poslano')
                time.sleep(0.5)
                self.state = 1
            self.oldTime = currTime

        # Premik dol
        elif error <= -0.2: 
            resp2 = self.tello.move_down(int(premik*100))
            if resp2:
                print('OK!')
                self.state = 2
            else:
                print('Ni poslano')
                time.sleep(0.5)
                self.state = 1
            self.oldTime = currTime
        else:
            self.state = 2

    def distC(self,T,yaw,ref):
        currTime = time.time()
        error = (ref-T[0])

        premik = abs(error)
        if premik > 0.4:
            premik = 0.4

        # Premik nazaj
        if error >= 0.2:  
            print("Premik nazaj")
            resp1 = self.tello.move_back(int(premik*100))  
            if resp1:
                print('OK!')
                self.state = 3
            else:
                print('Ni poslano')
                time.sleep(0.5)
                self.state = 2
            self.oldTime = currTime

        # Premik naprej
        elif error <= -0.2: 
            print("Premik naprej")
            resp2 = self.tello.move_forward(int(premik*100))
            if resp2:
                print('OK!')
                self.state = 3
            else:
                print('Ni poslano')
                time.sleep(0.5)
                self.state = 2
            self.oldTime = currTime
        else:
            self.state = 3

    def lefrigC(self,T,yaw):
        currTime = time.time()
        error = T[1]
        premik = abs(error)
        if premik > 0.4:
            premik = 0.4

        # Premik desno
        if error <= -0.2:   
            resp1 = self.tello.move_right(int(premik*100))  
            if resp1:
                print('OK!')
                self.state = 4
            else:
                print('Ni poslano')
                time.sleep(0.5)
                self.state = 3
            self.oldTime = currTime

        # Premik levo
        elif error >= 0.2: 
            resp2 = self.tello.move_left(int(premik*100))
            if resp2:
                print('OK!')
                self.state = 4
            else:
                print('Ni poslano') 
                time.sleep(0.5)
                self.state = 3
            self.oldTime = currTime
        else:
            self.state = 4

    def yawC(self,T,yaw):
        currTime = time.time()
        error = yaw
        zasuk = abs(error)
        if zasuk > 20 and abs(T[1]) < 0.3:
            zasuk = 20
        elif abs(T[1]) > 0.3:
            zasuk = 0
            self.state = 3
            self.oldTime = currTime

        if zasuk > 0:
            # Zasuk v  smeri urinega kazalca 
            if error < -3:     
                resp1 = self.tello.rotate_clockwise(int(round(zasuk)))
                if resp1:
                    print('OK!')
                    self.state = 0 #1 !!!
                else:
                    print('Ni poslano')
                    time.sleep(0.5)
                    self.state = 4
                self.oldTime = currTime

            # Zasuk v nasprotni smeri urinega kazalca 
            elif error > 3: 
                resp2 = self.tello.rotate_counter_clockwise(int(round(zasuk)))
                if resp2:
                    print('OK!')
                    self.state = 0 #1 !!!
                else:
                    print('Ni poslano')
                    time.sleep(0.5)
                    self.state = 4
                self.oldTime = currTime
            else:
                self.state = 0 #1 !!!

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()

