#!/usr/bin/env python
'''
GUI for SMIRKs
'''
import tkFileDialog
import Tkinter
import tkMessageBox
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage
from SMIRKs_head import *
import time # for test purpose
import pysynth_b as ps 
import pyaudio  
import wave  

class Gui(object):
    def __init__(self):
        self.root = Tkinter.Tk() # Create a GUI window
        self.root.title('SMIRKs')
        self.label = Tkinter.Label(self.root,text='Sheet Music Intelligent Reader & Keyboard Simulator')
        self.input = Tkinter.Entry(self.root,width=30) # Sheet Music Input Box
        self.input_path = '' # picture path
        self.input_button =  Tkinter.Button(self.root,command = lambda:self.askinputfile(),text='Sheet Music')
        self.destination = Tkinter.Entry(self.root,width=30) # WAV Output Path
        self.destination_path = '' # output path
        self.destination_button = Tkinter.Button(self.root,command = lambda:self.askoutputpath(),text='Save To')
        self.infobox = Tkinter.Listbox(self.root,width=30) # Information Box
        self.play_button = Tkinter.Button(self.root,command = lambda:self.missionstart(),text='Play!')

    def gui_arrange(self):
        self.label.grid(row=0)
        self.input.grid(row=1,column=0)
        self.input_button.grid(row=1,column=1)
        self.destination.grid(row=2,column=0)
        self.destination_button.grid(row=2,column=1)
        self.infobox.grid(row=3,column=0)
        self.play_button.grid(row=3,column=1)


    def askinputfile(self):
        self.input.delete('0',Tkinter.END)
        #self.input_path = tkFileDialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
        self.input_path = tkFileDialog.askopenfilename()
        self.input.insert(Tkinter.END,self.input_path)
    def askoutputpath(self):
        self.destination.delete('0',Tkinter.END)
        #self.destination_path = tkFileDialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.destination_path = tkFileDialog.askdirectory()
        self.destination.insert(Tkinter.END,self.destination_path)

    '''
    Main Function
    '''
    def missionstart(self):
        self.infobox.delete('0',Tkinter.END) # Clear the listbox 
        print('Start Converting!.....')
        self.infobox.insert(Tkinter.END,'Start Converting!....')
        self.infobox.update_idletasks()

        self.setup()  # setup, image reading
        self.infobox.insert(Tkinter.END,'Image Get!')
        self.infobox.update_idletasks()

        self.partition()
        self.infobox.insert(Tkinter.END,'Recognized!')
        self.infobox.update_idletasks()
        
        self.infobox.insert(Tkinter.END,'Writing to file...')
        self.infobox.update_idletasks()
        self.conversion() # convert to wave file

        self.infobox.insert(Tkinter.END,'Streaming...')
        self.infobox.update_idletasks()
        self.streaming() # streaming
    

    def setup(self):
        print('Setup!')
        print(self.input_path)
        img = cv2.imread(self.input_path) # use CV2 to read the image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, self.img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.img_bw = convert(self.img_bw)
        self.height, self.width = self.img_bw.shape
        

    
    def partition(self):
        print('Recognizing!')
        dash_filter = np.ones([1, 2])  
        staff_line_filter = np.ones([1, self.width//10])

        img_staff_lines, idx_staff = get_staff_lines(self.img_bw, dash_filter, staff_line_filter)

        #cv2.imshow('test',img_staff_lines)
        num_staff = len(idx_staff) // 5
        diff_staff = (idx_staff[4] - idx_staff[0])  // 4

        img_div_set = []
        for i in range(num_staff):
            idx_start = idx_staff[5*i] - 2 * diff_staff
            idx_end = idx_staff[5*i+4] + 2 * diff_staff
            if idx_start < 0:
                idx_start = 0
            if idx_end > self.height:
                idx_end = self.height
            
            img_div = self.img_bw[idx_start:idx_end, :]
            img_div_set.append(img_div)        

        for it in range(len(img_div_set)):
            img_div = img_div_set[it]
            hei_div, wid_div = img_div.shape
            
            hist_col = [0] * wid_div    
            for j in range(wid_div):
                hist_col[j] = np.sum(img_div[:, j] / 255.0)  
            
            # Get the most common element in hist_col
            median = get_most_common(hist_col)
            
            # Find the right and left edge of the staff
            # Right edge
            for i in range(wid_div):    
                
                if hist_col[i] == 0: 
                    if hist_col[i+2] == median: 
                            
                        right_start = i
                        right_end = i+2
                        break
                    
                    if hist_col[i+3] == median: 
                        
                        right_start = i
                        right_end = i+3
                        break
            
            # Left edge
            for i in range(wid_div//2, wid_div):
                if hist_col[i] == 0:
                    left_end = i - 1
                    break
                
            for i in range(wid_div//2, left_end)[::-1]:
                if hist_col[i] == median:
                    left_start = i
                    break
            
            img_tmp = img_div[:, right_end : left_start]
            img_div_set[it] = img_div[:, right_end : left_start]
  

        output_clef = [] # The length of the output should be same as len(img_division_set)
        

        for it in range(len(img_div_set)):
            img_div = img_div_set[it]
            hei_div, wid_div = img_div.shape
            
            div_staff_lines, idx_staff_set = get_staff_lines(img_div, dash_filter, staff_line_filter)
            
            hist_col = [0] * wid_div    
            for j in range(wid_div):
                hist_col[j] = np.sum(img_div[:, j] / 255.0)  
            
            # Get median from hist_col
            median = get_most_common(hist_col)
            
            # Find the right and left edge of the clef
            clef_start, clef_end = determine_edge(hist_col, median)
            
            staff_5 = idx_staff_set[-1]
            staff_4 = idx_staff_set[-2]   
            area = (staff_5 - staff_4) * (clef_end - clef_start)   
            
            n_pixels = 0
            for i in range(staff_4, staff_5):
                for j in range(clef_start, clef_end):
                    if img_div[i, j] == 255:
                        n_pixels += 1
                        
            density = 1.0 * n_pixels / area  
            if density > 0.25:
                type_clef = 1; # Treble       
            else:
                type_clef = 0; # Bass
                
            #output_clef.append(type_clef) 
            
            # Deal with the special situation when the type of clef is 'bass'
            if type_clef == 0:
                count = 0
                for i in range(clef_end, wid_div):
                    if hist_col[i] > median and count == 0:
                        count += 1
                        
                    if hist_col[i] <= median and count == 1:
                        clef_end = i
                        break
            
                        
            #img_tmp = img_div[:, clef_end : -1]           
            # After determine the type of clef, we no longer need them.
            img_div_set[it] = img_div[:, clef_end : -1]


        for it in range(0, 2):
            img_div = img_div_set[it]
            hei_div, wid_div = img_div.shape
            
            div_staff_lines, idx_staff_set = get_staff_lines(img_div, dash_filter, staff_line_filter)
            diff_staff = idx_staff_set[1] - idx_staff_set[0]
            
            horizontal_filter = np.ones([1, diff_staff])
            img_tmp = cv2.dilate(img_div, horizontal_filter, iterations = 1)

            hist_tmp = get_col_hist(img_tmp)
            hist_div = get_col_hist(img_div)     
            median = get_most_common(hist_div)
            key_start, key_end = determine_edge(hist_tmp, median)
     
            img_div_set[it] = img_div[:, key_end+1 : -1]


        output_note_pos = []
        output_note_type = []
        
        for it in range(len(img_div_set)):
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
            # Detect different types
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
            img_div = img_div_set[it]
            div_staff_lines, idx_staff = get_staff_lines(img_div, dash_filter, staff_line_filter)
            diff_staff = idx_staff[1] - idx_staff[0]
            
            # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
            # a) Detect the position of quarter and eighth notes
            # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
            
            # Quarter note
            disk_filter = generate_disk_filter(diff_staff//2.5)
            img_note1 = opening(img_div, disk_filter) 
            im1, contours1, hierarchy1 = cv2.findContours(img_note1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            moments1 = compute_moments(contours1)
            
            # Eighth note    
            disk_filter = generate_disk_filter(diff_staff//4)
            img_note3 = opening(img_div, disk_filter) 
            im3, contours3, hierarchy3 = cv2.findContours(img_note3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_area_set = []
            for i in range(len(contours3)):
                cnt = contours3[i]
                area = cv2.contourArea(cnt)
                contour_area_set.append(area)
                
            median = np.median(contour_area_set)
            std = np.std(contour_area_set)
            
            contour_dash = []
            canvas = np.zeros((img_note3.shape))
            for i in range(len(contours3)):
                if contour_area_set[i] > median + 1.3 * std:
                    contour_dash.append(contours3[i])           
            #cv2.drawContours(canvas, contour_dash, -1, 1)  


            moments_dash = compute_moments(contour_dash)
            
            # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
            # b) Detect the position of whole and half notes
            # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
            #img_note = remove_staff_lines(img_div, div_staff_lines, diff_staff)
            #img_note = cv2.dilate(img_note, np.ones((2,2)), iterations = 1)
            staff_line_filter = np.ones([1, 20])
            staff_lines, idx_staff = get_staff_lines(img_div, dash_filter, staff_line_filter)
            img_note = remove_staff_lines(img_div, staff_lines)
            img_note_fill = scipy.ndimage.binary_fill_holes(img_note).astype('uint8')
            img_note_fill[img_note_fill == 1] = 255
            img_note2 = img_note_fill - img_note
            
            disk1 = generate_disk_filter(diff_staff // 4)
            disk2 = generate_disk_filter(diff_staff // 2.5)
            tmp = cv2.erode(img_note2, disk1, iterations = 1)
            img_note_new = cv2.dilate(tmp, disk2, iterations = 1)
            
            im2, contours2, hierarchy2 = cv2.findContours(img_note_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
        # =============================================================================
        #     n_notes2 = len(contours2)   
        #     moments2 = np.empty((0, 2))
        #     for i in range(n_notes2):
        #         cnt = contours2[i]
        #         M = cv2.moments(cnt)
        #         col_ind = int(M['m10']/M['m00'])
        #         row_ind = int(M['m01']/M['m00'])  # We only care about its row index
        #         centroid = np.array([row_ind, col_ind])
        #         moments2 = np.vstack((moments2, centroid))
        # =============================================================================        
            moments2 = compute_moments(contours2)   
        
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
            # 1) Determine the position
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
            
            # Sort to make sure the notes are in order
            moments = np.vstack((moments1, moments2))  
            tmp_arg = np.argsort(moments[:, 1]) 
            moments = moments[tmp_arg]
            
            note_pos = []
            for i in range(len(moments)):
                row_idx = moments[i, 0]
                output = determine_note_pos(idx_staff, row_idx)
                note_pos.append(output)   
                
            output_note_pos.append(note_pos)
            
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
            # 2) Determine the type
            # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
            # Quarter and eighth note
            n_note1 = moments1.shape[0]
            note_type1 = np.column_stack((moments1[:, 1], [4] * n_note1))  
            for i in range(len(moments_dash)):
                for j in range(len(moments1)-1):
                    if moments1[j, 1] < moments_dash[i, 1] and moments1[j+1, 1] > moments_dash[i, 1]:
                        note_type1[j, 1] = 8
                        note_type1[j+1, 1] = 8
                        break
            # Half note
            n_note2 = moments2.shape[0]
            note_type2 = np.column_stack((moments2[:, 1], [2] * n_note2)) 
            
            # Combine and sort
            note_type = np.vstack((note_type1, note_type2))
            tmp_arg = np.argsort(note_type[:, 0]) 
            note_type = note_type[tmp_arg]
            
            # Add to output
            note_type = list(note_type[:, 1])   
            output_note_type.append(note_type)
        print(output_note_pos)
        print(output_note_type)
        self.output_note_pos = output_note_pos
        self.output_note_type = output_note_type
    
    def conversion(self):
        dict = {1:'c',2:'d',3:'e',4:'f',5:'g',6:'a',7:'b'}
        music = []
        for i in range(len(self.output_note_pos)):
            if self.output_note_pos[i] == []:
                continue
            for j in range(len(self.output_note_pos[i])):
                music.append((dict[self.output_note_pos[i][j]],int(self.output_note_type[i][j])))
        music = tuple(music)
        print(music)
        self.file_path = self.destination_path+'/music.wav'
        ps.make_wav(music, fn = self.file_path)

    def streaming(self):  
        chunk = 1024  #define stream chunk  
        f = wave.open(self.file_path,"rb") #open a wav format music  
        p = pyaudio.PyAudio()  #instantiate PyAudio  
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)  
        data = f.readframes(chunk)   
        while data != '':  
            stream.write(data)  
            data = f.readframes(chunk)  
        stream.stop_stream()  
        stream.close()    
        p.terminate()  


if __name__ == "__main__":
    GUI = Gui()
    GUI.gui_arrange()
    Tkinter.mainloop()
    pass