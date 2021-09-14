#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tkinter import *
import tkinter.ttk
import tkinter as tk
import tkinter.filedialog as filedialog  #to import the filedialog module
import tkinter.font
from PIL import ImageTk,Image  

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from scipy.special import erfinv; from scipy.stats.distributions import chi2; import GPy

import os
folder = os.getcwd()

#root= tk.Tk() 
root= tk.Toplevel()
root.title("PeakIt_1.45")
root.iconbitmap(folder+"\icons\peakIt_icon.ico")
#root.iconphoto(r'C:\Users\mikel\Desktop\peak_analyser_icon2.png')
#root.iconphoto(False, tk.PhotoImage(file='\Users\mikel\Desktop\peak_analyser_icon2.png'))


cp_row = 4

def data_process(x, y, cut_min = -11, cut_max = 11, x_shift = 0, y_shift = 0, x_scale = 1, y_scale = 1, bd = 0, interp = 0):
    x1 = x*x_scale + x_shift
    y1 = y*y_scale + y_shift

    y2 = y1[np.where((x1 > cut_min) & (x1 < cut_max))]
    x2 = x1[np.where((x1 > cut_min) & (x1 < cut_max))]

    x3 = x2; y3 = y2
    if bd > 0:    
        for i in range(bd):
            x3 = np.append(np.append(2*x3[0] - x3[1], x3), 2*x3[-1] - x3[-2])
            y3 = np.append(np.append(2*y3[0] - y3[1], y3), 2*y3[-1] - y3[-2])
    x4 = x3; y4 = y3
    if interp == 1:
        x_int = x3[0:-1] + np.diff(x3)/2        
        x4 = [None]*(len(x3)+len(x_int))
        x4[::2] = x3
        x4[1::2] = x_int
        
        y_int = y3[0:-1] + np.diff(y3)/2
        y4 = [None]*(len(y3)+len(y_int))
        y4[::2] = y3
        y4[1::2] = y_int
        x4 = np.array(x4); y4 = np.array(y4)

    return x4, y4

def selection(x_man, y_man, start_point, stop_point, n_train):
    if (start_point > stop_point): start_point, stop_point = stop_point, start_point

   
    points = list(range(start_point,stop_point+1))
    x_sel = [x_man[i] for i in points]
    y_sel = [y_man[i] for i in points]

    x_train_pre = [x_man[i] for i in range(start_point - n_train,start_point)]
    y_train_pre = [y_man[i] for i in range(start_point - n_train,start_point)]
    x_train_post = [x_man[i] for i in range(stop_point+1,stop_point + 1 + n_train)]
    y_train_post = [y_man[i] for i in range(stop_point+1,stop_point + 1 + n_train)]
    x_train = np.append(x_train_pre, x_train_post); y_train = np.append(y_train_pre, y_train_post)

    return(x_sel, y_sel, x_train, y_train, points)  
def peak_detection(sp_x, sp_y, rev_x = 0, rev_y = 0, rg = 5, delta_input = 0, delta_max = 2, nn = 2):
    alpha_vec = -np.diff(sp_y)/np.diff(sp_x)                                        #the derivatives are calculated for each point
    if (rev_x == 1): 
        sp_x = sp_x[::-1]; sp_y = sp_y[::-1]
        alpha_vec = np.diff(sp_y)/np.diff(sp_x)
    if rev_y == 1: alpha_vec = -alpha_vec

    df = pd.DataFrame({"x_vec": sp_x, "y_vec":sp_y})
    avg_vec = [np.mean(alpha_vec[(i-rg):i]) for i in range(len(alpha_vec))]        #calculate average derivative of rg preceedings points
    if rg == 0: avg_vec = [np.mean(alpha_vec[0:i+1]) for i in range(len(alpha_vec))]             #calculate average derivative of ALL preceedings points

    df.insert(len(df.columns), "alpha_pre", np.append(np.NaN,alpha_vec))
    df.insert(len(df.columns), "alpha_post", np.append(alpha_vec, np.NaN))
    df.insert(len(df.columns), "avg_vec", np.append(np.NaN, avg_vec))
    df.insert(len(df.columns), "delta_tb", np.round(df.alpha_pre - df.avg_vec,2))
    df.insert(len(df.columns), "delta_tp", np.zeros(len(df)))
    df.insert(len(df.columns), "trigger", np.zeros(len(df)))
    df.insert(len(df.columns), "peak", np.zeros(len(df)))

    delta_vec = np.array(df.alpha_pre - df.avg_vec)
    delta_vmax = np.max(delta_vec[rg+1:])
    delta = delta_input*delta_vmax/100
    
    df['trigger'] = np.where((df.alpha_pre - df.avg_vec > delta), 2, df.trigger)

    avg_peak = np.nan
    loop_range = list(range(rg+1,len(df)-rg-1))
    #if rev_x == 1: loop_range = list(reversed(loop_range))
    for i in loop_range:      
        if (np.sum(df.peak)%2==0) & (df.trigger[i] == 2) & (df.trigger[i+1] == 2):
            df.peak[i] = 1
            avg_peak = df.avg_vec[i]

        elif (np.sum(df.peak)%2==1) & (df.alpha_post[i] - avg_peak < delta_max):
            df.peak[i] = 3

        df.delta_tp[i] = df.alpha_post[i] - avg_peak



    peak_list = np.array(df.peak)
    trigger_list = np.array(df.trigger)

    index_peak_0 = np.where(peak_list == 3)[0]
    index_start_0 = np.where(peak_list == 1)[0]

    x_peak = sp_x[index_peak_0]                 #we store the index corresponding to the peaks
    x_start = sp_x[index_start_0]
    if len(x_peak)==0:
        x_peak = [sp_x[0]]
        x_start = [sp_x[0]]
        x_end = [sp_x[0]]
    if (x_start[0]>x_peak[0]): x_end = x_peak + np.abs(x_start[0:len(x_peak)] - x_peak)
    if (x_start[0]<x_peak[0]): x_end = x_peak - np.abs(x_start[0:len(x_peak)] - x_peak)
    x_end = np.asarray([sp_x[(np.abs(i - sp_x)).argmin()] for i in x_end])

    index_end_0 = np.array([np.where(sp_x == i)[0][0] for i in x_end])
    index_peak_0 = np.where(peak_list == 3)[0]
    index_start_0 = np.where(peak_list == 1)[0]

    x_peak = sp_x[index_peak_0]                 #we store the index corresponding to the peaks
    x_start = sp_x[index_start_0]
    x_end = x_peak + np.abs(x_start[0:len(x_peak)] - x_peak)
    if rev_x == 0: x_end = x_peak - np.abs(x_start[0:len(x_peak)] - x_peak)
    x_end = np.asarray([sp_x[(np.abs(i - sp_x)).argmin()] for i in x_end])

    index_end_0 = np.array([np.where(sp_x == i)[0][0] for i in x_end])   


    ## CONVERT THE PEAK LIST INTO COORDINATES AND INDEXES            

    y_peak = np.zeros(len(x_peak))                              #we need the y_coords of the peaks
    y_start = np.zeros(len(x_start))
    y_end = np.zeros(len(x_end))

    y_peak = np.array([sp_y[np.where(sp_x == x)][0] for x in x_peak])
    y_start = np.array([sp_y[np.where(sp_x == x)][0] for x in x_start])
    y_end = np.array([sp_y[np.where(sp_x == x)][0] for x in x_end])

    x_start_0 = x_start; x_peak_0 = x_peak; x_end_0 = x_end  
    y_start_0 = y_start; y_peak_0 = y_peak; y_end_0 = y_end 

    n_peaks = len(x_end) 

    x_peak = x_peak[0:n_peaks]
    y_peak = y_peak[0:n_peaks]
    x_start = x_start[0:n_peaks]
    y_start = y_start[0:n_peaks]

    index_start = np.array([np.where(sp_x == i)[0][0] for i in x_start])
    index_peak = np.array([np.where(sp_x == i)[0][0] for i in x_peak])
    index_end = index_end_0

    index_in = [list(range(index_start[i], index_end[i] + 1)) for i in range(n_peaks)]
    x_in = [sp_x[i] for i in index_in]
    y_in = [sp_y[i] for i in index_in]

    index_in_m = list(set([j for i in index_in for j in i]))
    x_in_m = [j for i in x_in for j in i]
    y_in_m = [j for i in y_in for j in i]

    index_out = [x for x in range(0, len(sp_x)) if x not in index_in_m]
    x_out = [sp_x[i] for i in index_out]
    y_out = [sp_y[i] for i in index_out]


    index_trigger = np.where(trigger_list == 2)[0]
    x_trigger = sp_x[index_trigger]
    y_trigger = np.array([sp_y[np.where(sp_x == x)][0] for x in x_trigger])

    ###GENERATE OUTPUT

    pd_out = (df, n_peaks, x_start, x_peak, x_end, x_start_0, x_peak_0, x_end_0,
              y_start, y_peak, y_end, y_start_0, y_peak_0, y_end_0,
              index_start, index_peak, index_end, index_start_0, index_peak_0, index_end_0,
              x_trigger, y_trigger,
              index_in, index_in_m, index_out, x_in, x_in_m, y_in, y_in_m, x_out, y_out)
    
    if (rev_x == 1) & (len(index_start)>0): index_start = len(sp_x)-index_start-1
    if (rev_x == 1) & (len(index_start)>0): index_end = len(sp_x)-index_end-1
    if len(index_start) == 0: index_start=[0,0]
    if len(index_end) == 0: index_end=[0,0]
    if len(x_start) == 0: x_start = [sp_x[0],sp_x[0]]
    if len(x_peak) == 0: x_peak = [sp_x[0],sp_x[0]]
    if len(x_end) == 0: x_end = [sp_x[0],sp_x[0]]


    #return index_start, index_end, x_start, x_peak, x_end, df, delta
    return index_start, index_end, x_start, x_peak, x_end
def intp(lst, ind):
    int1 = (lst[ind]+lst[ind+1])/2
    int2 = (lst[ind]+lst[ind-1])/2
    return int1, int2
def GP(x_man, y_man, x_train, y_train):
    X = np.array(x_train).reshape(-1,1)
    Y = np.array(y_train).reshape(-1,1)

    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize(); m.optimize_restarts(num_restarts = 20, verbose=False)

    x_test = x_man
    newX = x_man[:,None]
    newX = np.hstack([newX,np.ones_like(newX)])
    bg = m.predict(newX)
    y_test = bg[0]
    bg_err = m.predict_quantiles( X = newX, quantiles = (2.5, 97.5) )
    test_err_up = bg_err[1].reshape(1,-1)[0]
    test_err_down = bg_err[0].reshape(1,-1)[0]

    x_bg = x_test #= x_in[i]
    y_bg = y_test         #it could give an error if less than 5 training points per side available 
    conf_int_up = test_err_up
    conf_int_down = test_err_down
    err_bg = [j[0] - j[1] for j in zip(conf_int_up, y_bg)]
    x_sig = [x_bg[i] for i in points]
    y_sig = [y_bg[i] for i in points]
    err_sig = [err_bg[i] for i in points]

    h = [np.abs(j[0] - j[1]) for j in zip(y_sel, y_sig)]
    h_index = np.argmax(h)
    
    conf_up_sig = [j[0]+j[1] for j in zip(y_sig,err_sig)]
    conf_down_sig = [j[0]-j[1] for j in zip(y_sig,err_sig)]
    
    h_up = [np.abs(j[0] - j[1]) for j in zip(y_sel, conf_up_sig)]
    h_down = [np.abs(j[0] - j[1]) for j in zip(y_sel, conf_down_sig)]



    x_start = x_sel[0]; x_end = x_sel[-1]
    y_start = y_sel[0]; y_end = y_sel[-1]
    x_peak = x_sel[h_index]; y_peak = y_sel[h_index]
    
    area = np.sum([i[0]*i[1] for i in zip(np.abs(np.diff(x_sel)),h[0:-1])])
    area_up = np.sum([i[0]*i[1] for i in zip(np.abs(np.diff(x_sel)),h_up[0:-1])])
    area_down = np.sum([i[0]*i[1] for i in zip(np.abs(np.diff(x_sel)),h_down[0:-1])])
    
    ampl = y_sel[h_index] - y_sig[h_index]
    ampl_up = ampl + err_sig[h_index]
    ampl_down = ampl - err_sig[h_index]
    if ampl < 0:
        ampl = -ampl
        ampl_up = -ampl_up
        ampl_down = -ampl_down
        
    if intp(y_sel, h_index)[0]-intp(y_sig, h_index)[0] > ampl:
        x_peak = intp(x_sel, h_index)[0]; y_peak = intp(y_sel, h_index)[0]
        ampl = intp(y_sel, h_index)[0]-intp(y_sig, h_index)[0]
        
    if intp(y_sel, h_index)[1]-intp(y_sig, h_index)[1] > ampl:
        x_peak = intp(x_sel, h_index)[1]; y_peak = intp(y_sel, h_index)[1]
        ampl = intp(y_sel, h_index)[1]-intp(y_sig, h_index)[1]

    return x_bg, y_bg, conf_int_up, conf_int_down, x_start, y_start, x_end, y_end, x_peak, y_peak, area, err_bg, y_sig, area_up, area_down, ampl, ampl_up, ampl_down
def sig_ass(y_sig):
    n_pe = 500000
    #n_pe = 1744278

    chisq_data = np.sum(chi_square(y_sel, y_sig))   #calculate chisquare_data
    #pval_list_val.append(chi2.sf(x = chisq_data, df = len(x_in[i]) - 1))

    x_sig = [x_bg[i] for i in points]
    y_sig = [y_bg[i] for i in points]
    err_sig = [err_bg[i] for i in points]

    #pe = zip(*[np.random.normal(j[0], j[1], n_pe) for j in zip(y_sig, err_sig)])  #generate n_pe pseudo-experiments
    pe = []
    g_par = list(zip(y_sig, err_sig))
    for j in range(len(g_par)):
        pe.append(np.random.normal(g_par[j][0], g_par[j][1], n_pe))
        #progress['value'] = (j+1)*(220/len(g_par))
        root.update_idletasks()    
    pe = list(zip(*pe))
    
    chisq_pseudo = [np.sum(chi_square(j, y_sig)) for j in pe]
        
    chisq_bigger = [i for i in chisq_pseudo if i > chisq_data ]   #check how many chisquare_pe are bigger than chisquare_data

    n_bigger = len(chisq_bigger)
    p_val = float(n_bigger)/n_pe
    sig = np.sqrt(2)*erfinv(1 - p_val)
    
    return p_val, sig, pe, x_sig
def chi_square(observed, expected):
    return [((i_obs - i_exp)**2)/np.abs(i_exp) for i_obs, i_exp in zip(observed, expected)]

figure = plt.Figure(figsize=(5,4), dpi=150)
ax = figure.add_subplot(111)
scatter = FigureCanvasTkAgg(figure, root) 
ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title('')
scatter.get_tk_widget().grid(row=cp_row-4,column=0, rowspan=30)

canvas = Canvas(root, width = 250, height = 250)  
img = Image.open(folder+"\icons\peakIt_icon_2.png")
img = img.resize((251, 251), Image.ANTIALIAS) 
img = ImageTk.PhotoImage(img)
canvas.create_image(2, 2, anchor=NW, image=img) 
canvas.grid(row=5,column=0, rowspan=10)


def browseFiles():
    global data; global x; global y; global filename
    filename = filedialog.askopenfilename(initialdir = folder,
                                          title = "Select a File",
                                          filetypes = (("Excel files",
                                                        "*.xlsx*"),
                                                       ("all files",
                                                        "*.*")))
    data = pd.read_excel(filename, header = None)
    x = np.array(data[0]); y = np.array(data[1])
    if x[0]<x[1]:
        x = x[::-1]
        y = y[::-1]
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x,y, '.', color = 'k')
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title('')
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    xl1 = ax.get_xlim()[0]
    xl2 = ax.get_xlim()[1]
    yl1 = ax.get_ylim()[0]
    yl2 = ax.get_ylim()[1]
    if xl1<xl2: xl1,xl2=xl2,xl1
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
browse_b = Button(root, text = "Browse Files", padx= 50, pady= 18, command = browseFiles)
browse_b.grid(column = 1, row = cp_row-4, columnspan=4)

#creating a label widget
Label1 = Label(root, text = "x_min = ")
Label2 = Label(root, text = "x_max = ")
Label1.grid(row=cp_row-3,column=1)
Label2.grid(row=cp_row-2,column=1)

x_min_e = Entry(root, width=5)
x_min_e.grid(row=cp_row-3, column =2)
x_min_e.insert(0, "1")

x_max_e = Entry(root, width=5)
x_max_e.grid(row=cp_row-2, column =2)
x_max_e.insert(0, "3.5")


#var3 = tk.IntVar()
#interp_e = tk.Checkbutton(root, text='int',variable=var3, onvalue=1, offvalue=0)
#interp_e.grid(row=cp_row-4, column =4)
interp=0


def select_range():
    global x_min; global x_max; global x1; global y1; global xl1; global xl2; global yl1; global yl2; global xl3; global xl4; global yl3; global yl4
    x_min = float(x_min_e.get())
    x_max = float(x_max_e.get())
    #interp = int(var3.get())
    x1, y1 = data_process(x, y, cut_min = x_min, cut_max = x_max, y_scale = 1, bd = 5, interp=interp)
    print(type(x1))
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title('')

    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    xl1 = ax.get_xlim()[0]
    xl2 = ax.get_xlim()[1]
    yl1 = ax.get_ylim()[0]
    yl2 = ax.get_ylim()[1]
    if xl1<xl2: xl1,xl2=xl2,xl1
    xl3=xl1; xl4=xl2; yl3 = yl1; yl4 = yl2

    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);

    return
    
select_range_b = Button(root, text = "Select range", padx= 22, pady= 18,command = select_range)
select_range_b.grid(row=cp_row-3, column =3, rowspan=2, columnspan=2)


Label01 = Label(root, text = "delta = ")
Label02 = Label(root, text = "delta* = ")
Label03 = Label(root, text = "    range =")

Label01.grid(row=cp_row-1,column=1)
Label02.grid(row=cp_row-0,column=1)
Label03.grid(row=cp_row-1,column=3)

delta_e = Entry(root, width=5)
delta_e.grid(row=cp_row-1, column =2)
delta_e.insert(0, "0")

delta_st_e = Entry(root, width=5)
delta_st_e.grid(row=cp_row-0, column =2)
delta_st_e.insert(0, "0")

range_e = Entry(root, width=5)
range_e.grid(row=cp_row-1, column =4)
range_e.insert(0, "5")

var1 = tk.IntVar()
var2 = tk.IntVar()
rev_x_e = tk.Checkbutton(root, text='rev',variable=var1, onvalue=1, offvalue=0)
rev_x_e.grid(row=cp_row-0, column =3)
rev_x_e = tk.Checkbutton(root, text='up',variable=var2, offvalue=1, onvalue=0)
rev_x_e.grid(row=cp_row-0, column =4)


def automatic_detection():
    global delta; global delta_max; global rg; global point_min; global point_max; global n_peak
    rev_x = int(var1.get())
    rev_y = int(var2.get());
    delta = float(delta_e.get())
    delta_max = float(delta_st_e.get())
    rg = int(range_e.get())
    point_min, point_max, x_start, x_peak, x_end = peak_detection(x1, y1, rev_x, rev_y, rg, delta, delta_max)
    n_peak = 0
    point_min_e.delete(0,'end')
    point_min_e.insert(0, point_min[n_peak])
    point_max_e.delete(0,'end')
    point_max_e.insert(0, point_max[n_peak])
    
    npeaks = len(x_peak)
    if (npeaks == 2):
        if (x_start[0]==x_start[1]): 
            npeaks=0
    peak_label = Label(root, text = "number of peaks: "+str(npeaks))
    peak_label.grid(row=cp_row+2,column=1, columnspan = 2)

    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.axvline(x_peak[n_peak], color = "steelblue", linestyle = ":", label = "max of the peak")
    ax.axvline(x_start[n_peak], color = "red", linestyle = "--", label = "border of the peak")
    ax.axvline(x_end[n_peak], color = "red", linestyle = "--")
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
det_b = Button(root, text = "Detect peak", padx= 78, pady= 18,command = automatic_detection)
det_b.grid(row=cp_row+1, column =1, columnspan=4)

peak_label = Label(root, text = "number of peaks:")
peak_label.grid(row=cp_row+2,column=1, columnspan = 2)

def automatic_detection2():
    global delta; global delta_max; global rg; global point_min; global point_max; global n_peak
    rev_x = int(var1.get())
    rev_y = int(var2.get())
    delta = float(delta_e.get())
    delta_max = float(delta_st_e.get())
    rg = int(range_e.get())
    point_min, point_max, x_start, x_peak, x_end = peak_detection(x1, y1, rev_x, rev_y, rg, delta, delta_max)
    n_peak = n_peak + 1
    if n_peak == len(x_peak): n_peak = 0
    point_min_e.delete(0,'end')
    point_min_e.insert(0, point_min[n_peak])
    point_max_e.delete(0,'end')
    point_max_e.insert(0, point_max[n_peak])

    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.axvline(x_peak[n_peak], color = "steelblue", linestyle = ":", label = "max of the peak")
    ax.axvline(x_start[n_peak], color = "red", linestyle = "--", label = "border of the peak")
    ax.axvline(x_end[n_peak], color = "red", linestyle = "--")
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)

det_b = Button(root, text = "Change peak", padx= 22, pady= 9,command = automatic_detection2)
det_b.grid(row=cp_row+2, column =3, columnspan=2)


Label3 = Label(root, text = "point_min = ")
Label4 = Label(root, text = "point_max = ")
Label3.grid(row=cp_row+5,column=1)
Label4.grid(row=cp_row+6,column=1)

point_min_e = Entry(root, width=5)
point_min_e.grid(row=cp_row+5, column =2)
#point_min_e.insert(0, 35)

point_max_e = Entry(root, width=5)
point_max_e.grid(row=cp_row+6, column =2)
#point_max_e.insert(0, 54)

Labelnp = Label(root, text = "peak")


def select_peak():
    global start_point; global stop_point; global x_sel; global y_sel; global x_train; global y_train; global points
    start_point = int(point_min_e.get())
    stop_point = int(point_max_e.get())
    n_train = int(range_e.get())
    x_sel, y_sel, x_train, y_train, points = selection(x1, y1, start_point, stop_point, n_train)

    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)

    return

select_peak_b = Button(root, text = "Select peak", padx= 25, pady= 18,command = select_peak)
select_peak_b.grid(row=cp_row+5, column =3, rowspan=2, columnspan=2)

def detect_peak():
    peak_label = Label(root, text = "Closer peak selected")
    peak_label.grid(row=cp_row+7,column=1, columnspan = 4)
    return

#assess_sig_b = Button(root, text = "Sig-driven detection", padx= 55, pady= 18,command = detect_peak, state='disabled')
#assess_sig_b.grid(row=cp_row+7, column =1, columnspan=4)

peak_label = Label(root, text = "")
peak_label.grid(row=cp_row+7,column=1, columnspan = 4)

def calculate_bg():
    peak_label = Label(root, text = "Peak detected: ")
    peak_label.grid(row=cp_row+9,column=1, columnspan = 4)
    peak_label = Label(root, text = "Peak range: ")
    peak_label.grid(row=cp_row+10,column=1, columnspan = 4)
    peak_label = Label(root, text = "Peak height: ")
    peak_label.grid(row=cp_row+11,column=1, columnspan = 4)
    peak_label = Label(root, text = "Peak area: ")
    peak_label.grid(row=cp_row+12,column=1, columnspan = 4)

    global x_bg; global y_bg; global conf_int_up; global conf_int_down; global x_start; global y_start; global x_end; global y_end; global x_peak; global y_peak; global area; global err_bg; global y_sig; global area_up; global area_down; global ampl; global ampl_up; global ampl_down
    x_bg, y_bg, conf_int_up, conf_int_down, x_start, y_start, x_end, y_end, x_peak, y_peak, area, err_bg, y_sig, area_up, area_down, ampl, ampl_up, ampl_down = GP(x1, y1, x_train, y_train)
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);


    peak_label = Label(root, text = "Peak detected: " + str(np.round(x_peak,2)) + " ppm") 
    peak_label.grid(row=cp_row+9,column=1, columnspan = 4)

    if x_start<x_end: x_start,x_end=x_end,x_start
    peak_label = Label(root, text = "Peak range: "+ str(np.round(x_start,2)) + ' - ' + str(np.round(x_end,2)) + " ppm")
    peak_label.grid(row=cp_row+10,column=1, columnspan = 4)

    ampl_err = np.abs(ampl-ampl_up)
    peak_label = "Peak height: " + str("{:.2e}".format(ampl[0])) + " ± " + str("{:.2e}".format(ampl_err[0]))
    #peak_label = Label(root, text = "Peak amplitude: " + str("{:.2e}".format(ampl[0])) + " (" +  str("{:.2e}".format(ampl_up[0])) + "-" + str("{:.2e}".format(ampl_down[0])) + ")")
    peak_label = Label(root, text = peak_label)
    peak_label.grid(row=cp_row+11,column=1, columnspan = 4)
    
    area_err = np.abs(area-area_up)
    area_label = "Peak area: " + str("{:.2e}".format(area)) + " ± " + str("{:.2e}".format(area_err))
    #area_label = "Peak area: " + str("{:.2e}".format(area)) + " (" + str("{:.2e}".format(area_up)) + "-" + str("{:.2e}".format(area_down)) + ")"
    peak_label = Label(root, text = area_label)
    peak_label.grid(row=cp_row+12,column=1, columnspan = 4)
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    return

calculate_bg_b = Button(root, text = "Calculate background", padx= 50, pady= 18,command = calculate_bg)
calculate_bg_b.grid(row=cp_row+8, column =1, columnspan=4)

#progress = Progressbar(root, orient = HORIZONTAL,length = 220, mode = 'determinate')
#progress.grid(row=cp_row+14, column =1, columnspan=4)

peak_label = Label(root, text = "Peak detected: ")
peak_label.grid(row=cp_row+9,column=1, columnspan = 4)
peak_label = Label(root, text = "Peak range: ")
peak_label.grid(row=cp_row+10,column=1, columnspan = 4)
peak_label = Label(root, text = "Peak height: ")
peak_label.grid(row=cp_row+11,column=1, columnspan = 4)
peak_label = Label(root, text = "Peak area: ")
peak_label.grid(row=cp_row+12,column=1, columnspan = 4)

def assess_sig():
    global p_val, sig, pe, x_sig
    peak_label = Label(root, text = "p-value: ")
    peak_label.grid(row=cp_row+15,column=1, columnspan = 4)

    peak_label = Label(root, text = "z-score: ")
    peak_label.grid(row=cp_row+16,column=1, columnspan = 4)
    
    p_val, sig, pe, x_sig = sig_ass(y_sig)
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)

    
    if p_val == 0:
        p_val_r = "< 5 e-7"
        sig_r = "> 5 sigma"
    else:
        p_val_r = str("{:.2e}".format(p_val))
        sig_r = str(np.round(sig,2))
    
    peak_label = Label(root, text = "p-value: " + p_val_r)
    peak_label.grid(row=cp_row+15,column=1, columnspan = 4)

    peak_label = Label(root, text = "z-score: " + sig_r + " sigma")
    peak_label.grid(row=cp_row+16,column=1, columnspan = 4)
    return

assess_sig_b = Button(root, text = "Calculate significance", padx= 50, pady= 18,command = assess_sig)
assess_sig_b.grid(row=cp_row+13, column =1, columnspan=4)

peak_label = Label(root, text = "p-value: ")
peak_label.grid(row=cp_row+15,column=1, columnspan = 4)

peak_label = Label(root, text = "z-score: ")
peak_label.grid(row=cp_row+16,column=1, columnspan = 4)


def zoom_in_x():
    

    global xl3; global xl4;

    xl3=xl3-(xl3*5/100)
    xl4=xl4+(xl4*5/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
def zoom_in_y():
    global yl3; global yl4

    yl3=yl3+(yl3*3/100)
    yl4=yl4-(yl4*3/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
def zoom_out_x():
    

    global xl3; global xl4;

    xl3=xl3+(xl3*5/100)
    xl4=xl4-(xl4*5/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
def zoom_out_y():
    
    global yl3; global yl4

    yl3=yl3-(yl3*3/100)
    yl4=yl4+(yl4*3/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
def left():
    global xl3; global xl4;

    xl3=xl3+(xl3*5/100)
    xl4=xl4+(xl4*5/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
def right():
    global xl3; global xl4;

    xl3=xl3-(xl3*5/100)
    xl4=xl4-(xl4*5/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);

    return
    return
def up():
    global yl3; global yl4

    yl3=yl3+(yl3*3/100)
    yl4=yl4+(yl4*3/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);
    return
def down():
    global yl3; global yl4

    yl3=yl3-(yl3*3/100)
    yl4=yl4-(yl4*3/100)
    
    
    figure = plt.Figure(figsize=(5,4), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(x1,y1, '.',color = 'k')
    ax.plot(x_sel, y_sel, '.',color = 'red', label = "peak points")
    ax.plot(x_train, y_train, '.',color = 'green', label = "training points")
    ax.axvline(x_sel[0], color = 'red')
    ax.axvline(x_sel[-1], color = 'red')
    ax.plot(x_bg, y_bg, color = 'orange', label = "baseline")
    ax.fill_between(x_bg, conf_int_up, conf_int_down, alpha = 0.2, label = "confidence interval")
    ax.plot(x_peak, y_peak, 'or')
    ax.plot(x_start, y_start, 'or')
    ax.plot(x_end, y_end, 'or')
    ax.plot(x_sig, pe[0], '.y', label = "pseudo-experiment")

    
    scatter = FigureCanvasTkAgg(figure, root) 
    ax.set_xlabel('Saturation offset (ppm)'); ax.set_ylabel(r'$S_{sat}/S_{0}$'); ax.set_title(''); ax.legend()
    #scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    scatter.get_tk_widget().grid(row=0,column=0, rowspan=30)
    ax.set_xlim([xl1,xl2]); ax.set_ylim([yl1,yl2]);
    
    lab_tit = Label(root, text = filename)
    lab_tit.grid(row=0,column=0)
    
    
    ax.set_xlim([xl3,xl4]); ax.set_ylim([yl3,yl4]);
    return


x_label = Label(root, text = "x-axis ")
x_label.grid(row=cp_row+17,column=1, columnspan = 1)

x_b_plus = Button(root, text = "+", command = zoom_in_x)
x_b_plus.grid(row=cp_row+17, column =2, columnspan=1, sticky='W')

x_b_minus = Button(root, text = "-", command = zoom_out_x)
x_b_minus.grid(row=cp_row+17, column =2, columnspan=1, sticky='E')


y_label = Label(root, text = "y-axis ")
y_label.grid(row=cp_row+18,column=1, columnspan = 1)

y_b_plus = Button(root, text = "+", command = zoom_in_y)
y_b_plus.grid(row=cp_row+18, column =2, columnspan=1, sticky ='W')

y_b_minus = Button(root, text = "-", command = zoom_out_y)
y_b_minus.grid(row=cp_row+18, column =2, columnspan=1,sticky='E')


left = Button(root, text = "←", command = left)
left.grid(row=cp_row+17, column =3, rowspan=2, sticky="E",padx=10)
left = Button(root, text = "→", command = right)
left.grid(row=cp_row+17, column =4, rowspan=2)
left = Button(root, text = "↑", command = up)
left.grid(row=cp_row+17, column =4, rowspan=1,sticky='W')
left = Button(root, text = "↓", command = down)
left.grid(row=cp_row+18, column =4, rowspan=1,sticky='W')


root.mainloop()
