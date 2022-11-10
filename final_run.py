#!/usr/bin/env python
# coding: utf-8

# # Hackathon Solution : Multi-stream - Multi-DNN pipeline
# 
# In this notebook, you will build an Multi-stream Multi-DNN pipeline using the concepts learned from the previous notebooks. 
# 

# ## Building the pipeline
# 
# We will the using batched on the Multi-DNN network from [Notebook 3](Introduction_to_Multi-DNN_pipeline.ipynb) and combine it with the knowledge learnt in [Notebook 4](Multi-stream_pipeline.ipynb). 
# 
# 
# Here are the illustrations of the Pipeline 
# ![test2](images/test2.png)
# ![test3](images/test3.png)
# 
# Let us get started with the Notebook , You will have to fill in the `TODO` parts of the code present in the Notebook to complete the pipeline. Feel free to refer to the previous notebooks for the commands.

# In[1]:


# Import required libraries 
import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns

# Define variables to be used later
fps_streams={}

PGIE_CLASS_ID_FACE = 2
PGIE_CLASS_ID_BAG = 1
PGIE_CLASS_ID_PERSON = 0

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1080
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 0
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]

################ Three Stream Pipeline ###########
# Define Input and output Stream information 
num_sources = 4 
INPUT_VIDEO_1 = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'
INPUT_VIDEO_2  = 'sources/few_people.h264'
INPUT_VIDEO_3  = 'sources/few_people.h264'
INPUT_VIDEO_4 = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

OUTPUT_VIDEO_NAME = "output/final_output.mp4"


# We define a function `make_elm_or_print_err()` to create our elements and report any errors if the creation fails.
# 
# Elements are created using the `Gst.ElementFactory.make()` function as part of Gstreamer library.

# In[2]:


## Make Element or Print Error and any other detail
def make_elm_or_print_err(factoryname, name, printedname, detail=""):
  print("Creating", printedname)
  elm = Gst.ElementFactory.make(factoryname, name)
  if not elm:
     sys.stderr.write("Unable to create " + printedname + " \n")
  if detail:
     sys.stderr.write(detail)
  return elm


# #### Initialise GStreamer and Create an Empty Pipeline

# In[3]:


for i in range(0,num_sources):
        fps_streams["stream{0}".format(i)]=GETFPS(i)

# Standard GStreamer initialization
GObject.threads_init()
Gst.init(None)

# Create gstreamer elements */
# Create Pipeline element that will form a connection of other elements
print("Creating Pipeline \n ")
pipeline = Gst.Pipeline()

if not pipeline:
    sys.stderr.write(" Unable to create Pipeline \n")


# #### Create Elements that are required for our pipeline
# 
# Compared to the first notebook , we use a lot of queues in this notebook to buffer data when it moves from one plugin to another.

# In[4]:


########### Create Elements required for the Pipeline ########### 

######### Defining Stream 1 
# Source element for reading from the file
source1 = make_elm_or_print_err("filesrc", "file-source-1",'file-source-1')
# Since the data format in the input file is elementary h264 stream,we need a h264parser
h264parser1 = make_elm_or_print_err("h264parse", "h264-parser-1","h264-parser-1")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder1 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-1","nvv4l2-decoder-1")
   
##########

########## Defining Stream 2 
# Source element for reading from the file
source2 = make_elm_or_print_err("filesrc", "file-source-2","file-source-2")
# Since the data format in the input file is elementary h264 stream, we need a h264parser
h264parser2 = make_elm_or_print_err("h264parse", "h264-parser-2", "h264-parser-2")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder2 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-2","nvv4l2-decoder-2")
########### 

########## Defining Stream 3
# Source element for reading from the file
source3 = make_elm_or_print_err("filesrc", "file-source-3","file-source-3")
# Since the data format in the input file is elementary h264 stream, we need a h264parser
h264parser3 = make_elm_or_print_err("h264parse", "h264-parser-3", "h264-parser-3")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder3 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-3","nvv4l2-decoder-3")
########### 

########## Defining Stream 4
# Source element for reading from the file
source4 = make_elm_or_print_err("filesrc", "file-source-4","file-source-4")
# Since the data format in the input file is elementary h264 stream, we need a h264parser
h264parser4 = make_elm_or_print_err("h264parse", "h264-parser-4", "h264-parser-4")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder4 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-4","nvv4l2-decoder-4")
########### 


# Create nvstreammux instance to form batches from one or more sources.
streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer","Stream-muxer") 
# Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file
pgie = make_elm_or_print_err("nvinfer", "primary-inference" ,"pgie")
# Use nvtracker to give objects unique-ids
tracker = make_elm_or_print_err("nvtracker", "tracker",'tracker')
# Creating Tiler to present more than one streams
tiler=make_elm_or_print_err("nvmultistreamtiler", "nvtiler","nvtiler")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor","nvvidconv")
# Create OSD to draw on the converted RGBA buffer
nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay","nvosd")
# Creating queue's to buffer incoming data from pgie
queue1=make_elm_or_print_err("queue","queue1","queue1")
# Creating queue's to buffer incoming data from tiler
queue2=make_elm_or_print_err("queue","queue2","queue2")
# Creating queue's to buffer incoming data from nvvidconv
queue3=make_elm_or_print_err("queue","queue3","queue3")
# Creating queue's to buffer incoming data from nvosd
queue4=make_elm_or_print_err("queue","queue4","queue4")
# Creating queue's to buffer incoming data from nvvidconv2
queue5=make_elm_or_print_err("queue","queue5","queue5")
# Creating queue's to buffer incoming data from nvtracker
queue6=make_elm_or_print_err("queue","queue6","queue6")
# Creating queue's to buffer incoming data from sgie1
queue7=make_elm_or_print_err("queue","queue7","queue7")
# Creating queue's to buffer incoming data from sgie2
queue8=make_elm_or_print_err("queue","queue8","queue8")
# Creating queue's to buffer incoming data from sgie3
queue9=make_elm_or_print_err("queue","queue9","queue9")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2","nvvidconv2")
# Place an encoder instead of OSD to save as video file
encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder")
# Parse output from Encoder 
codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')
# Create a container
container = make_elm_or_print_err("qtmux", "qtmux", "Container")
# Create Sink for storing the output 
sink = make_elm_or_print_err("filesink", "filesink", "Sink")


# Now that we have created the elements ,we can now set various properties for out pipeline at this point. The configuration files are the same as in [Multi-DNN Notebook](Introduction_to_Multi-DNN_pipeline.ipynb)

# In[5]:


############ Set properties for the Elements ############
# Set Input Video files 
source1.set_property('location', INPUT_VIDEO_1)
source2.set_property('location', INPUT_VIDEO_2)
source3.set_property('location', INPUT_VIDEO_3)
source4.set_property('location', INPUT_VIDEO_4)


# Set Input Width , Height and Batch Size 
streammux.set_property('width', 1920)
streammux.set_property('height', 1080)
streammux.set_property('batch-size', num_sources)
# Timeout in microseconds to wait after the first buffer is available 
# to push the batch even if a complete batch is not formed.
streammux.set_property('batched-push-timeout', 4000000)
# Set configuration file for nvinfer 
# Set Congifuration file for nvinfer 
pgie.set_property('config-file-path', "config/config_infer_primary_peoplenet.txt")

#Set properties of tracker from tracker_config
config = configparser.ConfigParser()
config.read('config/dstest4_tracker_config.txt')
config.sections()
for key in config['tracker']:
    if key == 'tracker-width' :
        tracker_width = config.getint('tracker', key)
        tracker.set_property('tracker-width', tracker_width)
    if key == 'tracker-height' :
        tracker_height = config.getint('tracker', key)
        tracker.set_property('tracker-height', tracker_height)
    if key == 'gpu-id' :
        tracker_gpu_id = config.getint('tracker', key)
        tracker.set_property('gpu_id', tracker_gpu_id)
    if key == 'll-lib-file' :
        tracker_ll_lib_file = config.get('tracker', key)
        tracker.set_property('ll-lib-file', tracker_ll_lib_file)
    if key == 'll-config-file' :
        tracker_ll_config_file = config.get('tracker', key)
        tracker.set_property('ll-config-file', tracker_ll_config_file)
    if key == 'enable-batch-process' :
        tracker_enable_batch_process = config.getint('tracker', key)
        tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        
## Set batch size 
pgie_batch_size=pgie.get_property("batch-size")
print("PGIE batch size :",end='')
print(pgie_batch_size)
if(pgie_batch_size != num_sources):
    print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", num_sources," \n")
    pgie.set_property("batch-size",num_sources)
        
# Set display configurations for nvmultistreamtiler    
tiler_rows=int(2)
tiler_columns=int(2)
tiler.set_property("rows",tiler_rows)
tiler.set_property("columns",tiler_columns)
tiler.set_property("width", TILED_OUTPUT_WIDTH)
tiler.set_property("height", TILED_OUTPUT_HEIGHT)

# Set encoding properties and Sink configs
encoder.set_property("bitrate", 2000000)
sink.set_property("location", OUTPUT_VIDEO_NAME)
sink.set_property("sync", 0)
sink.set_property("async", 0)


# We now link all the elements in the order we prefer and create Gstreamer bus to feed all messages through it. 

# In[6]:


########## Add and Link ELements in the Pipeline ########## 

print("Adding elements to Pipeline \n")
pipeline.add(source1)
pipeline.add(h264parser1)
pipeline.add(decoder1)
pipeline.add(source2)
pipeline.add(h264parser2)
pipeline.add(decoder2)
pipeline.add(source3)
pipeline.add(h264parser3)
pipeline.add(decoder3)
pipeline.add(source4)
pipeline.add(h264parser4)
pipeline.add(decoder4)
pipeline.add(streammux)
pipeline.add(pgie)
pipeline.add(tracker)
pipeline.add(tiler)
pipeline.add(nvvidconv)
pipeline.add(nvosd)
pipeline.add(queue1)
pipeline.add(queue2)
pipeline.add(queue3)
pipeline.add(queue4)
pipeline.add(queue5)
pipeline.add(queue6)
pipeline.add(queue7)
pipeline.add(queue8)
pipeline.add(queue9)
pipeline.add(nvvidconv2)
pipeline.add(encoder)
pipeline.add(codeparser)
pipeline.add(container)
pipeline.add(sink)

print("Linking elements in the Pipeline \n")

source1.link(h264parser1)
h264parser1.link(decoder1)


###### Create Sink pad and connect to decoder's source pad 
sinkpad1 = streammux.get_request_pad("sink_0")
if not sinkpad1:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad1 = decoder1.get_static_pad("src")
if not srcpad1:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad1.link(sinkpad1)

######

###### Create Sink pad and connect to decoder's source pad 
source2.link(h264parser2)
h264parser2.link(decoder2)

sinkpad2 = streammux.get_request_pad("sink_1")
if not sinkpad2:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad2 = decoder2.get_static_pad("src")
if not srcpad2:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad2.link(sinkpad2)

######

###### Create Sink pad and connect to decoder's source pad 
source3.link(h264parser3)
h264parser3.link(decoder3)

sinkpad3 = streammux.get_request_pad("sink_2")
if not sinkpad2:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad3 = decoder3.get_static_pad("src")
if not srcpad3:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad3.link(sinkpad3)

######
###### Create Sink pad and connect to decoder's source pad 
source4.link(h264parser4)
h264parser4.link(decoder4)

sinkpad4 = streammux.get_request_pad("sink_3")
if not sinkpad4:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad4 = decoder4.get_static_pad("src")
if not srcpad3:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad4.link(sinkpad4)

######

streammux.link(queue1)
queue1.link(pgie)
pgie.link(queue2)
queue2.link(tracker)
tracker.link(queue3)
queue3.link(tiler)
tiler.link(queue7)
queue7.link(nvvidconv)
nvvidconv.link(queue8)
queue8.link(nvosd)
nvosd.link(queue9)
queue9.link(nvvidconv2)
nvvidconv2.link(encoder)
encoder.link(codeparser)
codeparser.link(container)
container.link(sink)


# In[7]:


# create an event loop and feed gstreamer bus mesages to it
loop = GObject.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect ("message", bus_call, loop)


# Our pipeline now carries the metadata forward but we have not done anything with it until now, but as mentoioned in the above pipeline diagram , we will now create a callback function to write relevant data on the frame once called and create a sink pad in the nvosd element to call the function. 
# 
# This callback function is the same as used in the previous notebook.

# In[8]:


tot_violations = [0,0,0,0]
person_ids = [[],[],[],[]]
frame_plot = [[],[],[],[]]
frame_viol_count = [[],[],[],[]]
aggrgate_100_frame_viol= [0,0,0,0]
#heatmap_data=[[],[],[],[]]
rows, cols = (11, 20) 
heatmap_data_0 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
heatmap_data_1 =  np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
heatmap_data_2 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
heatmap_data_3 =np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0


#graph = cv2.VideoWriter('final_graph.avi',  cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440,1440))
#heatmap_vid = cv2.VideoWriter('funal_heatmap.avi',  cv2.VideoWriter_fourcc(*'DIVX'), 15, (2400,1600))

# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data):
    global tot_violations
    global person_ids 
    global heatmap_data_0,heatmap_data_1,heatmap_data_2,heatmap_data_3
    
    
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_FACE:2,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BAG:1
    }
    # Set frame_number & rectangles to draw as 0 
    frame_number=0
    num_rects=0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    f=0 # To keep track of which frame of batch
    
    while l_frame is not None:
        

        
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        # Get frame number , number of rectables to draw and object metadata
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        
        # Get frame number , number of rectables to draw and object metadata
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        my_coords=[]
        parsed_list=[]
        this_pair=[]
        other_pair=[]
        violation_count=0
        my_obj_color=[]
        unsafe_list=[]
        obj_details=[]
        x_array=[]
        y_array=[]
        x1_array=[]
        y1_array=[]
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # Increment Object class by 1 and Set Box border to Red color 
            
            if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                obj_counter[obj_meta.class_id] += 1
            #obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 0.0)
            #print('obj_meta obj id is',obj_meta.object_id)

            ###my change
            if obj_meta.class_id != PGIE_CLASS_ID_PERSON:
                obj_meta.rect_params.border_width = 0
                obj_meta.text_params.display_text= ''
                #obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                obj_meta.text_params.set_bg_clr = 0
            else:
                my_coords.append(((obj_meta.rect_params.left+obj_meta.rect_params.width/2),(obj_meta.rect_params.top+obj_meta.rect_params.height/2),obj_meta.object_id,obj_meta))
                obj_details.append(obj_meta)
                #print(obj_meta.object_id)
                bbox_ht= obj_meta.rect_params.height
                #p=obj_meta
                #obj_meta_disp= pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                #obj_meta_disp.num_labels = 1
                #pyds.nvds_add_user_meta_to_obj(obj_meta,obj_meta_disp)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        
        
        #print ('my_coords : ',my_coords,'\n')
        for x, y, obj_id,object_meta in my_coords:
            for x1,y1,obj_id1,object_meta1 in my_coords:
                intersection_area=0
                total_area=0
                if (x==x1 and y==y1):  # skip if co-ordinates match (same object)
                    pass
                else:
                    this_pair = str(x)+','+str(y)+'and'+str(x1)+','+str(y1)   #this is to avoid repetition of re-doing already compared bboxes
                    other_pair = str(x1)+','+str(y1)+'and'+str(x)+','+str(y)
                    if this_pair not in parsed_list:
                        
                        ##heap map
                        if (f==0):
                            heatmap_data_0[int(y/100)][int(x/100)]+=1
                        if (f==1):
                            heatmap_data_1[int(y/100)][int(x/100)]+=1
                        if (f==2):
                            heatmap_data_2[int(y/100)][int(x/100)]+=1                            
                        if (f==3):
                            heatmap_data_3[int(y/100)][int(x/100)]+=1
                        
                        #print ('x =',x,'y =',y,'x1 =',x1,'y1 =',y1,'\n')
                        dist= ((x-x1)**2 + (y-y1)**2)**0.5    #distance calculation
                        bbox_ht = (object_meta.rect_params.height + object_meta1.rect_params.height)/2
                        scaling_factor = 5.5/bbox_ht
                        dist = dist * scaling_factor
                        
                        intersection_area= abs(object_meta.rect_params.left+object_meta.rect_params.width-object_meta1.rect_params.left) * abs(object_meta.rect_params.height+object_meta.rect_params.top-object_meta1.rect_params.top) 
                        total_area= abs(object_meta.rect_params.width * object_meta.rect_params.height)+(object_meta1.rect_params.width*object_meta1.rect_params.height) - intersection_area
                        #if (intersection_area/total_area) > 0:
                        #    intersection_area=0
                        #print('intersection_area-',intersection_area,'total_Area',total_area,'overlap area',abs(intersection_area/total_area),'\n')
                        #print('dist-',dist,'parsed_list',parsed_list,'scaling factor',scaling_factor,'\n')
                        if ((dist<3)): # or (abs(intersection_area/total_area)>0.8)): #3ft is violating threshold
                            #print(frame_number,dist*0.5 - 0.5*abs(intersection_area/total_area))
                        #if (dist*0.5 - 0.5*abs(intersection_area/total_area) <= 0.8):
                            x_array.append(x)
                            x1_array.append(x1)
                            y_array.append(y)
                            y1_array.append(y1)
                            violation_count += 1
                            if (obj_id,obj_id1) not in person_ids[f] and (obj_id1,obj_id) not in person_ids[f]:
                                tot_violations[f] += 1
                                person_ids[f].append((obj_id,obj_id1))
                                aggrgate_100_frame_viol[f]+=violation_count
   

                            unsafe_list.append(obj_id) 
                            unsafe_list.append(obj_id1)
                                                       
                            #print('violating Ids',obj_id,obj_id1)
                            #obj_col.set(1.0, 0.0, 0.0, 0.0)   #red
                            #obj_col1.set(1.0, 0.0, 0.0, 0.0)
                        #else:
                        #    obj_col.set(0.0, 1.0, 0.0, 0.0)    #green
                        #    obj_col1.set(0.0, 1.0, 0.0, 0.0)                        
                        
                        #print('dist- ',dist,'violation_count- ',violation_count,'\n')
                        parsed_list.append(this_pair)  #avoid comparison of elements (2,1) is elements (1,2) are done
                        parsed_list.append(other_pair)

        
        #print(person_ids,violation_count,tot_violations)
        unsafe_list = list(dict.fromkeys(unsafe_list))
        
        for i in obj_details:
            if i.object_id in unsafe_list:
                i.rect_params.border_color.set(1.0, 0.0, 0.0, 0.0)
                #i.text_params.font_params.font_color.set(1.0, 0.0, 0.0, 0.0)
                i.text_params.display_text= 'VIOLATION'
            else:
                i.rect_params.border_color.set(0.0, 1.0, 0.0, 0.0)
                i.text_params.display_text= 'SAFE'
                #i.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 0.0)
                
                
        ################## Setting Metadata Display configruation ############### 
        # Acquiring a display meta object.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        display_meta.num_lines = 1
        py_nvosd_line_params = display_meta.line_params  ## indicates pointer frame0 in that batch
        for i in range(len(x_array)):###changes for line drawing
            py_nvosd_line_params[i].x1 = int(x_array[i]);
            py_nvosd_line_params[i].y1 = int(y_array[i]);
            py_nvosd_line_params[i].x2 = int(x1_array[i]);
            py_nvosd_line_params[i].y2 = int(y1_array[i]);
            display_meta.num_lines+=1;
            py_nvosd_line_params[i].line_width = 6
            py_nvosd_line_params[i].line_color.set(1.0, 0.0, 0.0, 0.5)        
        

        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        #print(tot_violations)
        
        log_file = open(('output/jobs'+str(f)+'.log'),'a')
        log_file.write("Frame no. -"+str(frame_number)+" ; "+"            Violations count - " +str(tot_violations[f])+"\n\n\n")
        log_file.close()
        
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Person_count={} Frame_violation_count={} Total_violation_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_PERSON], violation_count,tot_violations[f])
        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # Set(red, green, blue, alpha); Set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # Set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string to print in notebook
        #print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        ############################################################################
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()


        if (frame_number%100==0):
            frame_plot[f].append(frame_number//100)
            frame_viol_count[f].append(aggrgate_100_frame_viol[f])

            plt.figure(figsize=(4,4))
            plt.title('Number of violations per 100 frames for feed '+str(f))
            plt.xlabel('Frame number/100')
            plt.ylabel('Num of violations')
            
            plt.plot(frame_plot[f], frame_viol_count[f],'b') 
            plt.savefig(('static/violation_count_image'+str(f)+'.png'))
            aggrgate_100_frame_viol[f]=0  

            
        if (frame_number%310 == 0):  #heatmap every 310 frames
            plt.figure()
            plt.title('Heatmap for feed '+str(f))
            if (f==0):
                ax = sns.heatmap(heatmap_data_0,cmap="YlGnBu")
                heatmap_data_0 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
            if (f==1):
                ax = sns.heatmap(heatmap_data_1,cmap="YlGnBu")
                heatmap_data_1 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
            if (f==2):
                ax = sns.heatmap(heatmap_data_2,cmap="YlGnBu")                
                heatmap_data_2 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
            if(f==3):
                ax = sns.heatmap(heatmap_data_3,cmap="YlGnBu")                
                heatmap_data_3 = np.zeros((11,20),dtype = "uint8")   ## for every frame make it 0
            
            plt.savefig(('static/heat'+str(f)+'.png'),dpi=50)        
        
            
        #if f== 3:   # when all 4 streams are available
            #h1 = cv2.imread('heat0.png')
            #h2 = cv2.imread('heat1.png')
            #h3 = cv2.imread('heat2.png')
            #h4 = cv2.imread('heat3.png')


            #hx1 = cv2.hconcat([h1,h2])
            #hx2 = cv2.hconcat([h3,h4])
            #hx = cv2.vconcat([hx1,hx2])
            #heatmap_vid.write(hx)
            
        #    print("Writing heat")

        f += 1
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
            
    
    ##Graph video generation
    #g1 = cv2.imread('violation_count_image0.png')
    #g2 = cv2.imread('violation_count_image1.png')
    #g3 = cv2.imread('violation_count_image2.png')
    #g4 = cv2.imread('violation_count_image3.png')
    
    
    #gx1 = cv2.hconcat([g1,g2])
    #gx2 = cv2.hconcat([g3,g4])
    #gx = cv2.vconcat([gx1,gx2])
    #graph.write(gx)
    #print("Writing graph")
    
    return Gst.PadProbeReturn.OK


# In[9]:


tiler_src_pad=tracker.get_static_pad("src")
if not tiler_src_pad:
    sys.stderr.write(" Unable to get src pad \n")
else:
    tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)


# Now with everything defined , we can start the playback and listen the events.

# In[10]:


# List the sources
print("Now playing...")
start_time = time.time()
print("Starting pipeline \n")
# start play back and listed to events		
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except:
    pass
# cleanup
print("Exiting app\n")
pipeline.set_state(Gst.State.NULL)
print("--- %s seconds ---" % (time.time() - start_time))

#graph.release()
#heatmap_vid.release()

