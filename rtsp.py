#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import time
import argparse
import sys
sys.path.append('../')
import configparser
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import numpy as np
import pyds

PGIE_CLASS_ID_FACE=2
PGIE_CLASS_ID_PERSON=0
PGIE_CLASS_ID_BAG=1


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

def osd_sink_pad_buffer_probe(pad,info,u_data):
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
        #for i in range(len(x_array)):###changes for line drawing
        #    py_nvosd_line_params[i].x1 = int(x_array[i]);
        #    py_nvosd_line_params[i].y1 = int(y_array[i]);
        #    py_nvosd_line_params[i].x2 = int(x1_array[i]);
        #    py_nvosd_line_params[i].y2 = int(y1_array[i]);
        #    display_meta.num_lines+=1;
        #    py_nvosd_line_params[i].line_width = 6
        #    py_nvosd_line_params[i].line_color.set(1.0, 0.0, 0.0, 0.5)        
        

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


def main(args):
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")
    
    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")
    
    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")
    

    source2 = Gst.ElementFactory.make("filesrc", "file-source2")
    h264parser2 = Gst.ElementFactory.make("h264parse", "h264-parser2")
    decoder2 = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder2")

    source3 = Gst.ElementFactory.make("filesrc", "file-source3")
    h264parser3 = Gst.ElementFactory.make("h264parse", "h264-parser3")
    decoder3 = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder3")

    source4 = Gst.ElementFactory.make("filesrc", "file-source4")
    h264parser4 = Gst.ElementFactory.make("h264parse", "h264-parser4")
    decoder4 = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder4")


    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    
    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)
    
    print("Playing file %s " %stream_path)
    source.set_property('location', stream_path)
    source2.set_property('location', stream_path)
    source3.set_property('location', stream_path)
    source4.set_property('location', stream_path)
    

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 4)
    streammux.set_property('batched-push-timeout', 4000000)
    
    pgie.set_property('config-file-path', "config/config_infer_primary_peoplenet.txt")
  
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler_rows=int(2)
    tiler_columns=int(2)
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", 1920)
    tiler.set_property("height", 1080)

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

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)

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
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Link the elements together:
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> nvvidconv_postosd -> 
    # caps -> encoder -> rtppay -> udpsink
    
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    
    srcpad.link(sinkpad)


    print("Linking elements in the Pipeline \n")
    source2.link(h264parser2)
    h264parser2.link(decoder2)
    sinkpad2 = streammux.get_request_pad("sink_2")
    if not sinkpad2:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    srcpad2 = decoder2.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    
    srcpad2.link(sinkpad2)

    print("Linking elements in the Pipeline \n")
    source3.link(h264parser3)
    h264parser3.link(decoder3)
    sinkpad3 = streammux.get_request_pad("sink_3")
    if not sinkpad3:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    srcpad3 = decoder3.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    
    srcpad3.link(sinkpad3)

    print("Linking elements in the Pipeline \n")
    source4.link(h264parser4)
    h264parser4.link(decoder4)
    sinkpad4 = streammux.get_request_pad("sink_4")
    if not sinkpad4:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    srcpad4 = decoder4.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    
    srcpad4.link(sinkpad4)


    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)
    
    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # Start streaming
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)
    
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def parse_args():

    global codec
    global bitrate
    global stream_path
    codec = 'H264'
    bitrate = 4000000
    stream_path = 'sources/few_people.h264'
    return 0

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))