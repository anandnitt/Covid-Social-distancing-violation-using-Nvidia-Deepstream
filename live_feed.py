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

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import configparser
import pyds

PGIE_CLASS_ID_FACE = 1
PGIE_CLASS_ID_BAG = 2
PGIE_CLASS_ID_PERSON = 0


tot_violations = [0,0,0,0,0,0,0]
person_ids = [[],[],[],[],[],[],[]]

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_FACE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BAG:0,
    }
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
        
        log_file = open(('output/jobs'+str(f)+'.log'),'a')
        log_file.write("Frame no. -"+str(frame_number)+" ; "+"            Violations count - " +str(tot_violations[f])+"\n\n\n")
        log_file.close()


        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        #print(tot_violations)
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
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        #fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        f += 1
        ############################################################################
        # Get frame rate through this probe
        #fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    INPUT_VIDEO_1 = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        sys.exit(1)

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
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)


    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")


    tracker = Gst.ElementFactory.make("nvtracker", "tracker")

    source1 = Gst.ElementFactory.make("filesrc", "file-source-1")
# Since the data format in the input file is elementary h264 stream,we need a h264parser
    h264parser1 = Gst.ElementFactory.make("h264parse", "h264-parser-1")
# Use nvdec_h264 for hardware accelerated decode on GPU
    decoder1 = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder-1")

    source1.set_property('location', INPUT_VIDEO_1)

    tiler= Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")

    tiler_rows=int(1)
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



    print("Playing cam")
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', '/dev/video0')
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "config/config_infer_primary_peoplenet.txt")
    #pgie.set_property('config-file-path', "N1/detectnetv2.txt")
    
    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)

    pipeline.add(source1)
    pipeline.add(h264parser1)
    pipeline.add(decoder1)

    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(tiler)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    source1.link(h264parser1)
    h264parser1.link(decoder1)


    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)


    sinkpad1 = streammux.get_request_pad("sink_1")
    if not sinkpad1:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad1 = decoder1.get_static_pad("src")
    if not srcpad1:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad1.link(sinkpad1)




    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

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

if __name__ == '__main__':
    sys.exit(main(sys.argv))
