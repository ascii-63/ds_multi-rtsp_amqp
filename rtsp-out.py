#!/usr/bin/env python3

###################### CORRECT IMPORT ######################

# import sys
# sys.path.append('../')

# import gi
# import configparser

# gi.require_version('Gst', '1.0')
# from gi.repository import GLib, Gst


# from os import path
# import os.path
# import os
# import cv2
# import pyds
# import numpy as np
# from common.FPS import PERF_DATA
# from common.bus_call import bus_call
# from common.is_aarch_64 import is_aarch64
# from common.utils import long_to_uint64
# import platform
# import math
# import time
# from ctypes import *
# import shutil
# from datetime import datetime, timezone
# import pytz

############################################################

import sys
sys.path.append('../')

import gi
import configparser

gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst ,GstRtspServer


from os import path
import os.path
import os
import cv2
import pyds
import numpy as np
from common.FPS import PERF_DATA
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
from common.utils import long_to_uint64
import platform
import math
import time
from ctypes import *
import shutil
from datetime import datetime, timezone
import pytz

################################################################

perf_data = None

number_sources = None
is_display = False
frame_count = {}
saved_count = {}
global_frame_count = 0
global codec 
codec = "H264"
global bitrate 
bitrate = int(4000000)

pgie_classes_str = ["Person", "TwoWheeler", "Person", "RoadSign"]

PGIE_CLASS_ID_VEHICLE = 2
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1280   # Need modify
MUXER_OUTPUT_HEIGHT = 720  # Need modify
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = MUXER_OUTPUT_WIDTH
TILED_OUTPUT_HEIGHT = MUXER_OUTPUT_HEIGHT
SINK_QOS = 0
SINK_SYNC = 0
MSGCONV_SCHEMA_TYPE = 1  # 0 for Full, 1 for Minimal

MIN_CONFIDENCE = 0.5
MAX_CONFIDENCE = 1
MAX_DISPLAY_LEN = 64
MAX_TIME_STAMP_LEN = 32
STREAM_FPS = 24
MESSAGE_RATE = 1
FRAMES_PER_MESSAGE = STREAM_FPS * MESSAGE_RATE
GST_CAPS_FEATURES_NVMM = "memory:NVMM"

#uncomment for yolov5 model 
PGIE_CONFIG_FILE = "/home/ivsr/DeepStream-Yolo/config_infer_primary_yoloV5.txt"
#PGIE_CONFIG_FILE = "pgie_config.txt"

AMQP_CONFIG_FILE = "amqp_config.txt"
MSGCONV_CONFIG_FILE = "message_convert_config.txt"

FRAMES_DIR = "/home/ivsr/frames"

AMQP_LIB_FILE = "/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_amqp_proto.so"

################################################################


def meta_copy_func(data, user_data):
    """
    Callback function for deep-copying an NvDsEventMsgMeta struct
    """

    # Cast data to pyds.NvDsUserMeta
    user_meta = pyds.NvDsUserMeta.cast(data)
    src_meta_data = user_meta.user_meta_data
    # Cast src_meta_data to pyds.NvDsEventMsgMeta
    srcmeta = pyds.NvDsEventMsgMeta.cast(src_meta_data)

    # Duplicate the memory contents of srcmeta to dstmeta
    dstmeta_ptr = pyds.memdup(pyds.get_ptr(srcmeta),
                              sys.getsizeof(pyds.NvDsEventMsgMeta))

    # Cast the duplicated memory to pyds.NvDsEventMsgMeta
    dstmeta = pyds.NvDsEventMsgMeta.cast(dstmeta_ptr)

    # Duplicate contents of ts field
    dstmeta.ts = pyds.memdup(srcmeta.ts, MAX_TIME_STAMP_LEN + 1)

    # Copy the sensorStr
    dstmeta.sensorStr = pyds.get_string(srcmeta.sensorStr)

    if srcmeta.objSignature.size > 0:
        dstmeta.objSignature.signature = pyds.memdup(
            srcmeta.objSignature.signature, srcmeta.objSignature.size)
        dstmeta.objSignature.size = srcmeta.objSignature.size

    if srcmeta.extMsgSize > 0:
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_VEHICLE:
            srcobj = pyds.NvDsVehicleObject.cast(srcmeta.extMsg)
            obj = pyds.alloc_nvds_vehicle_object()
            obj.type = pyds.get_string(srcobj.type)
            obj.make = pyds.get_string(srcobj.make)
            obj.model = pyds.get_string(srcobj.model)
            obj.color = pyds.get_string(srcobj.color)
            obj.license = pyds.get_string(srcobj.license)
            obj.region = pyds.get_string(srcobj.region)
            dstmeta.extMsg = obj
            dstmeta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)

        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON:
            srcobj = pyds.NvDsPersonObject.cast(srcmeta.extMsg)
            obj = pyds.alloc_nvds_person_object()
            obj.age = srcobj.age
            obj.gender = pyds.get_string(srcobj.gender)
            obj.cap = pyds.get_string(srcobj.cap)
            obj.hair = pyds.get_string(srcobj.hair)
            obj.apparel = pyds.get_string(srcobj.apparel)
            dstmeta.extMsg = obj
            dstmeta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)

    return dstmeta


def meta_free_func(data, user_data):
    """
    Callback function for freeing an NvDsEventMsgMeta instance
    """

    user_meta = pyds.NvDsUserMeta.cast(data)
    srcmeta = pyds.NvDsEventMsgMeta.cast(user_meta.user_meta_data)

    # Free the memory
    pyds.free_buffer(srcmeta.ts)
    pyds.free_buffer(srcmeta.sensorStr)

    if srcmeta.objSignature.size > 0:
        pyds.free_buffer(srcmeta.objSignature.signature)
        srcmeta.objSignature.size = 0

    if srcmeta.extMsgSize > 0:
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_VEHICLE:
            obj = pyds.NvDsVehicleObject.cast(srcmeta.extMsg)
            pyds.free_buffer(obj.type)
            pyds.free_buffer(obj.color)
            pyds.free_buffer(obj.make)
            pyds.free_buffer(obj.model)
            pyds.free_buffer(obj.license)
            pyds.free_buffer(obj.region)

        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON:
            obj = pyds.NvDsPersonObject.cast(srcmeta.extMsg)
            pyds.free_buffer(obj.gender)
            pyds.free_buffer(obj.cap)
            pyds.free_buffer(obj.hair)
            pyds.free_buffer(obj.apparel)

        pyds.free_gbuffer(srcmeta.extMsg)
        srcmeta.extMsgSize = 0


def generate_person_meta(data):
    """
    Generate Person metadata
    """

    obj = pyds.NvDsPersonObject.cast(data)
    obj.age = 20
    obj.hair = "Black"
    obj.gender = "Male"
    return obj


def generate_event_msg_meta(data, class_id):
    """
    Generate event message metadata
    """

    meta = pyds.NvDsEventMsgMeta.cast(data)
    meta.sensorId = 0
    meta.placeId = 0
    meta.ts = pyds.alloc_buffer(MAX_TIME_STAMP_LEN + 1)
    pyds.generate_ts_rfc3339(meta.ts, MAX_TIME_STAMP_LEN)

    # Attach custom objects
    if class_id == PGIE_CLASS_ID_PERSON: # FIX ASAP
        meta.type = pyds.NvDsEventType.NVDS_EVENT_MOVING
        meta.objType = pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON
        meta.objClassId = PGIE_CLASS_ID_PERSON
        obj = pyds.alloc_nvds_person_object()
        obj = generate_person_meta(obj)
        meta.extMsg = obj
        meta.extMsgSize = sys.getsizeof(pyds.NvDsPersonObject)
    return meta


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Extract metadata received on OSD sink pad and
    update params for drawing rectangle, object information, etc...
    """

    global global_frame_count

    frame_number = 0
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,
        PGIE_CLASS_ID_ROADSIGN: 0
    }

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print(f"[ERROR] Unable to get GstBuffer \n")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Cast to pyds.NvDsFrameMeta
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            continue

        # print(f"Timestamp is {frame_meta.ntp_timestamp}") # Example
        # batch_id = frame_meta.batch_id
        # frame_number = frame_meta.frame_num
        # print(f"Frame ID: {frame_number}\nBatch ID: {batch_id}")
        frame_number = global_frame_count
        global_frame_count = global_frame_count + 1
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                continue

            # Update the object text display
            txt_params = obj_meta.text_params

            # Set display_text
            txt_params.display_text = pgie_classes_str[obj_meta.class_id]

            obj_counter[obj_meta.class_id] += 1

            # Font, font-color, font-size and text background color
            txt_params.font_params.font_name = "Serif"
            txt_params.font_params.font_size = 10
            txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

            confidence = obj_meta.confidence
            class_id = obj_meta.class_id
	    
            # if (MIN_CONFIDENCE < confidence < MAX_CONFIDENCE) and (frame_number % FRAMES_PER_MESSAGE == 0):
            if (MIN_CONFIDENCE < confidence < MAX_CONFIDENCE):
                # Message is being sent for Person object with confidence in range (MIN_CONFIDENCE, MAX_CONFIDENCE), after FRAMES_PER_MESSAGE frames

                # Allocating an NvDsEventMsgMeta instance and getting reference to it
                msg_meta = pyds.alloc_nvds_event_msg_meta()

                msg_meta.bbox.top = obj_meta.rect_params.top
                msg_meta.bbox.left = obj_meta.rect_params.left
                msg_meta.bbox.width = obj_meta.rect_params.width
                msg_meta.bbox.height = obj_meta.rect_params.height
                msg_meta.frameId = frame_number
                msg_meta.trackingId = long_to_uint64(obj_meta.object_id)
                msg_meta.confidence = obj_meta.confidence
                msg_meta = generate_event_msg_meta(msg_meta, obj_meta.class_id)

                user_event_meta = pyds.nvds_acquire_user_meta_from_pool(
                    batch_meta)

                if user_event_meta:
                    user_event_meta.user_meta_data = msg_meta
                    user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META

                    # Setting callbacks in the event msg meta
                    pyds.user_copyfunc(user_event_meta, meta_copy_func)
                    pyds.user_releasefunc(user_event_meta, meta_free_func)
                    pyds.nvds_add_user_meta_to_frame(frame_meta,
                                                     user_event_meta)
                else:
                    print("[ERROR] in attaching event meta to buffer\n")  

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    print("Frame Number =", frame_number, "Vehicle Count =",
          obj_counter[PGIE_CLASS_ID_VEHICLE], "Person Count =",
          obj_counter[PGIE_CLASS_ID_PERSON])
    return Gst.PadProbeReturn.OK


################################################################

def cb_newpad(decodebin, _decoder_src_pad, _data):
    """
    New pad callback function
    """

    caps = _decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = _data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(_decoder_src_pad):
                sys.stderr.write(
                    "[ERROR] Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(
                "[ERROR] Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(_child_proxy, _object, _name, _user_data):
    """
    Add child decodebin
    """

    print("[ INFO] Decodebin child added:", _name, "\n")
    if _name.find("decodebin") != -1:
        _object.connect("child-added", decodebin_child_added, _user_data)

    if "source" in _name:
        source_element = _child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            _object.set_property("drop-on-latency", True)


def create_source_bin(_index, _uri):
    """
    Create sourcebin from a RTSP URI
    """

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % _index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("[ERROR] Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("[ERROR] Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", _uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("[ERROR] Failed to add ghost pad in source bin \n")
        return None
    return nbin


def draw_bounding_boxes(_image, _obj_meta, _confidence):
    """
    Draw bounding boxes
    """

    _confidence = '{0:.2f}'.format(_confidence)
    rect_params = _obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[_obj_meta.class_id]
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)

    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    _image = cv2.line(_image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    _image = cv2.line(_image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    _image = cv2.line(_image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    _image = cv2.line(_image, lineright_c1, lineright_c2, color, 6)
    _image = cv2.putText(_image, obj_name + ',C=' + str(_confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                         (0, 0, 255, 0), 2)
    return _image


def convert_timestamp(timestamp):
    """
    Convert timestamp from Unix epoch to human-readable format
    """

    # Convert the UTC datetime to UTC+7
    new_timestamp = timestamp + (7 * 60 * 60 * 10**9)

    seconds = int(new_timestamp) // 10**9
    milliseconds = (int(new_timestamp) % 10**9) // 10**6

    # Create a datetime object in UTC+7
    dt_object = datetime.utcfromtimestamp(seconds)

    formatted_timestamp = dt_object.strftime('%Y-%m-%dT%H:%M:%S.') + f"{milliseconds:03d}Z"

    return formatted_timestamp


def tiler_sink_pad_buffer_probe(_pad, _info, _u_data):
    """
    Extract metadata received on tiler src pad
    and update params for drawing rectangle, object information etc.
    """

    frame_number = 0
    num_rects = 0
    global global_frame_count

    gst_buffer = _info.get_buffer()
    if not gst_buffer:
        print("[ERROR] Unable to get GstBuffer")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # l_frame.data needs a cast to pyds.NvDsFrameMeta
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }

        timestamp = frame_meta.ntp_timestamp

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1

            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, annotate the frame with bboxes and confidence value.
            # Save the annotated frame to file.
            # if global_frame_count % FRAMES_PER_MESSAGE == 0 and (MIN_CONFIDENCE < obj_meta.confidence < MAX_CONFIDENCE):
            if saved_count["stream_{}".format(frame_meta.pad_index)] % FRAMES_PER_MESSAGE == 0 and (  # Each FRAMES_PER_MESSAGE frames get 1 image
                    MIN_CONFIDENCE < obj_meta.confidence < MAX_CONFIDENCE):
                if is_first_obj:
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    n_frame = pyds.get_nvds_buf_surface(
                        hash(gst_buffer), frame_meta.batch_id)
                    n_frame = draw_bounding_boxes(
                        n_frame, obj_meta, obj_meta.confidence)
                    # convert python array into numpy array format in the copy mode.
                    frame_copy = np.array(n_frame, copy=True, order='C')
                    # convert the array into cv2 default color format
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    # Unmapped the buffer from CPU
                    if is_aarch64(): # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                            #  The original array cannot be accessed after this call.
                    

                save_image = True

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        if save_image:
            img_path = "{}/{}.jpg".format(FRAMES_DIR, convert_timestamp(timestamp))
            cv2.imwrite(img_path, frame_copy)
        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def args_parser(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write(
            "Usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        return False

    global perf_data
    global number_sources
    perf_data = PERF_DATA(len(args) - 1)
    number_sources = len(args) - 1

    return True


def main(args):
    global perf_data
    global number_sources
    global is_display
    global frame_count
    global saved_count

    #############################

    # Registering callbacks
    pyds.register_user_copyfunc(meta_copy_func)
    pyds.register_user_releasefunc(meta_free_func)

    # Standard GStreamer initialization
    Gst.init(None)

    #############################

    # Create Pipeline element that will form a connection of other elements
    print("[ INFO] Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    is_live = False
    if not pipeline:
        sys.stderr.write("[ERROR] Unable to create Pipeline\n")

    #############################

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("[ERROR] Unable to create NvStreamMux \n")

    pipeline.add(streammux)

    for i in range(number_sources):
        os.mkdir(FRAMES_DIR + "/stream_" + str(i))
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("[ INFO] Creating source_bin ", i, " \n ")

        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True

        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("[ERROR] Unable to create source bin\n")
        pipeline.add(source_bin)

        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("[ERROR] Unable to create sink pad bin \n")

        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("[ERROR] Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    #############################

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("[ERROR] Unable to create pgie \n")

    #############################

    # Add nvvidconv1 and filter1 to convert the frames to RGBA, which is easier to work with in Python.
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write("[ERROR] Unable to create nvvidconv1 \n")

    # caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    # filter2 = Gst.ElementFactory.make("capsfilter", "filter2")
    # if not filter2:
    #     sys.stderr.write("[ERROR] Unable to get the caps filter2 \n")
    # filter2.set_property("caps", caps2)

    caps2 = Gst.ElementFactory.make("capsfilter", "filter2")
    #filter2 = Gst.ElementFactory.make("capsfilter", "filter2")
    caps2.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )

    #############################

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("[ERROR] Unable to create tiler \n")

    #############################

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    if not nvvidconv2:
        sys.stderr.write("[ERROR] Unable to create nvvidconv2 \n")

    #############################

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    #############################

    msgconv = Gst.ElementFactory.make("nvmsgconv", "nvmsg-converter")
    if not msgconv:
        sys.stderr.write("[ERROR] Unable to create msgconv \n")

    #############################

    msgbroker = Gst.ElementFactory.make("nvmsgbroker", "nvmsg-broker")
    if not msgbroker:
        sys.stderr.write(" Unable to create msgbroker \n")

    #############################

    # Create tee and 2 queues
    tee = Gst.ElementFactory.make("tee", "nvsink-tee")
    if not tee:
        sys.stderr.write("[ERROR] Unable to create tee \n")

    queue1 = Gst.ElementFactory.make("queue", "nvtee-que1")
    if not queue1:
        sys.stderr.write("[ERROR] Unable to create queue1 \n")

    queue2 = Gst.ElementFactory.make("queue", "nvtee-que2")
    if not queue2:
        sys.stderr.write("[ERROR] Unable to create queue2 \n")
    #############################


    #video convert for streaming

    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")



    #############################
   
    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if is_aarch64():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        #encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    #############################

    if is_display:
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
    ##############################################
    #Set propertie UDP sink for rtsp out 
    else:
        # sink = Gst.ElementFactory.make("fakesink", "fakesink")
        # if not sink:
        #     sys.stderr.write(" Unable to create fakesink \n")
        updsink_port_num = 5400
        sink = Gst.ElementFactory.make("udpsink", "udpsink")

        sink.set_property("host", "224.224.255.255")
        sink.set_property("port", updsink_port_num)
        sink.set_property("async", False)
        sink.set_property("sync", 1)

    #############################

    if is_live:
        print("[INFO] Atleast one of the rtsp video sources is live \n")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    #############################

    pgie.set_property('config-file-path', PGIE_CONFIG_FILE)
    pgie_batch_size = pgie.get_property("batch-size")

    if (pgie_batch_size != number_sources):
        print(
            f"[ WARN] Overriding infer-config batch-size {pgie_batch_size}with number of sources {number_sources}\n")
        pgie.set_property("batch-size", number_sources)

    #############################

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    #############################

    msgconv.set_property('config', MSGCONV_CONFIG_FILE)
    msgconv.set_property('payload-type', MSGCONV_SCHEMA_TYPE)

    #############################

    msgbroker.set_property('config', AMQP_CONFIG_FILE)
    msgbroker.set_property('proto-lib', AMQP_LIB_FILE)

    #############################

    sink.set_property("sync", SINK_SYNC)
    sink.set_property("qos", SINK_QOS)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
        
    #############################

    print("[ INFO] Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    #pipeline.add(filter2)
    pipeline.add(tiler)
    pipeline.add(nvvidconv2)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(msgconv)
    pipeline.add(msgbroker)
    pipeline.add(tee)
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(caps2)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    #############################

    print("[ INFO] Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(caps2)
    #nvvidconv1.link(tiler)
    caps2.link(tiler)
    tiler.link(nvvidconv2)
    nvvidconv2.link(nvosd)
    nvosd.link(tee)
    queue1.link(msgconv)
    msgconv.link(msgbroker)
    queue2.link(encoder)
    # queue2.link(nvvidconv_postosd)
    # nvvidconv_postosd.link(caps2)
    # caps2.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)
    
    

    sink_pad = queue1.get_static_pad("sink")
    tee_msg_pad = tee.get_request_pad('src_%u')
    tee_render_pad = tee.get_request_pad("src_%u")
    if not tee_msg_pad or not tee_render_pad:
        sys.stderr.write("[ERROR] Unable to get request pads\n")
    tee_msg_pad.link(sink_pad)
    sink_pad = queue2.get_static_pad("sink")
    tee_render_pad.link(sink_pad)

    #############################

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    #############################
    
    #start steaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
        % rtsp_port_num
    )

    #############################

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write("[ERROR] Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    #############################

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER,
                             osd_sink_pad_buffer_probe, 0)

    #############################

    # List the sources
    print("[ INFO] Now playing... \n")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)

    #############################

    print("[ INFO] Starting pipeline... \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    #############################

    # cleanup
    print("[INFO] Exiting app...\n")
    pipeline.set_state(Gst.State.NULL)


def cleanup():
    """
    Remove old frames directory and create a new one
    """

    try:
        shutil.rmtree(FRAMES_DIR)
    except OSError as e:
        print(f"[ERROR] {e}")

    try:
        os.mkdir(FRAMES_DIR)
    except OSError as e:
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    cleanup()
    ret = args_parser(sys.argv)
    if not ret:
        sys.exit(1)
    else:
        sys.exit(main(sys.argv))
