#!/usr/bin/env python3
import os
import hashlib
import subprocess
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import imghdr
import io
import logging
import multiprocessing
import random
import time
from pathlib import Path
from flask import Flask, render_template
import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from lama_cleaner.const import SD15_MODELS
from lama_cleaner.file_manager import FileManager
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.plugins import (
    InteractiveSeg,
    RemoveBG,
    RealESRGANUpscaler,
    MakeGIF,
    GFPGANPlugin,
    RestoreFormerPlugin,
    AnimeSeg,
)
from lama_cleaner.schema import Config

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from flask import (
    Flask,
    request,
    send_file,
    cli,
    make_response,
    send_from_directory,
    jsonify,
)
from flask_socketio import SocketIO

# Disable ability for Flask to display warning about using a development server in a production environment.
# https://gist.github.com/jerblack/735b9953ba1ab6234abb43174210d356
cli.show_server_banner = lambda *_: None
from flask_cors import CORS

from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
)


import numpy as np
# import mediapipe as mp

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import face_recognition
import cv2
import numpy as np
# def hair_detect(img):
#   BG_COLOR = (0, 0, 0) # gray
#   MASK_COLOR = (255, 255, 255) # white


#   # Create the options that will be used for ImageSegmenter
#   base_options = python.BaseOptions(model_asset_path='lama-cleaner/hair_segmenter.tflite')
#   options = vision.ImageSegmenterOptions(base_options=base_options,
#                                         output_category_mask=True)

#   # Create the image segmenter
#   with vision.ImageSegmenter.create_from_options(options) as segmenter:

#     # Loop through demo image(s)
  

#     # Retrieve the masks for the segmented image
#     new_img =  mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#     segmentation_result = segmenter.segment(new_img)
    
#     category_mask = segmentation_result.category_mask

#     # Generate solid color images for showing the output segmentation mask.
#     image_data = new_img.numpy_view()
#     fg_image = np.zeros(image_data.shape, dtype=np.uint8)
#     fg_image[:] = MASK_COLOR
#     bg_image = np.zeros(image_data.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR

#     condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
#     output_image = np.where(condition, fg_image, bg_image)
    
#     kernel = np.ones((5, 5), np.uint8) 
  

#     output_image = cv2.dilate(output_image, kernel, iterations=2) 
#     return output_image

# def create_mask_face(frame):
  
#     face_locations = []
#     rgb_small_frame = frame[:, :, ::-1]

#     face_locations = face_recognition.face_locations(rgb_small_frame)

#     mask = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
#     for (top, right, bottom, left) in face_locations:
#         top =max(0,top-int(frame.shape[0]/10))
#         bottom =min(frame.shape[1],bottom+int(frame.shape[1]/10))
#         left =max(0,left-int(frame.shape[1]/10))
#         right =min(frame.shape[0],right+int(frame.shape[0]/10))

#         mask[top:bottom,left:right]=255
    
#     hair_mask = hair_detect(frame)
#     hair_mask = cv2.cvtColor(hair_mask,cv2.COLOR_BGR2GRAY)
#     print('--------')
#     print(hair_mask.shape)
#     print(mask.shape)
#     mask=cv2.add(hair_mask,mask)
#     return mask
import cv2

def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers
def create_mask_person(image):

    cv2.imwrite('image.png',image)
    subprocess.run('python -W ignore u2net_test.py', shell=True, check=True)
    thresh = cv2.imread('results/image.png')
   
    return thresh

NUM_THREADS = str(multiprocessing.cpu_count())

# fix libomp problem on windows https://github.com/Sanster/lama-cleaner/issues/56
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")


class NoFlaskwebgui(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Running on http:" in msg:
            print(msg[msg.index("Running on http:") :])

        return (
            "flaskwebgui-keep-server-alive" not in msg
            and "socket.io" not in msg
            and "This is a development server." not in msg
        )


logging.getLogger("werkzeug").addFilter(NoFlaskwebgui())

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app, expose_headers=["Content-Disposition"])

sio_logger = logging.getLogger("sio-logger")
sio_logger.setLevel(logging.ERROR)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model: ModelManager = None
thumb: FileManager = None
output_dir: str = None
device = None
input_image_path: str = None
is_disable_model_switch: bool = False
is_controlnet: bool = False
controlnet_method: str = "control_v11p_sd15_canny"
is_enable_file_manager: bool = False
is_enable_auto_saving: bool = False
is_desktop: bool = False
image_quality: int = 95
plugins = {}


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def diffuser_callback(i, t, latents):
    socketio.emit("diffusion_progress", {"step": i})


@app.route("/save_image", methods=["POST"])
def save_image():
    if output_dir is None:
        return "--output-dir is None", 500

    input = request.files
    filename = request.form["filename"]
    origin_image_bytes = input["image"].read()  # RGB
    ext = get_image_ext(origin_image_bytes)
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)
    save_path = os.path.join(output_dir, filename)

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != image.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(image.shape[1], image.shape[0])
            )
        image = np.concatenate((image, alpha_channel[:, :, np.newaxis]), axis=-1)

    pil_image = Image.fromarray(image)

    img_bytes = pil_to_bytes(
        pil_image,
        ext,
        quality=image_quality,
        exif_infos=exif_infos,
    )
    with open(save_path, "wb") as fw:
        fw.write(img_bytes)

    return "ok", 200


@app.route("/medias/<tab>")
def medias(tab):
    if tab == "image":
        response = make_response(jsonify(thumb.media_names), 200)
    else:
        response = make_response(jsonify(thumb.output_media_names), 200)
    # response.last_modified = thumb.modified_time[tab]
    # response.cache_control.no_cache = True
    # response.cache_control.max_age = 0
    # response.make_conditional(request)
    return response


@app.route("/media/<tab>/<filename>")
def media_file(tab, filename):
    if tab == "image":
        return send_from_directory(thumb.root_directory, filename)
    return send_from_directory(thumb.output_dir, filename)


@app.route("/media_thumbnail/<tab>/<filename>")
def media_thumbnail_file(tab, filename):
    args = request.args
    width = args.get("width")
    height = args.get("height")
    if width is None and height is None:
        width = 256
    if width:
        width = int(float(width))
    if height:
        height = int(float(height))

    directory = thumb.root_directory
    if tab == "output":
        directory = thumb.output_dir
    thumb_filename, (width, height) = thumb.get_thumbnail(
        directory, filename, width, height
    )
    thumb_filepath = f"{app.config['THUMBNAIL_MEDIA_THUMBNAIL_ROOT']}{thumb_filename}"

    response = make_response(send_file(thumb_filepath))
    response.headers["X-Width"] = str(width)
    response.headers["X-Height"] = str(height)
    return response


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    # RGB
    origin_image_bytes = input["image"].read()
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)

    # mask, _ = load_img(input["mask"].read(), gray=True)
    
    mask = create_mask_person(image)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if image.shape[:2] != mask.shape[:2]:
        return (
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
            400,
        )

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    form = request.form
    size_limit = max(image.shape)

    if "paintByExampleImage" in input:
        paint_by_example_example_image, _ = load_img(
            input["paintByExampleImage"].read()
        )
        paint_by_example_example_image = Image.fromarray(paint_by_example_example_image)
    else:
        paint_by_example_example_image = None

    config = Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt="A bald head looking straight at camera",
        # prompt=form["prompt"],
        negative_prompt=form["negativePrompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        sd_scale=form["sdScale"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
        sd_match_histograms=form["sdMatchHistograms"],
        cv2_flag=form["cv2Flag"],
        cv2_radius=form["cv2Radius"],
        paint_by_example_steps=form["paintByExampleSteps"],
        paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
        paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
        paint_by_example_seed=form["paintByExampleSeed"],
        paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_steps=form["p2pSteps"],
        p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
        p2p_guidance_scale=form["p2pGuidanceScale"],
        controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
        controlnet_method=form["controlnet_method"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    start = time.time()
    try:
        res_np_img = model(image, mask, config)
    except RuntimeError as e:
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            logger.exception(e)
            return f"{str(e)}", 500
    finally:
        logger.info(f"process time: {(time.time() - start) * 1000}ms")
        torch_gc()

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    bytes_io = io.BytesIO(
        pil_to_bytes(
            Image.fromarray(res_np_img),
            ext,
            quality=image_quality,
            exif_infos=exif_infos,
        )
    )

    response = make_response(
        send_file(
            # io.BytesIO(numpy_to_bytes(res_np_img, ext)),
            bytes_io,
            mimetype=f"image/{ext}",
        )
    )
    response.headers["X-Seed"] = str(config.sd_seed)

    socketio.emit("diffusion_finish")
    return response


@app.route("/run_plugin", methods=["POST"])
def run_plugin():
    form = request.form
    files = request.files
    name = form["name"]
    if name not in plugins:
        return "Plugin not found", 500

    origin_image_bytes = files["image"].read()  # RGB
    rgb_np_img, alpha_channel, exif_infos = load_img(
        origin_image_bytes, return_exif=True
    )

    start = time.time()
    try:
        form = dict(form)
        if name == InteractiveSeg.name:
            img_md5 = hashlib.md5(origin_image_bytes).hexdigest()
            form["img_md5"] = img_md5
        bgr_res = plugins[name](rgb_np_img, files, form)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            logger.exception(e)
            return "Internal Server Error", 500

    logger.info(f"{name} process time: {(time.time() - start) * 1000}ms")
    torch_gc()

    if name == MakeGIF.name:
        return send_file(
            io.BytesIO(bgr_res),
            mimetype="image/gif",
            as_attachment=True,
            download_name=form["filename"],
        )
    if name == InteractiveSeg.name:
        return make_response(
            send_file(
                io.BytesIO(numpy_to_bytes(bgr_res, "png")),
                mimetype="image/png",
            )
        )

    if name in [RemoveBG.name, AnimeSeg.name]:
        rgb_res = bgr_res
        ext = "png"
    else:
        rgb_res = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2RGB)
        ext = get_image_ext(origin_image_bytes)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != rgb_res.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(rgb_res.shape[1], rgb_res.shape[0])
                )
            rgb_res = np.concatenate(
                (rgb_res, alpha_channel[:, :, np.newaxis]), axis=-1
            )

    response = make_response(
        send_file(
            io.BytesIO(
                pil_to_bytes(
                    Image.fromarray(rgb_res),
                    ext,
                    quality=image_quality,
                    exif_infos=exif_infos,
                )
            ),
            mimetype=f"image/{ext}",
        )
    )
    return response


@app.route("/server_config", methods=["GET"])
def get_server_config():
    return {
        "isControlNet": is_controlnet,
        "controlNetMethod": controlnet_method,
        "isDisableModelSwitchState": is_disable_model_switch,
        "isEnableAutoSaving": is_enable_auto_saving,
        "enableFileManager": is_enable_file_manager,
        "plugins": list(plugins.keys()),
    }, 200


@app.route("/model")
def current_model():
    return model.name, 200


@app.route("/model_downloaded/<name>")
def model_downloaded(name):
    return str(model.is_downloaded(name)), 200


@app.route("/is_desktop")
def get_is_desktop():
    return str(is_desktop), 200


@app.route("/model", methods=["POST"])
def switch_model():
    if is_disable_model_switch:
        return "Switch model is disabled", 400

    new_name = request.form.get("name")
    if new_name == model.name:
        return "Same model", 200

    try:
        model.switch(new_name)
    except NotImplementedError:
        return f"{new_name} not implemented", 403
    return f"ok, switch to {new_name}", 200


# @app.route("/")
# def index():
#     return send_file(os.path.join(BUILD_DIR, "index.html"))
@app.route("/", methods=['POST','GET'])
def index():
    if request.method == 'POST':
        input = request.files
        # RGB
        origin_image_bytes = input["image"].read()
        image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)

        # mask, _ = load_img(input["mask"].read(), gray=True)
        
        mask = create_mask_person(image)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            return (
                f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
                400,
            )

        original_shape = image.shape
        interpolation = cv2.INTER_CUBIC

        # form = request.form
        form = {'ldmSteps': '25',
         'ldmSampler': 'plms',
          'zitsWireframe': 'true',
           'hdStrategy': 'Original',
            'hdStrategyCropMargin': '128',
             'hdStrategyCropTrigerSize': '512',
              'hdStrategyResizeLimit': '768',
               'prompt': 'A bald head looking straight at camer',
                'negativePrompt': '', 
                'croperX': '56',
                 'croperY': '160',
                  'croperHeight': '512',
                   'croperWidth': '512',
                    'useCroper': 'false',
                     'sdMaskBlur': '5',
                      'sdStrength': '0.75',
                       'sdSteps': '50',
                        'sdGuidanceScale': '7.5',
                         'sdSampler': 'uni_pc',
                          'sdSeed': '-1',
                           'sdMatchHistograms': 'false',
                            'sdScale': '1',
                             'cv2Radius': '5',
                              'cv2Flag': 'INPAINT_NS',
                               'paintByExampleSteps': '50',
                                'paintByExampleGuidanceScale': '7.5',
                                 'paintByExampleSeed': '-1',
                                  'paintByExampleMaskBlur': '5',
                                   'paintByExampleMatchHistograms': 'false',
                                    'p2pSteps': '50',
                                     'p2pImageGuidanceScale': '1.5',
                                      'p2pGuidanceScale': '7.5',
                                       'controlnet_conditioning_scale': '0.4',
                                        'controlnet_method': 'control_v11p_sd15_canny'}
        size_limit = max(image.shape)
        print(form)
        print(form["ldmSteps"])
        if "paintByExampleImage" in input:
            paint_by_example_example_image, _ = load_img(
                input["paintByExampleImage"].read()
            )
            paint_by_example_example_image = Image.fromarray(paint_by_example_example_image)
        else:
            paint_by_example_example_image = None

        config = Config(
            ldm_steps=form["ldmSteps"],
            ldm_sampler=form["ldmSampler"],
            hd_strategy=form["hdStrategy"],
            zits_wireframe=form["zitsWireframe"],
            hd_strategy_crop_margin=form["hdStrategyCropMargin"],
            hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
            hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
            prompt="A bald head looking straight at camera",
            # prompt=form["prompt"],
            negative_prompt=form["negativePrompt"],
            use_croper=form["useCroper"],
            croper_x=form["croperX"],
            croper_y=form["croperY"],
            croper_height=form["croperHeight"],
            croper_width=form["croperWidth"],
            sd_scale=form["sdScale"],
            sd_mask_blur=form["sdMaskBlur"],
            sd_strength=form["sdStrength"],
            sd_steps=form["sdSteps"],
            sd_guidance_scale=form["sdGuidanceScale"],
            sd_sampler=form["sdSampler"],
            sd_seed=form["sdSeed"],
            sd_match_histograms=form["sdMatchHistograms"],
            cv2_flag=form["cv2Flag"],
            cv2_radius=form["cv2Radius"],
            paint_by_example_steps=form["paintByExampleSteps"],
            paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
            paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
            paint_by_example_seed=form["paintByExampleSeed"],
            paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
            paint_by_example_example_image=paint_by_example_example_image,
            p2p_steps=form["p2pSteps"],
            p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
            p2p_guidance_scale=form["p2pGuidanceScale"],
            controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
            controlnet_method=form["controlnet_method"],
        )

        if config.sd_seed == -1:
            config.sd_seed = random.randint(1, 999999999)
        if config.paint_by_example_seed == -1:
            config.paint_by_example_seed = random.randint(1, 999999999)

        logger.info(f"Origin image shape: {original_shape}")
        image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

        mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

        start = time.time()
        try:
            res_np_img = model(image, mask, config)
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                # NOTE: the string may change?
                return "CUDA out of memory", 500
            else:
                logger.exception(e)
                return f"{str(e)}", 500
        finally:
            logger.info(f"process time: {(time.time() - start) * 1000}ms")
            torch_gc()

        res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
                )
            res_np_img = np.concatenate(
                (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
            )

        ext = get_image_ext(origin_image_bytes)

        bytes_io = io.BytesIO(
            pil_to_bytes(
                Image.fromarray(res_np_img),
                ext,
                quality=image_quality,
                exif_infos=exif_infos,
            )
        )

        response = make_response(
            send_file(
                # io.BytesIO(numpy_to_bytes(res_np_img, ext)),
                bytes_io,
                mimetype=f"image/{ext}",
            )
        )
        response.headers["X-Seed"] = str(config.sd_seed)

        socketio.emit("diffusion_finish")
        return response
    else:
        return send_file(os.path.join(BUILD_DIR, "new_index.html"))
@app.route("/inputimage")
def set_input_photo():
    if input_image_path:
        with open(input_image_path, "rb") as f:
            image_in_bytes = f.read()
        return send_file(
            input_image_path,
            as_attachment=True,
            download_name=Path(input_image_path).name,
            mimetype=f"image/{get_image_ext(image_in_bytes)}",
        )
    else:
        return "No Input Image"


def build_plugins(args):
    global plugins
    if args.enable_interactive_seg:
        logger.info(f"Initialize {InteractiveSeg.name} plugin")
        plugins[InteractiveSeg.name] = InteractiveSeg(
            args.interactive_seg_model, args.interactive_seg_device
        )

    if args.enable_remove_bg:
        logger.info(f"Initialize {RemoveBG.name} plugin")
        plugins[RemoveBG.name] = RemoveBG()

    if args.enable_anime_seg:
        logger.info(f"Initialize {AnimeSeg.name} plugin")
        plugins[AnimeSeg.name] = AnimeSeg()

    if args.enable_realesrgan:
        logger.info(
            f"Initialize {RealESRGANUpscaler.name} plugin: {args.realesrgan_model}, {args.realesrgan_device}"
        )
        plugins[RealESRGANUpscaler.name] = RealESRGANUpscaler(
            args.realesrgan_model,
            args.realesrgan_device,
            no_half=args.realesrgan_no_half,
        )

    if args.enable_gfpgan:
        logger.info(f"Initialize {GFPGANPlugin.name} plugin")
        if args.enable_realesrgan:
            logger.info("Use realesrgan as GFPGAN background upscaler")
        else:
            logger.info(
                f"GFPGAN no background upscaler, use --enable-realesrgan to enable it"
            )
        plugins[GFPGANPlugin.name] = GFPGANPlugin(
            args.gfpgan_device, upscaler=plugins.get(RealESRGANUpscaler.name, None)
        )

    if args.enable_restoreformer:
        logger.info(f"Initialize {RestoreFormerPlugin.name} plugin")
        plugins[RestoreFormerPlugin.name] = RestoreFormerPlugin(
            args.restoreformer_device,
            upscaler=plugins.get(RealESRGANUpscaler.name, None),
        )

    if args.enable_gif:
        logger.info(f"Initialize GIF plugin")
        plugins[MakeGIF.name] = MakeGIF()


def main(args):
    global model
    global device
    global input_image_path
    global is_disable_model_switch
    global is_enable_file_manager
    global is_desktop
    global thumb
    global output_dir
    global is_enable_auto_saving
    global is_controlnet
    global controlnet_method
    global image_quality

    build_plugins(args)

    image_quality = args.quality

    if args.sd_controlnet and args.model in SD15_MODELS:
        is_controlnet = True
        controlnet_method = args.sd_controlnet_method

    output_dir = args.output_dir
    if output_dir:
        is_enable_auto_saving = True

    device = torch.device(args.device)
    is_disable_model_switch = args.disable_model_switch
    is_desktop = args.gui
    if is_disable_model_switch:
        logger.info(
            f"Start with --disable-model-switch, model switch on frontend is disable"
        )

    if args.input and os.path.isdir(args.input):
        logger.info(f"Initialize file manager")
        thumb = FileManager(app)
        is_enable_file_manager = True
        app.config["THUMBNAIL_MEDIA_ROOT"] = args.input
        app.config["THUMBNAIL_MEDIA_THUMBNAIL_ROOT"] = os.path.join(
            args.output_dir, "lama_cleaner_thumbnails"
        )
        thumb.output_dir = Path(args.output_dir)
        # thumb.start()
        # try:
        #     while True:
        #         time.sleep(1)
        # finally:
        #     thumb.image_dir_observer.stop()
        #     thumb.image_dir_observer.join()
        #     thumb.output_dir_observer.stop()
        #     thumb.output_dir_observer.join()

    else:
        input_image_path = args.input

    model = ModelManager(
        name=args.model,
        sd_controlnet=args.sd_controlnet,
        sd_controlnet_method=args.sd_controlnet_method,
        device=device,
        no_half=args.no_half,
        hf_access_token=args.hf_access_token,
        disable_nsfw=args.sd_disable_nsfw or args.disable_nsfw,
        sd_cpu_textencoder=args.sd_cpu_textencoder,
        sd_run_local=args.sd_run_local,
        sd_local_model_path=args.sd_local_model_path,
        local_files_only=args.local_files_only,
        cpu_offload=args.cpu_offload,
        enable_xformers=args.sd_enable_xformers or args.enable_xformers,
        callback=diffuser_callback,
    )

    if args.gui:
        app_width, app_height = args.gui_size
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app,
            socketio=socketio,
            width=app_width,
            height=app_height,
            host=args.host,
            port=args.port,
            close_server_on_exit=not args.no_gui_auto_close,
        )
        ui.run()
    else:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True,
        )
