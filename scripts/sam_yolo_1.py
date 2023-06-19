import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils
from ultralytics import YOLO
import matplotlib.pyplot as plt

import time


def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segmentation using SAM and Yolo.')
    parser.add_argument('--device', default='cpu', help='The device to run generation on.')
    parser.add_argument('--data-dir', default='data/room/lounge', help='specify the root path of images and masks')
    parser.add_argument('--ckpt-yolo', default='yolov8n-seg.pt', help='specify yolo segmentation model')
    parser.add_argument('--out-dir', default='ouput/', help='the dir to save semantic annotations')
    parser.add_argument('--save-img', default=True, action='store_true', help='whether to save annotated images')
     ###argment for SAM, inherit from SAM repos###
    parser.add_argument('--ckpt-path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument("--sam-model-type", type=str, default='vit_h', help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--points-per-side", type=int, default=64, help="Generate masks by sampling a grid over the image with this many points to a side.")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86, help="Exclude masks with a predicted score from the model that is lower than this threshold.")
    parser.add_argument("--stability-score-thresh", type=float,default=0.92, help="Exclude masks with a stability score lower than this threshold.")
    parser.add_argument("--crop-n-layers", type=int, default=1, help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
        ),
    )
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2, help="The number of points-per-side in each layer of crop is reduced by this factor.")
    parser.add_argument("--min-mask-region-area", type=int, default=100, help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
        ),
    )
    
    parser.add_argument("--convert-to-rle", action="store_true", help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
        ),
    )
    
    parser.add_argument("--file-suffix", type=str, default='.png', help='suffix of images in dataset')
    parser.add_argument('--using-sam', action="store_true", help='use sam for masks generation.')
    
 
    args = parser.parse_args()
    return args

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


def segmentor_inference(filename, output_path, img=None, save_img=False,
                                 predictor=None,
                                 yolo_model=None,
                                 use_sam=True):
    t = time.time()
    #start yolo predict
    results = yolo_model.predict(source=img)   
    print("Yolo preducit Elapse {} s".format(time.time()-t))
    #start sam predict
    
    for result in results:    
        boxes = result.boxes
        print(boxes.cls)
        for bbox, cls in zip(boxes, boxes.cls):
            input_box = np.array(bbox)
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(img)
                show_mask(mask, plt.gca())
                show_box(input_box, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show() 
            
                segmentation_mask = mask
                binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

                white_background = np.ones_like(img) * 255
                new_image = white_background * (1 - binary_mask[..., np.newaxis]) + img * binary_mask[..., np.newaxis]
                plt.imshow(new_image.astype(np.uint8))
                plt.axis('off')
                plt.show()
            
        
  
def main(args): 
    #generate SAM masks
    print(args)
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.ckpt_path)
    _ = sam.to(device=args.device)
    predictor = SamPredictor(sam)   
    print('SAM is loaded.')

    print('Semantic Segmentor is loaded.')
    
    ext = args.file_suffix
    # Check if the file with current extension exists
    local_filenames = [fn_.replace(ext, '') for fn_ in os.listdir(args.data_dir) if ext in fn_]
        
    print(local_filenames)
    print('Images Loaded')
    print('Inference starts')

    yolo_model = YOLO(args.ckpt_yolo)

    for i, file_name in enumerate(local_filenames):
        t = time.time()
        print('Processing ', i + 1 , '/', len(local_filenames), ' ', file_name+ext)
        img = mmcv.imread(os.path.join(args.data_dir, file_name+ext))
        with torch.no_grad():
            # start to process segemtor      
            predictor.set_image(img)
            
            from ultralytics.yolo.data.annotator import auto_annotate
            auto_annotate(img, det_model="yolov8x.pt", sam_model='sam_b.pt')
    
            """ segmentor_inference(file_name, args.out_dir, img=img, save_img=args.save_img,
                                   predictor=predictor,
                                   yolo_model=yolo_model,
                                   use_sam=args.using_sam) """
        print("Totally Elapse {} s".format(time.time()-t))

if __name__ == '__main__':
    args = parse_args()
    #print(args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(args)