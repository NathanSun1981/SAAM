import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes, draw_masks,get_palette
import numpy as np
import pycocotools.mask as maskUtils
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABLE

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
    parser.add_argument('--using_sam',  action="store_true", help='use sam for masks generation.')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
      
 
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
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def segmentor_inference(filename, output_path, img=None, save_img=False, 
                        mask_generator = None,
                        masked_color = None, 
                        predictor=None,
                        yolo_model=None,
                        sam=None,
                        use_sam=True,
                        convert_to_rle = True):

    if use_sam:      
        t = time.time()
        masks_sam = mask_generator.generate(img)
        print("generating elapse {} s".format(time.time()-t))
        anns = {'annotations': masks_sam}        
        
        ######## viusilization ############
        #print(len(masks_sam))
        #print(masks_sam[0].keys())          
        """ plt.figure(figsize=(20,20))
        plt.imshow(img0)
        show_anns(masks_sam)
        plt.axis('off')
        plt.show()  """                                             
    else:
        anns = {'annotations': []}
             
    t = time.time()
    #start yolo predict
    results = yolo_model.predict(source=img)   
    print("Yolo preducit Elapse {} s, counts of results = {}".format((time.time()-t), len(results)))
      
    result = results[0]
    
    boxes = result.boxes
    labels = boxes.cls
    print(boxes.cls)
    masks = result.masks.data
    #show_mask(masks, plt.gca(), True)   
    masks = np.array(masks.tolist()).astype(int)  
    
         
    img0 = mmcv.imread(img).astype(np.uint8)
    img0 = mmcv.bgr2rgb(img0)
    img0 = np.ascontiguousarray(img0)  
    mask_palette = get_palette(masked_color, len(labels))   
    colors = [mask_palette[i] for i in range(len(labels))]
    colors = np.array(colors, dtype=np.uint8)
    
    h, w, _ = img0.shape
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off') 
    plt.imshow(img0)
    _, _img = draw_masks(ax,img=img0,masks=masks,color = colors, alpha=0.5)
    plt.imshow(_img)
    plt.show()    
    
    labled_mask = np.zeros([h,w])
        
    for mask, bbox, cls in zip(masks, boxes.xyxy.tolist(), boxes.cls):
        input_box = np.array(bbox)
        
        """ masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        ) """
        
        labled_mask += mask * int(cls)
                    
        #for i, (mask, score) in enumerate(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca(), True)
        #show_box(input_box, plt.gca())
        #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 
    
        segmentation_mask = mask
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        white_background = np.ones_like(img) * 255
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + img * binary_mask[..., np.newaxis]
        plt.imshow(new_image.astype(np.uint8))
        plt.axis('off')
        plt.show()
            
    print(labled_mask.shape)
    print(np.unique(labled_mask))   
    
    class_ids = torch.tensor(labled_mask.astype(int))
    
    semantc_mask = class_ids.clone()   
    
    id2label = CONFIG_COCO_ID2LABLE
    class_names = []
    
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        if convert_to_rle:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        else:
            valid_mask = torch.tensor(ann['segmentation']).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(np.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])
        # bitmasks.append(maskUtils.decode(ann['segmentation']))  
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []

    # semantic prediction
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
    
    print("Create Semantic Masks Elapse {} s".format(time.time()-t))
    if save_img:       
        if use_sam:
            filename = filename + '_' + 'withSAM'
        else:
            filename = filename + '_' + 'withoutSAM'
        
        imshow_det_bboxes(img,
                            bboxes=None,
                            labels=np.arange(len(sematic_class_in_img)),
                            segms=np.stack(semantic_bitmasks),
                            class_names=semantic_class_names,
                            font_size=25,
                            show=False,
                            out_file=os.path.join(output_path,  filename + '.png'))
        print('save predictions to ', os.path.join(output_path, filename + '.png'))
    mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
  
def main(args): 
    #generate SAM masks
    print(args)
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.ckpt_path)
    _ = sam.to(device=args.device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="coco_rle" if args.convert_to_rle else "binary_mask",
    )
    
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
        img0 = mmcv.bgr2rgb(img)
        with torch.no_grad():
            # start to process segemtor      
            predictor.set_image(img)
    
            segmentor_inference(file_name, args.out_dir, img=img, save_img=args.save_img,
                                mask_generator=mask_generator,
                                predictor=predictor,
                                yolo_model=yolo_model,
                                sam=sam,
                                use_sam=args.using_sam,
                                convert_to_rle = args.convert_to_rle
                                )
        print("Totally Elapse {} s".format(time.time()-t))

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(args)