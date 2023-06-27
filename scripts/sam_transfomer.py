import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
import cv2
import open3d as o3d
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABLE

import mmcv
from mmdet.core.visualization.image import draw_masks, get_palette, palette_val, draw_bboxes, draw_labels, _get_adaptive_scales, get_palette
from mmdet.core.mask.structures import bitmap_to_polygon
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import time
from config import ConfigParser
import dense_slam
import glob

from ultralytics import YOLO


EPS = 1e-2



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
    


def segformer_segmentation(image, processor, model):
    h, w, _ = image.shape
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
    predicted_semantic_map = logits.argmax(dim=1).squeeze(0)
    return predicted_semantic_map

def oneformer_coco_segmentation(image, oneformer_coco_processor, oneformer_coco_model):
    inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = oneformer_coco_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(image, oneformer_cityscapes_processor, oneformer_cityscapes_model):
    inputs = oneformer_cityscapes_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = oneformer_cityscapes_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def yolo_segmentation(img, yolo_model, masked_color=None):
    #start yolo predict
    t = time.time()
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
    
    """ fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off') 
    plt.imshow(img0)
    _, _img = draw_masks(ax,img=img0,masks=masks,color = colors, alpha=0.5)
    plt.imshow(_img)
    plt.show()  """   
    
    labled_mask = np.zeros([h,w])
    binary_mask = np.zeros([h,w])
        
    for mask, bbox, cls in zip(masks, boxes.xyxy.tolist(), boxes.cls):
        input_box = np.array(bbox)
        
        """ masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        ) """
        real_mask = mask - binary_mask
        real_mask = np.where(real_mask > 0, real_mask, 0)
        #binary_mask = real_mask + binary_mask
        #print(np.unique(real_mask))
        #print(np.unique(binary_mask))
        
        labled_mask = np.where(labled_mask > 0, labled_mask, real_mask * int(cls))
        
        #labled_mask += real_mask * int(cls)
                    
        #for i, (mask, score) in enumerate(masks, scores)):
        #plt.figure()
        #plt.imshow(img)
        #show_mask(mask, plt.gca(), True)
        #show_box(input_box, plt.gca())
        #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #plt.axis('off')
        #plt.show() 
    
        segmentation_mask = mask
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        white_background = np.ones_like(img) * 255
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + img * binary_mask[..., np.newaxis]
        #plt.imshow(new_image.astype(np.uint8))
        #plt.axis('off')
        #plt.show()
            
    print(labled_mask.shape)
    print(np.unique(labled_mask))   
    
    class_ids = torch.tensor(labled_mask.astype(int))
    return class_ids

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
}


def draw_only_masks(ax, img, depth, masks):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    h, w, _ = img.shape
    masked_img = np.zeros([h,w])

    for i, mask in enumerate(masks):
        
        masked_img = np.bitwise_or(mask.astype(bool),masked_img.astype(bool)).astype(int)

        mask_tmp = np.unique(masked_img)
        tmp = {}
        for v in mask_tmp:
            tmp[v] = np.sum(masked_img == v)
        print("mask值为：")
        print(mask_tmp)
        print("统计结果：")
        print(tmp)
    white_background = np.ones_like(img) * 255    
    img = white_background * (1 - masked_img[..., np.newaxis]) + img * masked_img[..., np.newaxis]
    depth = depth * masked_img[..., np.newaxis]
        
    return img, depth


def imshow_det_bboxes(img,
                      depth=None,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      sematic_classes = None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      classes = None,
                      wait_time=0,
                      color_out_file=None,
                      depth_out_file=None,
                      draw_label = False):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        classes (array): only show the filtered classes
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = img.astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    
    inds = np.full(len(labels), False, dtype=bool)

    if classes is not None:
        for cls in classes:
            inds = (inds | (sematic_classes == cls))
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)
    depth = np.ascontiguousarray(depth)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        if draw_label:
            horizontal_alignment = 'left'
            positions = bboxes[:, :2].astype(np.int32) + thickness
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
            
            draw_labels(
                ax,
                labels[:num_bboxes],
                positions,
                scores=scores,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        if classes is None:
            draw_masks(ax, img, segms, colors, with_edge=True)
        else:
            img, depth = draw_only_masks(ax, img, depth, segms)
        

        if num_bboxes < segms.shape[0] and draw_label:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)

            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)
    
    depth = depth.astype('uint16')
    

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if color_out_file is not None:
        mmcv.imwrite(img, color_out_file)       
    
    if depth_out_file is not None:
        mmcv.imwrite(depth, depth_out_file)
        

    plt.close()

    return img


def segmentor_inference(filename, color_output_path, depth_output_path, img=None, depth=None, save_img=False, classes = None,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None,
                                 model='oneformer',
                                 use_sam=True,
                                 convert_to_rle = True):
    t = time.time()
    if use_sam:
        anns = {'annotations': mask_branch_model.generate(img)}
    else:
        anns = {'annotations': []}
    
    print("Generating Masks Elapse {} s".format(time.time()-t))
    
    t = time.time()
    h, w, _ = img.shape
    class_names = []
    print("run on model", model)
    if model == 'oneformer':
        class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
                                                                        semantic_branch_model)
    elif model == 'segformer':
        class_ids = segformer_segmentation(img, semantic_branch_processor, semantic_branch_model)   
    
    elif model == 'yolo':
        class_ids = yolo_segmentation(img, semantic_branch_model)
        
         
    else:
        raise NotImplementedError()

    semantc_mask = class_ids.clone()   
    print("Generating Class Ids Elapse {} s".format(time.time()-t))
    
    t = t = time.time()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        if convert_to_rle:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        else:
            valid_mask = torch.tensor(ann['segmentation']).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
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
    print(sematic_class_in_img)
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
        """ if use_sam:
            filename = filename + '_' + 'withSAM'
        else:
            filename = filename + '_' + 'withoutSAM' """
        
        imshow_det_bboxes(img, depth,
                            bboxes=None,
                            labels=np.arange(len(sematic_class_in_img)),
                            sematic_classes = np.array(sematic_class_in_img),
                            segms=np.stack(semantic_bitmasks),
                            class_names=semantic_class_names,
                            font_size=25,
                            show=False,
                            classes = classes,
                            draw_label=False,
                            color_out_file=os.path.join(color_output_path, filename + '.png'),
                            depth_out_file=os.path.join(depth_output_path, filename + '.png'))

        print('save predictions to ', os.path.join(color_output_path, filename + '.png'))
    #mmcv.dump(anns, os.path.join(output_path, dataset + '_' + model + '_' + filename + '_semantic.json'))
    
def main(args): 
    #generate SAM masks
    #print(args)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.ckpt_path)
    _ = sam.to(device=(args.sam_device))

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="coco_rle" if args.convert_to_rle else "binary_mask",
    )
    print('SAM is loaded.')
    if args.seg_model == 'oneformer':
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
        elif args.dataset == 'cityscapes':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
        elif args.dataset == 'coco':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_coco_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_coco_swin_large")
        else:
            raise NotImplementedError()
    elif args.seg_model == 'segformer':
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
        elif args.dataset == 'cityscapes':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        else:
            raise NotImplementedError()    
    elif args.seg_model == 'yolo':
        semantic_branch_processor = None
        semantic_branch_model = YOLO(args.ckpt_yolo)
        
    else:
        raise NotImplementedError()
    print('Semantic Segmentor is loaded.')
    
    ext = args.file_suffix
    # Check if the file with current extension exists

    local_filenames = [fn_.replace(ext, '') for fn_ in os.listdir(args.color_dir) if ext in fn_]
        
    #print(local_filenames)
    print('Images Loaded')
    print('Inference starts')

    for i, file_name in enumerate(local_filenames):
        t = time.time()
        print('Processing ', i + 1 , '/', len(local_filenames), ' ', file_name+ext)
        img = mmcv.imread(os.path.join(args.color_dir, file_name+ext))
        depth_ref = np.array(o3d.t.io.read_image(os.path.join(args.depth_dir, file_name + '.png')))
        if args.dataset == 'ade20k':
            id2label = CONFIG_ADE20K_ID2LABEL
        elif args.dataset == 'cityscapes':
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        elif args.dataset == 'coco':
            id2label = CONFIG_COCO_ID2LABLE
        else:
            raise NotImplementedError()
        with torch.no_grad():
            # start to process segemtor
            segmentor_inference(file_name, os.path.join(args.path_dataset, args.color_folder), os.path.join(args.path_dataset, args.depth_folder),
                                img=img, depth=depth_ref, save_img=args.save_img, classes= args.classes,
                                semantic_branch_processor=semantic_branch_processor,
                                semantic_branch_model=semantic_branch_model,
                                mask_branch_model=mask_branch_model,
                                dataset=args.dataset,
                                id2label=id2label,
                                model=args.seg_model,
                                use_sam=args.using_sam,
                                convert_to_rle = args.convert_to_rle)
        print("Totally Elapse {} s".format(time.time()-t))

def load_config():
    parser = ConfigParser()
    config = parser.get_config()
    
    return config   
    
if __name__ == '__main__':
    #load config for slam
    args = load_config()
    #print(args)
    if not os.path.exists(os.path.join(args.path_dataset, args.color_folder)):
        os.mkdir(os.path.join(args.path_dataset, args.color_folder))
        
    files = glob.glob(os.path.join(args.path_dataset, args.color_folder, '*.png'))
    if len(files) > 0:
        print("processed files exist!!!")  
        #main(args)
    else: 
        main(args)
        
    dense_slam.run_slam(args)