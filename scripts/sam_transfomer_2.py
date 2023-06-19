import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABLE

import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils

import torch.distributed as dist
import torch.multiprocessing as mp
import time

from functools import reduce



def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segmentation using SAM and Oneformer/Segfromer.')
    parser.add_argument('--device', default='cpu', help='The device to run generation on.')
    parser.add_argument('--data_dir', default='data/room/lounge', help='specify the root path of images and masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', default='ouput/', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
    parser.add_argument('--dataset', type=str, default='coco', choices=['ade20k', 'cityscapes', 'coco'], help='specify the set of class names')
    parser.add_argument('--seg_model', type=str, default='yolo', choices=['oneformer', 'segformer', 'yolo'], help='specify the segmenta model')
    ###argment for SAM, inherit from SAM code###
    parser.add_argument("--sam_model_type", type=str, default='vit_h', help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--points_per_side", type=int, default=64, help="Generate masks by sampling a grid over the image with this many points to a side.")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86, help="Exclude masks with a predicted score from the model that is lower than this threshold.")
    parser.add_argument("--stability_score_thresh", type=float,default=0.92, help="Exclude masks with a stability score lower than this threshold.")
    parser.add_argument("--crop_n_layers", type=int, default=1, help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
        ),
    )
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=2, help="The number of points-per-side in each layer of crop is reduced by this factor.")
    parser.add_argument("--min_mask_region_area", type=int, default=100, help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
        ),
    )
    
    parser.add_argument("--convert_to_rle", action="store_true", default=True, help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
        ),
    )
    
    parser.add_argument("--file_suffix", type=str, default='.png', help='suffix of images in dataset')
    
 
    args = parser.parse_args()
    return args

def yolo_segmentation(image, model):
    h, w, _ = image.shape
    result = model.predict(source=image)   

    predicted_semantic_map = []
    masks=torch.tensor([])
    ind = 0
    
    if result[0].boxes is not None and masks is not None:
        print(result[0].masks.data.ndim)      
        for mask, cls in zip(result[0].masks.data, result[0].boxes.cls): 
            print(mask.shape)
            print(np.unique(mask.to('cpu')))
            if ind == 0:
                masks = mask * cls
            else:               
                masks = masks.append(mask * cls)
            ind += 1
        #masks = torch.add(result[0].masks.data, 0)
        print(masks.shape)
        predicted_semantic_map = reduce(
            torch.Tensor.add_,
            masks,
            torch.zeros_like(masks[0])  # optionally set initial element to avoid changing `x`
        )
        print(predicted_semantic_map.shape)
        print(np.unique(predicted_semantic_map.to('cpu')))
        print(predicted_semantic_map.shape)
                 

    predicted_semantic_map = mask
    
    return predicted_semantic_map

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


oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
}

def segmentor_inference(filename, output_path, img=None, save_img=False,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None,
                                 model='oneformer'):
    t = time.time()
    #anns = {'annotations': mask_branch_model.generate(img)}
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
    print(np.unique(class_ids))
    semantc_mask = class_ids.clone()   
    print("Generating Class Ids Elapse {} s".format(time.time()-t))
    
    t = t = time.time()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
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
        """ fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
        _ax, _img = draw_masks(ax,img=img,masks=np.stack(semantic_bitmasks),alpha=0.5)
        plt.imshow(_img)
        plt.show()
        mmcv.imwrite(_img, os.path.join(output_path, filename + '_semantic.png')) """
        
        imshow_det_bboxes(img,
                            bboxes=None,
                            labels=np.arange(len(sematic_class_in_img)),
                            segms=np.stack(semantic_bitmasks),
                            class_names=semantic_class_names,
                            font_size=25,
                            show=False,
                            out_file=os.path.join(output_path,  dataset + '_' + model + '_' + filename + '_semantic.png'))
        print('save predictions to ', os.path.join(output_path, filename + '_semantic.png'))
    mmcv.dump(anns, os.path.join(output_path, dataset + '_' + model + '_' + filename + '_semantic.json'))
    
def main(args): 
    #generate SAM masks
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.ckpt_path)
    _ = sam.to(device=args.device)

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
        from ultralytics import YOLO
        if args.dataset == 'coco':
            semantic_branch_processor = None
            semantic_branch_model = YOLO('yolov8n-seg.pt')
        else:
            raise NotImplementedError()
       
    else:
        raise NotImplementedError()
    print('Semantic Segmentor is loaded.')
    
    ext = args.file_suffix
    # Check if the file with current extension exists
    local_filenames = [fn_.replace(ext, '') for fn_ in os.listdir(args.data_dir) if ext in fn_]
        
    print(local_filenames)
    print('Images Loaded')
    print('Inference starts')

    for i, file_name in enumerate(local_filenames):
        t = time.time()
        print('Processing ', i + 1 , '/', len(local_filenames), ' ', file_name+ext)
        img = mmcv.imread(os.path.join(args.data_dir, file_name+ext))
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
            segmentor_inference(file_name, args.out_dir, img=img, save_img=args.save_img,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=args.dataset,
                                   id2label=id2label,
                                   model=args.seg_model)
        print("Totally Elapse {} s".format(time.time()-t))

if __name__ == '__main__':
    args = parse_args()
    #print(args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(args)