# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder, Modified_ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import queue

import torch.nn as nn
from pysot.tracker.modified_siamrpn_tracker import Modified_SiamRPNTracker
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()
torch.cuda.set_device(0)
torch.set_num_threads(8)
forget_factor_layer3=torch.tensor([0.5,0.004,0.012,0.037,0.11,0.33],dtype=torch.float32).cuda()
forget_factor_layer4=torch.tensor([0.5,0.004,0.012,0.037,0.11,0.33],dtype=torch.float32).cuda()
forget_factor_layer5=torch.tensor([0.5,0.004,0.012,0.037,0.11,0.33],dtype=torch.float32).cuda()
# forget_factor_layer3=torch.tensor([0.0940, 0.0868, 0.1019, 0.1409, 0.2189,0.3575],dtype=torch.float32).cuda()
# forget_factor_layer4=torch.tensor([0.0787, 0.0780, 0.0889, 0.1161, 0.2021,0.4363],dtype=torch.float32).cuda()
# forget_factor_layer5=torch.tensor([0.1573, 0.1589, 0.1635, 0.1697, 0.1743,0.1762],dtype=torch.float32).cuda()
def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = Modified_ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = Modified_SiamRPNTracker(model)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    template_queue_layer3 = []
    template_queue_layer4 = []
    template_queue_layer5 = []

    similarity_layer3=[]
    similarity_layer4=[]
    similarity_layer5=[]

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            scores = []
            #clear for every video
            template_queue_layer3.clear()
            template_queue_layer4.clear()
            template_queue_layer5.clear()
            first_frmae_template = []  # for the beginning of every video, store the first frame template ...
            pre_similarity = [1, 1, 1]  # for 3 4 5 layers
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    template_queue_layer3.clear()
                    template_queue_layer4.clear()
                    template_queue_layer5.clear()
                    first_frmae_template = []

                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img,gt_bbox_)
                    zf=tracker.Get_Updated_Template(img,gt_bbox_)
                    template_queue_layer3.append(zf[0])
                    template_queue_layer4.append(zf[1])
                    template_queue_layer5.append(zf[2])

                    first_frmae_template.append(zf[0])
                    first_frmae_template.append(zf[1])
                    first_frmae_template.append(zf[2])
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    zf_temp = []
                    t = torch.zeros(template_queue_layer3[0].shape)
                    t = t.cuda()

                    for i in range(len(template_queue_layer3)):
                        t += template_queue_layer3[i].cuda() * forget_factor_layer3[i]
                        # t += template_queue_layer3[i].cuda() * (
                        #             1 / (3 ** (len(template_queue_layer3) - i)))
                    zf_temp.append(t)

                    t = torch.zeros(template_queue_layer4[0].shape)
                    t = t.cuda()
                    for i in range(len(template_queue_layer4)):
                        t += template_queue_layer4[i] * forget_factor_layer4[i]  # *temp_similarity
                    zf_temp.append(t)

                    t = torch.zeros(template_queue_layer5[0].shape)
                    t = t.cuda()
                    for i in range(len(template_queue_layer5)):
                        t += template_queue_layer5[i] * forget_factor_layer5[i]  # *temp_similarity
                        # t += template_queue_layer5[i].cuda() * (
                        #         1 / (3 ** (len(template_queue_layer5) - i)))  # *temp_similarity
                    zf_temp.append(t)
                    outputs = tracker.track(img, zf_temp)
                    print("the length of template queue layer3: ",len(template_queue_layer3))
                    # outputs['best_score']>=0.90 and outputs['best_score_list'][0]>0.85  0.90 0.94
                    # if follow the wrong target,clear template queue and store only the first frame template..
                    if outputs['best_score'] <= 0.2:
                        template_queue_layer3.clear()
                        similarity_layer3.clear()
                        template_queue_layer3.append(first_frmae_template[0])
                        similarity_layer3.append(1.0)
                    if outputs['best_score'] <= 0.2:
                        template_queue_layer4.clear()
                        similarity_layer4.clear()
                        template_queue_layer4.append(first_frmae_template[1])
                        similarity_layer4.append(1.0)
                    if outputs['best_score'] <= 0.2:
                        template_queue_layer5.clear()
                        similarity_layer5.clear()
                        template_queue_layer5.append(first_frmae_template[2])
                        similarity_layer5.append(1.0)

                    if outputs['best_score'] >= 0.90 and outputs['best_score_list'][0] > 0.90 and abs(
                            outputs['best_score_list'][0] - pre_similarity[0]) <= 0.12:
                        if len(template_queue_layer3) < cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer3.append(tracker.Get_Updated_Template(img, outputs['bbox'])[0])
                        else:
                            for k in range(2, len(template_queue_layer3)):
                                template_queue_layer3[k - 1] = template_queue_layer3[k]
                            template_queue_layer3[len(template_queue_layer3) - 1] = \
                            tracker.Get_Updated_Template(img, outputs['bbox'])[0]
                    if outputs['best_score'] >= 0.90 and outputs['best_score_list'][1] > 0.95 and abs(
                            outputs['best_score_list'][1] - pre_similarity[1]) <= 0.12:
                        if len(template_queue_layer4) < cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer4.append(tracker.Get_Updated_Template(img, outputs['bbox'])[1])
                        else:
                            for k in range(2, len(template_queue_layer4)):
                                template_queue_layer4[k - 1] = template_queue_layer4[k]
                            template_queue_layer4[len(template_queue_layer4) - 1] = \
                            tracker.Get_Updated_Template(img, outputs['bbox'])[1]

                    if outputs['best_score'] >= 0.90 and outputs['best_score_list'][2] > 0.99 and abs(
                            outputs['best_score_list'][2] - pre_similarity[2]) <= 0.12:
                        if len(template_queue_layer5) < cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer5.append(tracker.Get_Updated_Template(img, outputs['bbox'])[2])
                        else:
                            for k in range(2, len(template_queue_layer5)):
                                template_queue_layer5[k - 1] = template_queue_layer5[k]
                            template_queue_layer5[len(template_queue_layer5) - 1] = \
                            tracker.Get_Updated_Template(img, outputs['bbox'])[2]
                    pre_similarity[0] = outputs['best_score_list'][0]
                    pre_similarity[1] = outputs['best_score_list'][1]
                    pre_similarity[2] = outputs['best_score_list'][2]
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            #clear for every video
            template_queue_layer3.clear()
            template_queue_layer4.clear()
            template_queue_layer5.clear()
            #clear for every video
            similarity_layer3.clear() #stands for the similarity with template
            similarity_layer4.clear()
            similarity_layer5.clear()
            pre_similarity=[1,1,1] #for 3 4 5 layers
            first_frmae_template = [] #for the beginning of every video, store the first frame template ...
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    #tracker.init(img。 gt_bbox_)
                    tracker.init(img, gt_bbox_)
                    zf = tracker.Get_Updated_Template(img, gt_bbox_)
                    template_queue_layer3.append(zf[0])
                    template_queue_layer4.append(zf[1])
                    template_queue_layer5.append(zf[2])

                    first_frmae_template.append(zf[0])
                    first_frmae_template.append(zf[1])
                    first_frmae_template.append(zf[2])

                    similarity_layer3.append(1.0) #because the first frame is template...
                    similarity_layer4.append(1.0)
                    similarity_layer5.append(1.0)


                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    zf_temp=[]
                    t=torch.zeros(template_queue_layer3[0].shape)
                    t=t.cuda()

                    for i in range(len(template_queue_layer3)):
                        t+=template_queue_layer3[i].cuda()*forget_factor_layer3[i]
                        # t += template_queue_layer3[i].cuda() * (
                        #             1 / (3 ** (len(template_queue_layer3) - i)))
                    zf_temp.append(t)

                    t=torch.zeros(template_queue_layer4[0].shape)
                    t = t.cuda()
                    for i in range(len(template_queue_layer4)):
                        t += template_queue_layer4[i] *forget_factor_layer4[i] #*temp_similarity
                    zf_temp.append(t)

                    t = torch.zeros(template_queue_layer5[0].shape)
                    t = t.cuda()
                    for i in range(len(template_queue_layer5)):
                        t += template_queue_layer5[i] *forget_factor_layer5[i] #*temp_similarity
                        # t += template_queue_layer5[i].cuda() * (
                        #         1 / (3 ** (len(template_queue_layer5) - i)))  # *temp_similarity
                    zf_temp.append(t)
                    outputs = tracker.track(img,zf_temp)
                    print("outputs['best_score']: ",outputs['best_score']," outputs['best_score_list'][0]:",outputs['best_score_list'][0],
                          " outputs['best_score_list'][1]:",outputs['best_score_list'][1]," outputs['best_score_list'][2]:",outputs['best_score_list'][2])
                    #outputs['best_score']>=0.90 and outputs['best_score_list'][0]>0.85  0.90 0.94
                    #if follow the wrong target,clear template queue and store only the first frame template..
                    if outputs['best_score']<=0.4 :
                        template_queue_layer3.clear()
                        similarity_layer3.clear()
                        template_queue_layer3.append(first_frmae_template[0])
                        similarity_layer3.append(1.0)
                    if outputs['best_score']<=0.4 :
                        template_queue_layer4.clear()
                        similarity_layer4.clear()
                        template_queue_layer4.append(first_frmae_template[1])
                        similarity_layer4.append(1.0)
                    if outputs['best_score']<=0.4 :
                        template_queue_layer5.clear()
                        similarity_layer5.clear()
                        template_queue_layer5.append(first_frmae_template[2])
                        similarity_layer5.append(1.0)

                    if outputs['best_score']>=0.90 and outputs['best_score_list'][0]> 0.90 and abs(outputs['best_score_list'][0]-pre_similarity[0])<=0.2:
                        if len(template_queue_layer3)<cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer3.append(tracker.Get_Updated_Template(img,outputs['bbox'])[0])
                            similarity_layer3.append(outputs['best_score_list'][0])
                        else:
                            for k in range(2,len(template_queue_layer3)):
                                template_queue_layer3[k-1]=template_queue_layer3[k]
                                similarity_layer3[k-1]=similarity_layer3[k]
                            template_queue_layer3[len(template_queue_layer3)-1]=tracker.Get_Updated_Template(img,outputs['bbox'])[0]
                            similarity_layer3[len(template_queue_layer3)-1]=outputs['best_score_list'][0]
                    if outputs['best_score'] >= 0.90 and outputs['best_score_list'][1] > 0.95 and abs(outputs['best_score_list'][1]-pre_similarity[1])<=0.2:
                        if len(template_queue_layer4) < cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer4.append(tracker.Get_Updated_Template(img, outputs['bbox'])[1])
                            similarity_layer4.append(outputs['best_score_list'][1])
                        else:
                            for k in range(2,len(template_queue_layer4)):
                                template_queue_layer4[k-1]=template_queue_layer4[k]
                                similarity_layer4[k-1]=similarity_layer4[k]
                            template_queue_layer4[len(template_queue_layer4)-1] = tracker.Get_Updated_Template(img, outputs['bbox'])[1]
                            similarity_layer4[len(template_queue_layer4)-1]=outputs['best_score_list'][1]

                    if outputs['best_score'] >= 0.90 and outputs['best_score_list'][2] > 0.99 and abs(outputs['best_score_list'][2]-pre_similarity[2])<=0.2:
                        if len(template_queue_layer5) < cfg.PARA.MAX_QUEUE_SIZE:
                            template_queue_layer5.append(tracker.Get_Updated_Template(img, outputs['bbox'])[2])
                            similarity_layer5.append(outputs['best_score_list'][2])
                        else:
                            for k in range(2,len(template_queue_layer5)):
                                template_queue_layer5[k - 1] = template_queue_layer5[k]
                                similarity_layer5[k - 1] = similarity_layer5[k]
                            template_queue_layer5[len(template_queue_layer5)-1] = tracker.Get_Updated_Template(img, outputs['bbox'])[2]
                            similarity_layer5[len(template_queue_layer5)-1]=outputs['best_score_list'][2]
                    pre_similarity[0]=outputs['best_score_list'][0]
                    pre_similarity[1] = outputs['best_score_list'][1]
                    pre_similarity[2] = outputs['best_score_list'][2]
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (255, 0, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
