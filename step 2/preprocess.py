import torchvision.ops.roi_pool as roi_pool
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torchvision.utils as utils
import os
import cv2
import numpy as np
import tensorflow as tf
import json

def mask_visual(img_path='./data/images/',mask_path='./data/masks_rgb/', video_path='./data/visual_mask_video/'):
    for sub_folder in os.listdir(img_path):
        for vid in os.listdir(os.path.join(img_path, sub_folder)):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(video_path, str(vid)+'.avi'), fourcc, 2, (256, 256))
            out_mask = cv2.VideoWriter(os.path.join(video_path, str(vid)+'_mask.avi'), fourcc, 3, (256, 256))

            print('saving', os.path.join(video_path, str(vid)+'.avi'))
            img_name = os.path.join(img_path, sub_folder, vid)
            mask_name = img_name.replace(img_path, mask_path)

            img_list = os.listdir(img_name)
            img_list.sort(key=lambda x: int(x[:-4]))

            for frame in img_list:
                img = cv2.imread(os.path.join(img_name, frame))
                mask = cv2.imread(os.path.join(mask_name, frame))
                combined_img = cv2.addWeighted(img, 1, mask, 0.4, 0)
                out.write(combined_img)
                out_mask.write(mask)
            out.release()
            out_mask.release()



def PositionalEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding




def geometric_encoding_pair(box_grasper, boxes_other):
    '''

    :param box_grasper:
    :param boxes_other:
    :return: geo list (K, 4) K is variable length
    cls list (K) K is variable length
    '''
    class2, center2_x, center2_y, w2, h2 = box_grasper
    geo = []
    clas = []
    for box in boxes_other:
        class1, center1_x, center1_y, w1, h1 = box

        # [K, R]
        delta_x = center1_x - center2_x
        delta_x = np.clip(np.abs(delta_x), a_min=1e-3, a_max=None)
        delta_x = math.log(delta_x, 0.2)

        delta_y = center1_y - center2_y
        delta_y = np.clip(np.abs(delta_y), a_min=1e-3, a_max=None)
        delta_y = math.log(delta_y, 0.2)

        delta_w = math.log(w1 / w2, 0.2)

        delta_h = math.log(h1 / h2, 0.2)

        geo.append([delta_x, delta_y, delta_w, delta_h])
        clas.append(class1)
    return geo, clas

from PIL import Image
from collections import OrderedDict

def bbox2geo(img_path='/home/kun/Desktop/miccai21/data/images/', bbox_anno='/home/kun/Desktop/miccai21/data/annotations/bbox_annotations_5_4/',
             other_box_seq_list_name='/home/kun/Desktop/miccai21/data/annotations/other_box_seq_list_5_5.json',
             other_box_class_seq_list_name='/home/kun/Desktop/miccai21/data/annotations/other_box_class_seq_list_5_5.json'):
    '''

    :return: Save a this_other_box list for each video (seq_len, K, 4) K is variable length
    Save a this_other_box_class list for each video (seq_len, K)
    '''
    # [N,1] a list of variable number of boxes
    other_box_seq_list = OrderedDict()
    # [N,1] # a list of variable number of boxes classes
    other_box_class_seq_list = OrderedDict()

    for sub_folder in os.listdir(img_path):

        for vid in os.listdir(os.path.join(img_path, sub_folder)):
            vid_name_img = os.path.join(img_path, sub_folder, vid)

            vid_length = len(os.listdir(vid_name_img))
            this_other_box = [] # (seq_len, K, 4)
            this_other_box_class = [] # (seq_len, K)


            img_list = os.listdir(vid_name_img)
            img_list.sort(key=lambda x: int(x[:-4]))

            for frame in img_list:
                img_name = os.path.join(vid_name_img, frame)
                frame_id = img_name[-12:-4]
                bbox_name = os.path.join(bbox_anno, vid, frame_id+'.txt')

                # ['Grasper':0, 'Bipolar':1, 'Hook':2, 'Scissors':3, 'Clipper':4, 'Irrigator':5, 'SpecimenBag':6]
                if os.path.exists(bbox_name):
                    bboxes = np.loadtxt(bbox_name)
                    print('bboxes', bboxes)
                    if len(bboxes.shape) < 2:
                        print('no interaction')
                        this_other_box.append([])
                        this_other_box_class.append([])
                        continue
                    if 0 in bboxes[:, 0]:
                        grasper_box = bboxes[bboxes[:, 0] == 0][0]
                        other_boxes = bboxes[bboxes[:, 0] != 0]
                        if other_boxes.shape == (0, 5):
                            print('only grasper presents')
                            this_other_box.append([])
                            this_other_box_class.append([])
                            continue
                        bboxes, bboxes_class = geometric_encoding_pair(grasper_box, other_boxes)
                        print('grasper: ', grasper_box, ' others: ', other_boxes, ' out: ', bboxes, bboxes_class)
                        this_other_box.append(bboxes)
                        this_other_box_class.append(bboxes_class)
                    else:
                        print('no grasper')
                        this_other_box.append([])
                        this_other_box_class.append([])
                        continue
                else:
                    print('no detection file, boxes is none')
                    this_other_box.append([])
                    this_other_box_class.append([])
                    continue
            assert len(this_other_box_class) == len(this_other_box)
            assert len(this_other_box_class) == vid_length
            other_box_seq_list[vid] = this_other_box # (N, seq_len, K, 4) seq_len & K are variable
            other_box_class_seq_list[vid] = this_other_box_class # (N, seq_len, K)
    print(max(other_box_seq_list['video01']))
    json_str = json.dumps(other_box_seq_list)
    with open(other_box_seq_list_name, 'w') as json_file:
        json_file.write(json_str)


    json_str = json.dumps(other_box_class_seq_list)
    with open(other_box_class_seq_list_name, 'w') as json_file:
        json_file.write(json_str)



def visual_bbox2geo(img_path='/home/kun/Desktop/miccai21/data/images_raw_resolution/', other_box_seq_list='/home/kun/Desktop/miccai21/data/annotations/other_box_seq_list_5_5.json',
                    other_box_class_seq_list='/home/kun/Desktop/miccai21/data/annotations/other_box_class_seq_list_5_5.json', bbox_anno='/home/kun/Desktop/miccai21/data/annotations/bbox_annotations_5_4/',
                    save_path='/home/kun/Desktop/miccai21/data/visual_bbox2geo/'):
    '''
    Raw resolution: (854, 480)
    :param img_path:
    :param other_box_seq_list:
    :param other_box_class_seq_list:
    :return:
    '''

    with open(other_box_seq_list) as f:
        other_box_seq_ = json.load(f)
    with open(other_box_class_seq_list) as f:
        other_box_class_seq_ = json.load(f)

    tool_category = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

    for sub_folder in os.listdir(img_path):

        for vid in os.listdir(os.path.join(img_path, sub_folder)):
            vid_name_img = os.path.join(img_path, sub_folder, vid)

            this_other_box = other_box_seq_[vid]
            this_other_box_class = other_box_class_seq_[vid]

            save_name = vid_name_img.replace(img_path, save_path)+'.avi'

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            videoWriter = cv2.VideoWriter(save_name, fourcc, 2, (640, 384))


            img_list = os.listdir(vid_name_img)
            img_list.sort(key=lambda x: int(x[:-4]))

            for frame in img_list:
                img_name = os.path.join(vid_name_img, frame)
                bbox_name = os.path.join(bbox_anno, vid, frame.replace('png','txt'))
                frame_id = int(img_name[-12:-4])

                in_img = cv2.imread(img_name)

                if os.path.exists(bbox_name):
                    bboxes = np.loadtxt(bbox_name)
                    if len(bboxes.shape) > 1:
                        for box in bboxes:
                            _, x, y, w, h = box[0], box[1], box[2], box[3], box[4]
                            x = x - w/2
                            y = y - h/2
                            x, y, w, h = int(x*640), int(y*384), int(w*640), int(h*384)
                            cv2.rectangle(in_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            text = tool_category[int(_)]
                            cv2.putText(in_img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

                    elif len(bboxes.shape) == 1:
                        box = bboxes
                        _, x, y, w, h = box[0], box[1], box[2], box[3], box[4]
                        x = x - w / 2
                        y = y - h / 2
                        x, y, w, h = int(x*640), int(y*384), int(w*640), int(h*384)
                        cv2.rectangle(in_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        text = tool_category[int(_)]
                        cv2.putText(in_img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)


                this_frame_other_box = this_other_box[frame_id]
                this_frame_other_box_class = this_other_box_class[frame_id]

                add_text = False
                for idx, title in enumerate(this_frame_other_box_class):
                    text = tool_category[int(title)]
                    AddText = in_img.copy()
                    cv2.putText(AddText, text, (idx*80, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 200, 200), 5)

                    add_text = True

                if add_text:
                    videoWriter.write(AddText)
                else:
                    videoWriter.write(in_img)
            videoWriter.release()
    return False


# def bbox2bboxlist(img_path='./data/images/', bbox_anno='./data/annotations/bbox_annotations/'):
#     '''
#
#     :return: Save a box class list for each video (seq_len, K)
#     Save a box list for each video (seq_len, K, 4)
#     K is variable length
#     save format .json file
#     '''
#     # [N,1] a list of variable number of boxes
#     box_seq_list = OrderedDict()
#     box_class_seq_list = OrderedDict()
#
#
#     for sub_folder in os.listdir(img_path):
#
#         for vid in os.listdir(os.path.join(img_path, sub_folder)):
#             vid_name_img = os.path.join(img_path, sub_folder, vid)
#
#             vid_length = len(os.listdir(vid_name_img))
#             this_box = [] # (seq_len, K, 4)
#             this_box_class = [] # (seq_len, K, 4)
#
#             img_list = os.listdir(vid_name_img)
#             img_list.sort(key=lambda x: int(x[:-4]))
#
#             for frame in img_list:
#                 img_name = os.path.join(vid_name_img, frame)
#                 frame_id = img_name[-12:-4]
#                 bbox_name = os.path.join(bbox_anno, vid, frame_id+'.txt')
#
#                 # ['Grasper':0, 'Bipolar':1, 'Hook':2, 'Scissors':3, 'Clipper':4, 'Irrigator':5, 'SpecimenBag':6]
#                 if os.path.exists(bbox_name):
#                     bboxes = np.loadtxt(bbox_name)
#                     if len(bboxes.shape) > 1:
#                         coor = []
#                         cls = []
#                         for box in bboxes:
#                             c, x, y, w, h = box[0], box[1], box[2], box[3], box[4]
#                             coor.append([x,y,w,h])
#                             cls.append(c)
#                         this_box.append(coor)
#                         this_box_class.append(cls)
#                     else:
#                         assert len(bboxes.shape) == 1
#                         c, x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3], bboxes[4]
#                         this_box.append([[x, y, w, h]])
#                         this_box_class.append([c])
#                 else:
#                     print('no detection file, boxes is none')
#                     bboxes = []
#                     this_box.append(bboxes)
#                     this_box_class.append(bboxes)
#                     continue
#             assert len(this_box) == vid_length
#             assert len(this_box) == len(this_box_class)
#             box_seq_list[vid] = this_box # (N, seq_len, K, 4) seq_len & K are variable
#             box_class_seq_list[vid] = this_box_class # (N, seq_len, K, 4) seq_len & K are variable
#
#
#     json_str = json.dumps(box_seq_list)
#     with open('./data/annotations/box_seq_list.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     json_str = json.dumps(box_class_seq_list)
#     with open('./data/annotations/box_class_seq_list.json', 'w') as json_file:
#         json_file.write(json_str)

import csv
def bbox2tool_presence(img_path='/home/kun/Desktop/miccai21/data/images/', bbox_anno='/home/kun/Desktop/miccai21/data/annotations/bbox_annotations_5_4/',
                       output='/home/kun/Desktop/miccai21/data/annotations/tool_annotations_yolo_5_5/', tool_annotation='/home/kun/Desktop/miccai21/data/annotations/tool_annotations/'):
    '''
    For each video, create a video**-tool.txt
    Calculate the accuracy for each video
    Format:
    Frame	Grasper  	Bipolar	  Hook	  Scissors	  Clipper	Irrigator	SpecimenBag
    0	1	0	0	0	0	0	0
    25	1	0	0	0	0	0	0
    50	1	0	0	0	0	0	0
    75	1	0	0	0	0	0	0
    100	0	0	0	0	0	0	0
    125	0	0	0	0	0	0	0
    150	0	0	0	0	0	0	0
    '''
    for sub_folder in os.listdir(img_path):

        for vid in os.listdir(os.path.join(img_path, sub_folder)):
            vid_name_img = os.path.join(img_path, sub_folder, vid)

            vid_length = len(os.listdir(vid_name_img))

            wfile = open(output+vid+'-tool.txt', 'w')
            init_row = "Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n"
            wfile.writelines(init_row)

            img_list = os.listdir(vid_name_img)
            img_list.sort(key=lambda x: int(x[:-4]))


            with open(tool_annotation+vid+'-tool.txt', "r") as f:
                tool_sig = []
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)
                for i, row in enumerate(reader):
                    tool_sig.append([int(row[x]) for x in [1, 2, 3, 4, 5, 6, 7]])
                real_tool_sig = tool_sig
            predict_tool_sig = []

            for frame in img_list:
                img_name = os.path.join(vid_name_img, frame)
                frame_id = img_name[-12:-4]
                bbox_name = os.path.join(bbox_anno, vid, frame_id+'.txt')

                frame_num = str(int(frame_id)*25)

                # ['Grasper':0, 'Bipolar':1, 'Hook':2, 'Scissors':3, 'Clipper':4, 'Irrigator':5, 'SpecimenBag':6]
                if os.path.exists(bbox_name):
                    li = np.zeros(shape=(7,))
                    bboxes = np.loadtxt(bbox_name)
                    if len(bboxes.shape) == 1:
                        bboxes = bboxes[np.newaxis, :]
                    presence = bboxes[:, 0]
                    for i in presence:
                        i = int(i)
                        li[i] = 1
                else:
                    li = np.zeros(shape=(7,)).astype(np.int).tolist()
                predict_tool_sig.append(li)

                li = [str(int(i)) for i in li]
                li = "\t".join(li)
                row = frame_num+"\t"+li+'\n'
                wfile.writelines(row)
            wfile.close()

            # Plot tool presence for gt and predictions
            tool_ids = [0,1,2,3,4,5,6]
            fig, axes = plt.subplots(len(tool_ids))

            for ind, (ax, tool_id) in enumerate(zip(axes, tool_ids)):

                y_li = []
                t_li = []
                for idx, row in enumerate(predict_tool_sig[:-1]):
                    y_li.append(row[ind])
                    t_li.append(real_tool_sig[idx][ind])
                x = np.arange(len(y_li))
                l1 = ax.plot(x, y_li, label='yolo', linewidth=1, color='red')

                l1 = ax.plot(x, t_li, label='gt', linewidth=1, color='black')

                ax.legend(fontsize=5, loc='upper right')
            plt.savefig('/home/kun/Desktop/miccai21/data/visual_yolo_tool/' + vid+'.png')
            plt.show()




def bbox2tool_presence_map(img_path='/home/kun/Desktop/miccai21/data/images/', bbox_anno='/home/kun/Desktop/miccai21/data/annotations/bbox_annotations_5_4_conf/',
                           tool_annotation='/home/kun/Desktop/miccai21/data/annotations/tool_annotations/'):
    '''
    For each video, create a video**-tool.txt
    Calculate the accuracy for each video
    Format:
    Frame	Grasper  	Bipolar	  Hook	  Scissors	  Clipper	Irrigator	SpecimenBag
    0	1	0	0	0	0	0	0
    25	1	0	0	0	0	0	0
    50	1	0	0	0	0	0	0
    75	1	0	0	0	0	0	0
    100	0	0	0	0	0	0	0
    125	0	0	0	0	0	0	0
    150	0	0	0	0	0	0	0
    '''
    from sklearn.metrics import precision_score, accuracy_score, average_precision_score, recall_score, confusion_matrix, plot_confusion_matrix

    y_true = []
    y_score = []
    for sub_folder in os.listdir(img_path):

        for vid in os.listdir(os.path.join(img_path, sub_folder)):
            if int(vid[-2:]) > 40:
                continue
            vid_name_img = os.path.join(img_path, sub_folder, vid)
            img_list = os.listdir(vid_name_img)
            img_list.sort(key=lambda x: int(x[:-4]))


            with open(tool_annotation+vid+'-tool.txt', "r") as f:
                tool_sig = []
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)
                for i, row in enumerate(reader):
                    tool_sig.append([int(row[x]) for x in [1, 2, 3, 4, 5, 6, 7]])
                real_tool_sig = tool_sig # (N, 7)
            img_list = img_list[:-1]
            assert len(real_tool_sig) == len(img_list)
            for idx, frame in enumerate(img_list):
                frame_id = os.path.join(vid_name_img, frame)[-12:-4]
                bbox_name = os.path.join(bbox_anno, vid, frame_id+'.txt')

                # ['Grasper':0, 'Bipolar':1, 'Hook':2, 'Scissors':3, 'Clipper':4, 'Irrigator':5, 'SpecimenBag':6]
                if os.path.exists(bbox_name):
                    li = np.zeros(shape=(7,))
                    bboxes = np.loadtxt(bbox_name)
                    if len(bboxes.shape) == 1:
                        bboxes = bboxes[np.newaxis, :]
                    classses = np.unique(bboxes[:, 0])
                    conf = bboxes[:, -1]
                    for im, i in enumerate(classses):
                        i = int(i)
                        li[i] = conf[im]
                    # print(bboxes[:, :2], li, real_tool_sig[idx])
                    # print(li, real_tool_sig[idx])
                    y_score.append(li)
                    y_true.append(real_tool_sig[idx])
                # else:
                #     y_score.append([0,0,0,0,0,0,0])
                #     y_true.append(real_tool_sig[idx])

    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    for i in range(7):
        print(average_precision_score(y_true[:, i], y_score[:, i], pos_label=1))

if __name__ == '__main__':
    # Generate Geometric Feature
    # bbox2geo()


    # Generate files for each video containing tool presence signal
    # Generate tool presence signal comparison plots
    # bbox2tool_presence()

    # Evaluate the mAP from gt tool presence signal and predicted tool presence signal
    bbox2tool_presence_map()
    exit()

    # Visualize the geo pair
    # visual_bbox2geo()



