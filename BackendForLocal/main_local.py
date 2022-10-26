import os
from functools import partial
from flask import Flask,request,send_from_directory
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import numpy as np
import queue
import cv2
import torch
from yolo.darknet import Darknet
from utils import *
from FastPose.main_fast_inference import InferenNet_fast,Mscoco
from videopose3d.model import TemporalModel

app=Flask(__name__)


#输入：原始视频
#输出：numpy, shape=[batch,3,608,608]
def preProcessForYOLO(video):
    video.save('example.gif')
    cap=cv2.VideoCapture('example.gif')
    imgs=[]
    orig_imgs=[]
    im_dim_list=[]
    #读取视频每一帧并处理
    tag, img = cap.read()
    while tag:
        orig_img=img.copy()
        #app.logger.debug('orig_img',orig_img.shape)
        dim = orig_img.shape[1], orig_img.shape[0]
        im_dim_list.append(dim)
        orig_imgs.append(orig_img)
        #resize
        W,H=img.shape[1], img.shape[0]
        nW=int(W*min(608/W,608/H))
        nH = int(H * min(608 / W, 608 / H))
        nImg = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_CUBIC)
        canvas= np.full((608, 608, 3), 128)
        canvas[(608 - nH) // 2:(608 - nH) // 2 + nH, (608 - nW) // 2:(608 - nW) // 2 + nW, :] =nImg
        img=canvas
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()/255.0  #shape=[3,608,608]
        imgs.append(img)
        tag, img = cap.read()
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    return np.stack(imgs, axis=0),orig_imgs,im_dim_list


#对YOLO返回结果进行后处理
#输入0：prediction [batch,22743]
#输入1：ori_imgs []
#输出: ori_imgs,boxes,inps [batch,3,256,192]
def preProcessForFastPose(prediction,orig_imgs,im_dim_list):
    new_orig_imgs=[]
    all_boxes=[]
    all_scores=[]
    all_pt1=[]
    all_pt2=[]
    inpses=[]
    for i in range(0,len(orig_imgs)):
        orig_img=orig_imgs[i]
        pred=prediction[i]
        pred=torch.unsqueeze(pred,dim=0)
        dim_list=im_dim_list[i]
        dim_list=torch.unsqueeze(dim_list,dim=0)
        dets = dynamic_write_results(pred, 0.05, 80, nms=True, nms_conf=0.6)
        if isinstance(dets, int) or dets.shape[0] == 0:
            continue
        dets=dets.cpu()
        dim_list = torch.index_select(dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(608 / dim_list, 1)[0].view(-1, 1)

        # coordinate transfer
        dets[:, [1, 3]] -= (608 - scaling_factor * dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (608 - scaling_factor * dim_list[:, 1].view(-1, 1)) / 2

        dets[:, 1:5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, dim_list[j, 1])
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        for k in range(len(orig_img)):
            boxes_k = boxes[dets[:, 0] == k]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                continue
            inps = torch.zeros(boxes_k.size(0), 3, 320, 256)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)
            if boxes_k.size(0)>1:
                continue
            new_orig_imgs.append(orig_img)
            all_boxes.append(boxes_k)
            all_scores.append(scores[dets[:, 0] == k])
            all_pt1.append(pt1)
            all_pt2.append(pt2)
            inpses.append(inps)

    final_inps=[]
    new_boxes=[]
    new_scores=[]
    new_pt1=[]
    new_pt2=[]
    for i in range(0,len(new_orig_imgs)):
        orig_img=new_orig_imgs[i]
        inps=inpses[i]
        boxes_k=all_boxes[i]
        pt1=all_pt1[i]
        pt2=all_pt2[i]
        scores=all_scores[i]
        inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        inps, pt1, pt2 = crop_from_dets(inp, boxes_k, inps, pt1, pt2)
        final_inps.append(inps)
        new_boxes.append(boxes_k)
        new_scores.append(scores)
        new_pt1.append(pt1)
        new_pt2.append(pt2)
    return np.concatenate(final_inps,axis=0),new_boxes,new_scores,new_pt1,new_pt2

#输出：[total_frames,17,2]
def preProcessForVideoPose3D(hms,all_boxes,all_scores,all_pt1,all_pt2,orig_imgs):
    final_result=[]
    for i in range(hms.shape[0]):
        boxes=all_boxes[i]
        scores=all_scores[i]
        hm_data=hms[i]
        pt1=all_pt1[i]
        pt2=all_pt2[i]
        preds_hm, preds_img, preds_scores = getPrediction(hm_data, pt1, pt2, 320, 256, 80, 64)
        result = pose_nms(boxes, scores, preds_img, preds_scores)
        result = {'result': result}
        # print(result['result'])
        final_result.append(result)
        img = vis_frame(orig_imgs[i], result)
    kpts = []
    no_person = []
    for i in range(len(final_result)):
        if not final_result[i]['result']:  # No people
            no_person.append(i)
            kpts.append(None)
            continue
        kpt = max(final_result[i]['result'],
                  key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']
        kpts.append(kpt.data.numpy())
        for n in no_person:
            kpts[n] = kpts[-1]
        no_person.clear()
    for n in no_person:
        kpts[n] = kpts[-1] if kpts[-1] else kpts[n - 1]
    keypoints = np.array(kpts).astype(np.float32)
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    keypoints=np.pad(keypoints,((121,121),(0,0),(0,0)),'edge')
    keypoints=np.expand_dims(keypoints,axis=0)
    keypoints=np.concatenate((keypoints,keypoints),axis=0)
    keypoints[1,:,:,0]*=-1
    keypoints[1,:,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]]=keypoints[1,:,[2,4,6,8,10,12,14,16,1,3,5,7,9,11,13,15]]
    
    return keypoints

yolo_model = Darknet("YOLO/yolov3-spp.cfg")
yolo_model.load_weights('YOLO/weight/yolov3-spp.weights')
yolo_model.net_info['height'] = 608
yolo_model.cuda()
yolo_model.eval()
fastpose=InferenNet_fast(5,Mscoco())
videopose3d=TemporalModel(17,2,17,filter_widths=[3,3,3,3,3],causal=False,dropout=0.25,channels=1024,dense=False)
videopose3d.load_state_dict(torch.load('VideoPose3d/weights/weight.bin')['model_pos'])
videopose3d=videopose3d.cuda()
videopose3d.eval()

@app.route('/processVideo',methods=['POST','GET'])
def processVideo():
    video=request.files['video']
    imgs,orig_imgs,im_dim_list=preProcessForYOLO(video)

    temp_batch=2
    cnt=0
    preds=[]
    while cnt<imgs.shape[0]:
        leng=min(imgs.shape[0]-cnt,temp_batch)
        imgs_cuda = torch.from_numpy(imgs[cnt:cnt+leng].astype(np.float32)).cuda()
        with torch.no_grad():
            pred = yolo_model(imgs_cuda, CUDA=True)
            pred=pred.cpu()
        preds.append(pred)
        cnt+=temp_batch
    prediction=torch.from_numpy(np.concatenate(preds,axis=0))
    torch.cuda.empty_cache()
    inpses,all_boxes,all_scores,all_pt1,all_pt2=preProcessForFastPose(
        prediction,orig_imgs,im_dim_list)
    hms=[]
    for i in range(0,inpses.shape[0]):
        with torch.no_grad():
             hm=fastpose(torch.from_numpy(np.expand_dims(inpses[i],axis=0)).cuda())
        hms.append(hm.cpu())
    hms=np.concatenate(hms,axis=0)
    joints2d=preProcessForVideoPose3D(hms,all_boxes,all_scores,
     all_pt1,all_pt2,orig_imgs)

    with torch.no_grad():
         joints3d=videopose3d(torch.from_numpy(joints2d.astype(np.float32)).cuda())
    joints3d[1,:,:,0]*=-1
    joints3d[1,:,[4,5,6,11,12,13,1,2,3,14,15,16]]=joints3d[1,:,[1,2,3,14,15,16,4,5,6,11,12,13]]
    joints3d=torch.mean(joints3d,dim=0,keepdim=True)
    joints3d=joints3d.cpu().numpy()
    joints3d=joints3d.squeeze()
    ans=''
    for frame in range(0,joints3d.shape[0]):
        for joint in range(0,joints3d.shape[1]):
            for i in range(0,joints3d.shape[2]):
                ans+=str(joints3d[frame][joint][i])+','
    ans=ans[0:len(ans)-1]
    return ans


if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
