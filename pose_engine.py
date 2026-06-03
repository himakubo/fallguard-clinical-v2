import mediapipe as mp
import numpy as np
import logging
import urllib.request
import os

log = logging.getLogger(__name__)
_pose = None

LM = {
    "NOSE":0,"LEFT_EYE":2,"RIGHT_EYE":5,"LEFT_EAR":7,"RIGHT_EAR":8,
    "LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,
    "LEFT_HIP":23,"RIGHT_HIP":24,
    "LEFT_KNEE":25,"RIGHT_KNEE":26,
    "LEFT_ANKLE":27,"RIGHT_ANKLE":28,
    "LEFT_HEEL":29,"RIGHT_HEEL":30,
    "LEFT_TOE":31,"RIGHT_TOE":32,
}
VIS = 0.35

def init():
    global _pose
    model_path = os.path.expanduser("~/fallguard/models/pose_landmarker_lite.task")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        log.info("Downloading pose landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            model_path
        )
        log.info("Model downloaded")
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _pose = vision.PoseLandmarker.create_from_options(opts)
    log.info("MediaPipe PoseLandmarker ready")

def process(rgb_frame, patient_bbox=None):
    global _pose
    if _pose is None:
        init()
    h, w = rgb_frame.shape[:2]
    if patient_bbox:
        nx1,ny1,nx2,ny2 = patient_bbox
        x1=max(0,int(nx1*w)-20); y1=max(0,int(ny1*h)-20)
        x2=min(w,int(nx2*w)+20); y2=min(h,int(ny2*h)+20)
        crop=rgb_frame[y1:y2,x1:x2]; crop_w,crop_h=x2-x1,y2-y1
    else:
        crop=rgb_frame; x1=y1=0; crop_w,crop_h=w,h
    if crop.size==0: return None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = _pose.detect(mp_image)
    if not result.pose_landmarks or len(result.pose_landmarks)==0: return None
    lms = result.pose_landmarks[0]

    def avg_vis(indices):
        xs=[]; ys=[]
        for i in indices:
            lm=lms[i]
            if lm.visibility>=VIS:
                xs.append((lm.x*crop_w+x1)/w)
                ys.append((lm.y*crop_h+y1)/h)
        if not xs: return None,None
        return float(np.mean(xs)),float(np.mean(ys))

    hx,hy=avg_vis([LM["NOSE"],LM["LEFT_EYE"],LM["RIGHT_EYE"],LM["LEFT_EAR"],LM["RIGHT_EAR"]])
    tx,ty=avg_vis([LM["LEFT_SHOULDER"],LM["RIGHT_SHOULDER"],LM["LEFT_HIP"],LM["RIGHT_HIP"]])
    fx,fy=avg_vis([LM["LEFT_ANKLE"],LM["RIGHT_ANKLE"],LM["LEFT_HEEL"],LM["RIGHT_HEEL"],LM["LEFT_TOE"],LM["RIGHT_TOE"]])

    nose=lms[LM["NOSE"]]; la=lms[LM["LEFT_ANKLE"]]; ra=lms[LM["RIGHT_ANKLE"]]
    av=(la.y+ra.y)/2 if la.visibility>VIS and ra.visibility>VIS else la.y if la.visibility>VIS else ra.y
    body_angle=min(90.0,float(abs(nose.y-av)*180))
    if body_angle<20: posture="LYING"
    elif body_angle<40: posture="RECLINING"
    elif body_angle<65: posture="SITTING"
    else: posture="STANDING"

    raw=[[(lms[i].x*crop_w+x1)/w,(lms[i].y*crop_h+y1)/h] for i in range(min(17,len(lms)))]
    motion=0.0
    if hasattr(process,"_prev") and process._prev is not None:
        motion=float(min(100.0,np.mean(np.abs(np.array(raw)-np.array(process._prev)))*2000))
    process._prev=raw

    out_lms=[]
    for name,idx in LM.items():
        if idx<len(lms):
            lm=lms[idx]
            out_lms.append({"name":name,"x":round((lm.x*crop_w+x1)/w,4),"y":round((lm.y*crop_h+y1)/h,4),"visibility":round(lm.visibility,3)})

    return {"landmarks":out_lms,"head":[hx,hy],"torso":[tx,ty],"feet":[fx,fy],
            "body_angle":round(body_angle,1),"motion_score":round(motion,1),"posture":posture}
