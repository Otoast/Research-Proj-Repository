import cv2
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, MMPoseInferencer
from tqdm import tqdm
import torch
import numpy as np
torch.cuda.set_device(0) # Setting it to GPU. 
from data_containers import *

class RiskPredictionModel:
    def __init__(self, checkpoint_filename, config_filename, yolo_model_filename, pose_confidence_threshold=0.5, object_confidence_threshold=0.5):
        self.checkpoint_file = checkpoint_filename
        self.config_file = config_filename
        self.yolo_model = YOLO(yolo_model_filename, task='detect')
        self.pose_model = MMPoseInferencer(pose2d=config_filename, pose2d_weights=checkpoint_filename, device='cuda:0')
        self.pose_confidence_threshold = pose_confidence_threshold
        self.object_confidence_threshold = object_confidence_threshold
        self.NA = [np.nan, np.nan]
    def _calc_distance_point_line(self, point, line):
        pt1 = line[0]
        pt2 = line[1]
        top = np.cross(pt2-pt1,point-pt1)
        bottom = np.linalg.norm(pt2-pt1)
        return top / bottom
    
    def predict(self, frame):
        """ 
        Estimates pose and key points for posture and exoskeleton object. Returns a dict containing the data for both models, with the keys {pose, object}.     
            Arguments:
                frame -> Numpy Array
            Returns:
                data_dict -> Estimations from Models
        """
        return PredictionDataContainer(
            {
                'pose': next(self.pose_model(frame, return_vis=True)), 
                'object': self.yolo_model.predict(frame, verbose=False, conf=self.object_confidence_threshold, iou=0.3)
            }, 
            self.pose_confidence_threshold, 
            self.object_confidence_threshold
        )
    def plot_fit_results(self, img, bad_fits_names:list, reba_score:int):
        x = int(5 * img.shape[1] / 100)
        y = int(5 * img.shape[0] / 100)
        new_frame = img.copy()
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"REBA Score {reba_score}:"
        font_scale = .009 * x;   thickness = 2
        (_, dy), _ = cv2.getTextSize(text, font, font_scale, thickness) 
        dy = 2 * int(dy + y / 200)

        add_text = lambda text: cv2.putText(new_frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)
        inc = lambda y, times: y + dy * times
        new_frame = add_text(text)
        y = inc(y, 1)
        if reba_score >= 4:
            new_frame = add_text("ELEVATED RISK. Immediate Remidiation is Reccomended")
            y = inc(y, 1)
        y = inc(y, 1)
        new_frame = add_text("Exoskeleton has fitting issues in the following places:")
        y = inc(y, 2)
        for name in bad_fits_names:
            new_frame = add_text(name)
            y = inc(y, 1)
        y = inc(y, 2)
        if len(bad_fits_names) > 0:
            new_frame = add_text("Please take corrective action.")
        return new_frame



def main():    
    print('#'*60 + ' SETTING UP VIDEO STREAM ' + '#'*60)
    output_name = "well_fitting.mp4"
    video_name = 'well_fitting.MOV'
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_vid = cv2.VideoWriter(
                                filename=output_name,
                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps=cap.get(cv2.CAP_PROP_FPS),
                                frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            )
    print('#'*60 + ' SETTING UP MODEL ' + '#'*67)
    checkpoint_file = "rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth"
    config_file = "rtmpose-m_8xb64-210e_mpii-256x256.py"
    yolo_filename = "exo_model.onnx"
    model = RiskPredictionModel(checkpoint_file, config_file, yolo_filename)
    
    print('#'*60 + " RUNNING MODEL ON STREAM " + '#'*60)
    successful_read, frame = cap.read()
    frames_bad = {
        'Shoulder Straps': [0, False], 'Leg Straps': [0, False], 'Exoskeleton Spine' : [0, False], 'Leg Strips' : [0, False]
    }
    bad_frame_limit = 10
    for i in tqdm(range(length)):
        if not successful_read: continue
        # Get prediction of both pose and object
        data = model.predict(frame)
        exoskel_fit = data.check_exoskeleton_fit()
        reba_pose = data.check_pose_reba()
        bad_fits_above_limit = list()
        for key in exoskel_fit.keys():
            if frames_bad[key][0] == 0: frames_bad[key][1] = False
            
            if not exoskel_fit[key]:
                frames_bad[key][0] = min(61, frames_bad[key][0] + 2)
            
            if frames_bad[key][0] >= bad_frame_limit or frames_bad[key][1]:
                if frames_bad[key][1] == False: frames_bad[key][0] = 61
                bad_fits_above_limit.append(key)
                frames_bad[key][1] = True
            frames_bad[key][0] = max(0, frames_bad[key][0] - 1)
        
        img = data.plot()
        img = model.plot_fit_results(img, bad_fits_above_limit, reba_pose['REBA Score'])
            


        # Creating image plotting both model predictions and putting into video for saving #
        # cv2.imshow('', cv2.resize(img, (900, 900)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break            
        output_vid.write(img)
        successful_read, frame = cap.read()
    cap.release()
    output_vid.release()
    print(f"Successful read. Output video is filename {output_name}")

if __name__ == '__main__':
    main()











# def main2():
#     print('#'*60 + ' SETTING UP VIDEO STREAM ' + '#'*60)
#     video_name = 'videos/bad_fit_bad_align_leg_straps.MOV'
#     cap = cv2.VideoCapture(0)
#     checkpoint_file = "rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth"
#     config_file = "rtmpose-m_8xb64-210e_mpii-256x256.py"
#     yolo_filename = "yolo_od_model.onnx"

#     model = RiskPredictionModel(checkpoint_file, config_file, yolo_filename)
#     print('#'*60 + " RUNNING MODEL ON STREAM " + '#'*60)
#     successful_read, frame = cap.read()
#     while True:
#         if not successful_read: break
#         # Get prediction of both pose and object
#         data = model.predict(frame)
#         prediction = model.plot(data)
#         prediction = cv2.resize(prediction ,(1920, 1080))
#         cv2.imshow('', prediction)
    
#         if cv2.waitKey(1) & 0xFF == ord('q'): break
#         cv2.resize(frame,(1920, 1080))
#         successful_read, frame = cap.read()
        
#     cap.release()