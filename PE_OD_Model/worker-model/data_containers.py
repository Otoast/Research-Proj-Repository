import cv2
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, MMPoseInferencer
from tqdm import tqdm
import torch
import numpy as np
torch.cuda.set_device(0) # Setting it to GPU. 


class REBA_Tables:
    def __init__(self):
        self.TABLE_A = {
            1: {  # Trunk posture score = 1
                1: {1: 1, 2: 2, 3: 3, 4: 4},  # Neck posture score = 1
                2: {1: 2, 2: 3, 3: 4, 4: 5},  # Neck posture score = 2
                3: {1: 3, 2: 4, 3: 5, 4: 6},  # Neck posture score = 3
            },
            2: {  # Trunk posture score = 2
                1: {1: 2, 2: 3, 3: 4, 4: 5},
                2: {1: 3, 2: 4, 3: 5, 4: 6},
                3: {1: 4, 2: 5, 3: 6, 4: 7},
            },
            3: {  # Trunk posture score = 3
                1: {1: 3, 2: 4, 3: 5, 4: 6},
                2: {1: 4, 2: 5, 3: 6, 4: 7},
                3: {1: 5, 2: 6, 3: 7, 4: 8},
            },
            4: {  # Trunk posture score = 4
                1: {1: 4, 2: 5, 3: 6, 4: 7},
                2: {1: 5, 2: 6, 3: 7, 4: 8},
                3: {1: 6, 2: 7, 3: 8, 4: 9},
            },
            5: {  # Trunk posture score = 5
                1: {1: 5, 2: 6, 3: 7, 4: 8},
                2: {1: 6, 2: 7, 3: 8, 4: 9},
                3: {1: 7, 2: 8, 3: 9, 4: 9},
            }
        }
        self.TABLE_B = {
            1: {  # Upper arm posture score = 1
                1: {1: 1, 2: 2, 3: 2, 4: 3},  # Lower arm posture score = 1
                2: {1: 2, 2: 2, 3: 3, 4: 4},  # Lower arm posture score = 2
                3: {1: 2, 2: 3, 3: 4, 4: 4},  # Lower arm posture score = 3
            },
            2: {  # Upper arm posture score = 2
                1: {1: 2, 2: 2, 3: 3, 4: 4},
                2: {1: 2, 2: 3, 3: 4, 4: 5},
                3: {1: 3, 2: 4, 3: 4, 4: 5},
            },
            3: {  # Upper arm posture score = 3
                1: {1: 3, 2: 3, 3: 4, 4: 4},
                2: {1: 3, 2: 4, 3: 4, 4: 5},
                3: {1: 4, 2: 5, 3: 5, 4: 6},
            },
            4: {  # Upper arm posture score = 4
                1: {1: 4, 2: 4, 3: 5, 4: 5},
                2: {1: 4, 2: 5, 3: 5, 4: 6},
                3: {1: 5, 2: 6, 3: 6, 4: 7},
            },
            5: {  # Upper arm posture score = 5
                1: {1: 5, 2: 5, 3: 6, 4: 6},
                2: {1: 5, 2: 6, 3: 6, 4: 7},
                3: {1: 6, 2: 7, 3: 7, 4: 8},
            },
            6: {  # Upper arm posture score = 6
                1: {1: 7, 2: 7, 3: 8, 4: 8},
                2: {1: 7, 2: 8, 3: 8, 4: 9},
                3: {1: 8, 2: 9, 3: 9, 4: 10},
            }
        }
        self.TABLE_C = {
            1: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
            2: {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 10},
            3: {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 10, 10: 10},
            4: {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 10, 9: 10, 10: 10},
            5: {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
            6: {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
            7: {1: 7, 2: 8, 3: 9, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
            8: {1: 8, 2: 9, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
            9: {1: 9, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
            10: {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10},
        }
    def table_a(self, trunk_score, neck_score, leg_score):
        return self.TABLE_A[trunk_score][neck_score][leg_score]
    def table_b(self, upper_arm_score, lower_arm_score, wrist_score):
        return self.TABLE_B[upper_arm_score][lower_arm_score][wrist_score]
    def table_c(self, table_a_score, table_b_score):
        return self.TABLE_C[table_a_score][table_b_score]

class PredictionDataContainer:
    def __init__(self, data, pose_confidence_threshold=0.5, object_confidence_threshold=0.5):
        self.data = data
        self.pose_confidence_threshold = pose_confidence_threshold
        self.object_confidence_threshold = object_confidence_threshold
        self.NA = [np.nan, np.nan]
        self.reba_tables = REBA_Tables()
        pose_data = data['pose']['predictions'][0][0]
        keypoints = pose_data['keypoints']
        confidence =  pose_data['keypoint_scores']
        # Replace all keypoints with less than threshold confidence threshold with None
        self.pose_keypoints = np.array([
            self.NA if confidence[i] < self.pose_confidence_threshold else keypoints[i] for i in range(len(confidence))
        ])
        object_data = self.data['object'][0]
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        class_names = object_data.names.keys()
        obj_cls_pred = object_data.boxes.cls
        confidence = object_data.boxes.conf
        point_boxes = object_data.boxes.xywh
        bbox_boxes = object_data.boxes.xyxy
        class_points = dict()
        class_bbox = dict()
        for cls_name in class_names:
            cls_indx = torch.where(obj_cls_pred == cls_name)
            # Get top 1 prediction if spine top or bot, else get top 2
            i = -1 if cls_name in [1, 2] else -2
            i = torch.argsort(confidence[cls_indx])[i:]
            # select the top 2 or 1 boxes that correspond with the classname
            c_boxes = point_boxes[cls_indx]
            c_boxes = c_boxes[i]
            # turns the xy width height to points 
            points = torch.stack((c_boxes[:, 0], c_boxes[:, 1]), dim=1)
            class_points[cls_name] = np.array(points.tolist())
            # Getting bbox
            c_boxes = bbox_boxes[cls_indx]
            c_boxes = c_boxes[i]
            class_bbox[cls_name] = np.array(c_boxes.tolist())
        self.exoskel_keypoints = class_points
        self.exoskel_bbox = class_bbox

        self.MPII_KEYPOINT_INDEX = {
            0: "Right Ankle", 1: "Right Knee", 2: "Right Hip", 3: "Left Hip", 4: "Left Knee",
            5: "Left Ankle", 6: "Pelvis", 7: "Thorax", 8: "Neck", 9: "Head Top", 10: "Right Wrist",
            11: "Right Elbow", 12: "Right Shoulder", 13: "Left Shoulder", 14: "Left Elbow", 15: "Left Wrist"
        }
        self.MPII_KEYPOINT_NAMES = {
            "Right Ankle": 0, "Right Knee": 1, "Right Hip": 2,
            "Left Hip": 3, "Left Knee": 4, "Left Ankle": 5,
            "Pelvis": 6, "Thorax": 7, "Neck": 8, "Head Top": 9,
            "Right Wrist": 10, "Right Elbow": 11, "Right Shoulder": 12,
            "Left Shoulder": 13, "Left Elbow": 14, "Left Wrist": 15
        }

        self.MPII_SKELETON = [
            (0, 1),   # Right Ankle -> Right Knee
            (1, 2),   # Right Knee -> Right Hip
            (2, 6),   # Right Hip -> Pelvis
            (3, 4),   # Left Hip -> Left Knee
            (4, 5),   # Left Knee -> Left Ankle
            (3, 6),   # Left Hip -> Pelvis
            (6, 7),   # Pelvis -> Thorax
            (7, 8),   # Thorax -> Neck
            (8, 9),   # Neck -> Head Top
            (10, 11), # Right Wrist -> Right Elbow
            (11, 12), # Right Elbow -> Right Shoulder
            (12, 7),  # Right Shoulder -> Thorax
            (13, 14), # Left Shoulder -> Left Elbow
            (14, 15), # Left Elbow -> Left Wrist
            (13, 7)   # Left Shoulder -> Thorax
        ]

    def check_exoskeleton_fit(self):
        """
            Provides a dictionary containing whether the exoskeleton is well-fit in particular tracked keypoint areas. These areas are: 
            
            {Leg Strips, Shoulder Straps, Exoskeleton Spine, Leg Straps} <--- Also the keys of the dict.
            
            If any keypoint is not visible or unavailable, the model will assume proper fitting.
                Paramters: 
                    None
                Returns: 
                    dict of True/False conditions
        """
        fits = dict()
        fits['Leg Strips'] = self._strips_is_fitted()
        fits['Shoulder Straps'] = self._shoulder_is_fitted()
        fits['Exoskeleton Spine'] = self._exo_spine_is_fitted()
        fits['Leg Straps'] = self._leg_strap_is_fitted()    
        return fits
    def check_pose_reba(self):
        """
            Provides the dictionary containing the Reba Score of particular keypoints in the person's pose, and the final reba score. Also includes the Table A and B scores.
            If any critical keypoint is missing for a particular score, the model will assume minimum risk for that particular category.
                
                Parameters: 
                    None
                Returns
                    Dict of keys -> {Leg, Lower Arm, Neck, Trunk, Upper Arm, Table A Score, Table B Score, REBA Score}, each with values of type int.
        """
        reba = dict()
        reba['Leg'] = self._calc_leg_score()
        reba['Lower Arm'] = self._calc_lower_arm_score()
        reba['Neck'] = self._calc_neck_score()    
        reba['Trunk'] = self._calc_trunk_score()
        reba['Upper Arm'] = self._calc_upper_arm_score()
        reba['Table A Score'] = self.reba_tables.table_a(reba['Trunk'], reba['Neck'], reba['Leg'])
        reba['Table B Score'] = self.reba_tables.table_b(reba['Upper Arm'], reba['Lower Arm'], 1)
        reba['REBA Score'] = self.reba_tables.table_c(reba['Table A Score'], reba['Table B Score'])
        return reba
    
    def plot(self, img:np.ndarray=None) -> np.ndarray:
        """
            Plots points on original image. If the img parameter is set, will plot on the provided image instead. Please ensure similar dimensions between the original image and the provided image if provided. The function does not check for dimensions

                Parameters:
                    img (optional) -> numpy array
                Returns:
                    plotted_img -> numpy array 
        """
        data = self.data
        if img is None or type(img) is not np.ndarray:
            img = data['object'][0].orig_img
        img = self.plot_pose_points(img)
        img = self.plot_exoskeleton_points(img)
        return img
    
    def plot_pose_points(self, img):
        """
            Plots pose points on original image. Please ensure similar dimensions between the original image and the provided one for proper plotting.
                Parameters:
                    img -> numpy array
                Returns:
                    plotted_img -> numpy array 
        """
        new_frame = img.copy()
        for joint in self.MPII_SKELETON:
            point1 = self.pose_keypoints[joint[0]]
            point2 = self.pose_keypoints[joint[1]]
            if np.isnan(point1[0]) or np.isnan(point2[0]): continue
            point1 = [int(round(x)) for x in point1]
            point2 = [int(round(x)) for x in point2]
            new_frame = cv2.line(new_frame, point1, point2,  (255,255,255), 3)
            new_frame = cv2.circle(new_frame, point1, 8, (0, 0, 255), -1)
            new_frame = cv2.circle(new_frame, point2, 8, (0, 0, 255), -1)
        return new_frame
    def plot_exoskeleton_points(self, img):
        """
            Plots exoskeleton points on original image. Please ensure similar dimensions between the original image and the provided one for proper plotting.
                Parameters:
                    img -> numpy array
                Returns:
                    plotted_img -> numpy array 
        """
        object_data = self.data['object'][0]
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        class_names = object_data.names.keys()
        class_points = self.exoskel_keypoints
        class_colors = [(255, 0, 127), (0, 0, 200), (0, 0, 200), (0, 200, 0), (255, 153, 51)]
        new_frame = img.copy()
        for n in class_names:
            if len(class_points[n]) == 0: continue
            for point in class_points[n]:
                point = point.round().astype(int)
                new_frame = cv2.circle(
                    img=new_frame, center=point, 
                    radius=8, color=class_colors[n], thickness=-1
                )
                x = int(point[0] - new_frame.shape[0] / 100)
                y = int(point[1] - new_frame.shape[1] / 100)
                if x < 0: x = 0
                if y < 0: y = 0
                new_frame = cv2.putText(
                    img=new_frame, text=object_data.names[n],
                    org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=class_colors[n], thickness=2
                )
        return new_frame
    
    def _calc_distance_point_line(self, point, line):
        pt1 = line[0]
        pt2 = line[1]
        top = np.cross(pt2-pt1,point-pt1)
        bottom = np.linalg.norm(pt2-pt1)
        return top / bottom

    def _is_facing_side(self, threshold=0.13):
        th = self.pose_keypoints[self.MPII_KEYPOINT_NAMES['Thorax']]
        p = self.pose_keypoints[self.MPII_KEYPOINT_NAMES[f'Pelvis']]
        lh = self.pose_keypoints[self.MPII_KEYPOINT_NAMES['Left Hip']]
        rh = self.pose_keypoints[self.MPII_KEYPOINT_NAMES['Right Hip']]
        spine = th - p
        ratio_hip_spine = list()
        for h in [lh, rh]:
            if np.isnan(h[0]):  continue
            ratio_hip_spine.append(self._vec_mag(h - p) / self._vec_mag(spine))
        if len(ratio_hip_spine) == 0: return 0
        ratio_hip_spine = sum(ratio_hip_spine) / len(ratio_hip_spine)
        return ratio_hip_spine < threshold
    
    def _vec_mag(self ,x:list) -> float:
        """Calculates the magnitude of a vector."""
        return np.sqrt(np.dot(x, x))
    
    def _calc_angle(self, line1:list, line2:list) -> float:
        """Calculates the angle between two vectors. Lines are in the format ((x1, y1), (x2, y2))"""
        bottom = self._vec_mag(line1) * self._vec_mag(line2)
        if bottom == 0: return 0
        v = np.dot(line1, line2) / bottom
        if v < -1: return 180
        elif v > 1: return 0
        return (180 / np.pi) * np.arccos(v)
    
    def _calc_neck_score(self):
        """ 
            Calculates elevated risk score inspired by REBA model guidelines. Checks the angle of neck to the thorax.
                Arguments: 
                    keypoints -> list | List of Keypoints above confidence threshold, length 16
                Returns: 
                    REBA Score -> int | Based on calculated angle.
        """
        keypoints = self.pose_keypoints
        k = self.MPII_KEYPOINT_NAMES
        # Calculate angle between neck head and lungs. If any one of them isn't present doesn't calculate
        neck = keypoints[k['Neck']]
        head_top = keypoints[k['Head Top']]
        thorax = keypoints[k['Thorax']]
        if any([np.isnan(x[0]) for x in [neck, head_top, thorax]]):
            return 1
        line1 = thorax - neck
        line2 = head_top - neck
        angle = 180 - self._calc_angle(line1, line2)
        # If angle is above a certain amount its really, really bad for the neck to do that
        # Either that person is really going throguh it or the model has some poor proportions
        if angle > 120: return 1
        elif angle > 20: return 2
        else: return 1

    def _calc_upper_arm_score(self):
        """ 
            Calculates elevated risk score inspired by REBA model guidelines. Checks the angle of the upper arm to the thorax -> pelvis connection.
            Will only consider the arm with the most flexion, and score based off of score.
                Arguments: 
                    keypoints -> list | List of Keypoints above confidence threshold, length 16
                Returns: 
                    REBA Score -> int | Based on calculated angle.
        """
        keypoints = self.pose_keypoints
        k = self.MPII_KEYPOINT_NAMES        
        th = keypoints[k['Thorax']]; pel = keypoints[k['Pelvis']]
        if np.isnan(th[0]) or np.isnan(pel[0]): return 1

        angle = list()
        for lr in ['Left', 'Right']:
            sh = keypoints[k[f'{lr} Shoulder']]; el = keypoints[k[f'{lr} Elbow']]
            if np.isnan(sh[0]) or np.isnan(el[0]): continue
            line1 = sh - el
            line2 = th - pel
            angle.append(self._calc_angle(line1, line2))
        if len(angle) == 0: return 1
        # Assuming REBA score cares about the angle with the worst score
        # Need to consider whether there is extension backwards
        angle = max(angle)
        thresholds = [20, 45, 90]
        for i in range(len(thresholds)):
            if angle < thresholds[i]: return i + 1
        return 4
    
    def _calc_lower_arm_score(self):
        """ 
            Calculates elevated risk score inspired by REBA model guidelines. Checks the angle of the upper arm to the wrist connection.
            Will only consider the arm with the most flexion, and score based off of score.
                Arguments: 
                    keypoints -> list | List of Keypoints above confidence threshold, length 16
                Returns: 
                    REBA Score -> int | Based on calculated angle.
        """
        keypoints = self.pose_keypoints
        k = self.MPII_KEYPOINT_NAMES
        angle = list()
        for lr in ['Left', 'Right']:
            sh = keypoints[k[f'{lr} Shoulder']]; el = keypoints[k[f'{lr} Elbow']]
            wr = keypoints[k[f'{lr} Wrist']]
            if any([np.isnan(x[0]) for x in [sh, wr, el]]): continue
            upper_arm = sh - el
            lower_arm = el - wr
            angle.append(self._calc_angle(upper_arm, lower_arm))
        if len(angle) == 0: return 1
        # Assuming REBA score cares about the angle with the worst score
        # Need to consider whether there is extension backwards
        angle = max(angle)
        if angle < 60 or angle > 100: return 2
        else: return 1

    def _calc_leg_score(self):
        """ 
            Calculates elevated risk score inspired by REBA model guidelines. Calculates the angle of the leg.
            Will only consider the arm with the most flexion, and score based off of score.
                Arguments: 
                    keypoints -> list | List of Keypoints above confidence threshold, length 16
                Returns: 
                    REBA Score -> int | Based on calculated angle.
        """
        # Left leg
        keypoints = self.pose_keypoints
        k = self.MPII_KEYPOINT_NAMES

        h = keypoints[k['Left Hip']]; kn = keypoints[k['Left Knee']]
        a = keypoints[k['Left Ankle']]
        l_angle = None
        r_angle = None
        if not any([np.isnan(x[0]) for x in [h, a, kn]]):       
            line1 = h - kn
            line2 = kn - a
            l_angle = self._calc_angle(line1, line2)

        h = keypoints[k['Right Hip']]; kn = keypoints[k['Right Knee']]
        a = keypoints[k['Right Ankle']]
        if not any([np.isnan(x[0]) for x in [h, a, kn]]):  
            line1 = h - kn
            line2 = kn - a
            r_angle = self._calc_angle(line1, line2)
        # If both legs located
        if l_angle is None and r_angle is None: return 1
        noneto0 = lambda x: 0 if x is None else x
        
        angle = max([noneto0(l_angle), noneto0(r_angle)])
        if angle < 60: return 1
        else:   return 2

    def _calc_trunk_score(self):
        """ 
            Calculates elevated risk score inspired by REBA model guidelines. Calculates the angle of the spine. If the person is facing their side, will instead calculate whether the person is leaning forward \
            or backwards. 
            Will only consider the arm with the most flexion, and score based off of score.
                Arguments: 
                    keypoints -> list | List of Keypoints above confidence threshold, length 16
                Returns: 
                    REBA Score -> int | Based on calculated angle.
        """
        # If the shoulders and hips are wide in stance, its likely that the person is toward or away the camera
        # As such, we can use hips to see side flexion of trunk/ In contrast, if shoulders and hips are shorter, makes more sense to use
        # Legs to calculate angle width
        # If it is in a reigon of uncertainty, we should just leave it alone for the most part 
        k = self.MPII_KEYPOINT_NAMES
        keypoints = self.pose_keypoints
        lr = 'Left'
        h = keypoints[k[f'{lr} Hip']]
        kn = keypoints[k[f'{lr} Knee']]

        if np.isnan(h[0]) or np.isnan(kn[0]):          # If the left leg is not defined above knee
            lr = 'Right'
            h = keypoints[k[f'{lr} Hip']]
            kn = keypoints[k[f'{lr} Knee']]
            if np.isnan(h[0]) or np.isnan(kn[0]):   
                return 1
        line1 = keypoints[k['Thorax']] - keypoints[k['Pelvis']]
        if self._is_facing_side() < 0.2:  # We will assume fat people aren't typically construction workers
            line2 = keypoints[k[f'{lr} Hip']] - keypoints[k[f'{lr} Ankle']]
            angle = self._calc_angle(line1, line2)
            if angle > 60: return 4
            elif angle > 20: return 3

            if angle < -20: return 3
            elif angle < 0 or angle > 0: return 2        
        else:
            line2 = keypoints[k[f'{lr} Hip']] - keypoints[k['Pelvis']]
            angle = self._calc_angle(line1, line2)
            if angle > 30: return 1
        return 1


    def _shoulder_is_fitted(self) -> bool:
        if self._is_facing_side(0.18): 
            return 1 # If person is facing side we can't see alignment
        k = self.MPII_KEYPOINT_NAMES
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        sh_straps = np.array(self.exoskel_keypoints[0].tolist())
        neck = self.pose_keypoints[k['Neck']]
        thorax = self.pose_keypoints[k['Thorax']]
        if np.isnan(neck[0]) or np.isnan(thorax[0]) or len(sh_straps) < 2: return 1
        # Distance formula using two points for a line and a third point
        
        if self._vec_mag(neck - thorax) < 0.001: return 1 # Needs to be noticable distance between neck and thorax
        s = list()
        for strap in sh_straps:
            s.append(self._calc_distance_point_line(strap, [neck, thorax]))
        if len(s) < 2: return True
        l_bound = 0.5
        h_bound = 1.3

        ratio = abs(s[1] / s[0])
        if l_bound < ratio < h_bound:
            return True
        return False

    def _exo_spine_is_fitted(self) -> bool:
        pose_keypoints = self.pose_keypoints
        exoskel_keypoints = self.exoskel_keypoints
        if self._is_facing_side(0.19): 
            return 1 # If person is facing side we can't see alignment
        
        
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        spine_top = np.array(exoskel_keypoints[2].tolist())
        spine_bot = np.array(exoskel_keypoints[1].tolist())
        if len(spine_top) < 1 or len(spine_bot) < 1: return True
        exo_line = spine_top[0] - spine_bot[0]
        k = self.MPII_KEYPOINT_NAMES
        thorax = pose_keypoints[k['Thorax']]
        pelvis = pose_keypoints[k['Pelvis']]
   
        if (np.isnan(pelvis[0]) or np.isnan(thorax[0])):
            return True
        spine = thorax - pelvis
        angle = self._calc_angle(exo_line, spine)
        
        top_dist = abs(self._calc_distance_point_line(spine_top, [thorax, pelvis]) / self._vec_mag(spine))
        bot_dist = abs(self._calc_distance_point_line(spine_bot, [thorax, pelvis]) / self._vec_mag(spine))

        # Neither top dist or bot dist should be above 0.07, so adding them up allows for one easy check
        if angle > 2 or top_dist + bot_dist > 0.15: return False
        return True

    def _leg_strap_is_fitted(self) -> bool:
        pose_keypoints = self.pose_keypoints
        exoskel_keypoints = self.exoskel_keypoints
        k = self.MPII_KEYPOINT_NAMES
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        kn_straps = np.array(exoskel_keypoints[3].tolist())
        # Getting the x_values to check where are the straps closer located to
        r_hip = pose_keypoints[k['Right Hip']][0]
        r_knee = pose_keypoints[k['Right Knee']][0]
        l_hip = pose_keypoints[k['Left Hip']][0]
        l_knee =  pose_keypoints[k['Left Knee']][0]
        middle_rx = (r_hip + r_knee) / 2
        middle_lx = (l_hip + l_knee) / 2
        for strap in kn_straps:
            s = strap[0]
            lr = 'Left' if abs(middle_lx - s) < abs(middle_rx - s) else 'Right'
            hip = pose_keypoints[k[f'{lr} Hip']]
            knee = pose_keypoints[k[f'{lr} Knee']]
            # Get the projected strap point on the hip/knee line
            ab = knee - hip
            ap = strap - hip
            strap_projection = np.dot(ap, ab) / np.dot(ab, ab)
            strap_projection = hip + strap_projection * ab       
            ap = strap_projection - hip
            strap_hip_dist = self._vec_mag(ap)
            knee_hip_mag = self._vec_mag(ab)        
            percent_dist = strap_hip_dist / knee_hip_mag
            if percent_dist > 0.8:    return False
        return True

    def _strips_is_fitted(self) -> bool:
        pose_keypoints = self.pose_keypoints
        exoskel_bbox = self.exoskel_bbox
        k = self.MPII_KEYPOINT_NAMES
        thigh_mag = list()
        for lr in ['Right', 'Left']:
            hip = pose_keypoints[k[f'{lr} Hip']]
            kn = pose_keypoints[k[f'{lr} Knee']]
            if np.isnan(hip[0]) or np.isnan(kn[0]): continue
            thigh_mag.append(self._vec_mag(hip - kn))
        if len(thigh_mag) == 0: return True
        thigh_mag = max(thigh_mag)
        # 0 -> shoulder_straps, 1 -> spine_bot, 2 -> spine_top, 3 -> strap, 4 -> strip
        strips_bbox = exoskel_bbox[4]
        for strip in strips_bbox:
            strip_len = strip[3] - strip[1]
            thigh_strip_ratio = strip_len / thigh_mag
            if thigh_strip_ratio > 0.7: return False
        return True
    
