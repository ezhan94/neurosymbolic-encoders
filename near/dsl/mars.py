import torch
from .library_functions import AffineFeatureSelectionFunction
import pickle
import numpy as np

# Hard coded for SVD 5
MARS_FEATURE_SUBSETS = {
    "res_position" : torch.LongTensor([0, 1]),
    "intr_position" : torch.LongTensor([9, 10]),    
}
# 18 states, 18 actions
MARS_FULL_FEATURE_DIM = 36

PI = 3.1415926

svd_computer_path = 'datasets/mouse/data/svd/svd_computer.pkl'
mean_path = 'datasets/mouse/data/svd/mean.pkl'

svd_loaded = False     

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[1] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.multiply(data, scale) + shift

def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, 7, 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, 14)


def transform_svd_to_keypoints(data):

    global svd_computer
    global mean
    global svd_loaded
    if not svd_loaded:
        with open(svd_computer_path, 'rb') as f:
            svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            mean = pickle.load(f)
        svd_loaded = True

    num_components = data.shape[1] // 2
    resident_center = data[:, :2]
    resident_rotation = data[:, 2:4]
    resident_components = data[:, 4:num_components]
    intruder_center = data[:, num_components:num_components + 2]
    intruder_rotation = data[:, num_components + 2:num_components + 4]
    intruder_components = data[:, num_components + 4:]

    resident_keypoints = svd_computer.inverse_transform(resident_components)
    intruder_keypoints = svd_computer.inverse_transform(intruder_components)

    if mean is not None:
        resident_keypoints = resident_keypoints + mean
        intruder_keypoints = intruder_keypoints + mean

    # Compute rotation angle from sine and cosine representation
    resident_rotation = np.arctan2(
        resident_rotation[:, 0], resident_rotation[:, 1])
    intruder_rotation = np.arctan2(
        intruder_rotation[:, 0], intruder_rotation[:, 1])

    resident_keypoints = unnormalize_keypoint_center_rotation(
        resident_keypoints, resident_center, resident_rotation)
    intruder_keypoints = unnormalize_keypoint_center_rotation(
        intruder_keypoints, intruder_center, intruder_rotation)

    data = np.concatenate([resident_keypoints, intruder_keypoints], axis=-1)


    return data



def get_angle(Ax, Ay, Bx, By):
    angle = (np.arctan2(Ax - Bx, Ay - By) + np.pi/2) % (np.pi*2)
    return angle

def social_angle(x1, y1, x2, y2):
    #input batch x keypoints
    x_dif = np.mean(x1, axis = 1) - np.mean(x2, axis = 1)
    y_dif = np.mean(y1, axis = 1) - np.mean(y2, axis = 1)
    theta = (np.arctan2(y_dif, x_dif) + 2*np.pi) % 2*np.pi

    ori_body = get_angle(x1[:, 6], y1[:, 6], x1[:, 3], y1[:, 3])
    ang = np.mod(theta - ori_body, 2*np.pi)
    return np.minimum(ang, 2*np.pi - ang)


def dist_nose_nose(x1, y1, x2, y2):
    x_dif = x1[:,0] - x2[:,0]
    y_dif = y1[:,0] - y2[:,0]
    return np.linalg.norm(np.stack([x_dif, y_dif]), axis = 0)

def dist_nose_tail(x1, y1, x2, y2):
    x_dif = x1[:,0] - x2[:,6]
    y_dif = y1[:,0] - y2[:,6]
    return np.linalg.norm(np.stack([x_dif, y_dif]), axis = 0)


def interior_angle(p0, p1, p2):

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p1) - np.array(p2)


    return np.arctan2(np.linalg.det(np.stack([v0, v1], axis = 1)), np.sum(v0*v1, axis = 1))


#############################################
# 1 DIM LFs
#############################################

class ResMARSHeadBodyAngleComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="ResMARSHeadBodyAngleCompute",
            compute_function = self.head_body_angle_compute)

    def head_body_angle_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  

        angle_1 = interior_angle(keypoints_resident[:, 0], keypoints_resident[:, 3], keypoints_resident[:,6])

        label_tensor = torch.from_numpy(np.array(angle_1)[:, np.newaxis])

        label_tensor = label_tensor.to(batch.device).float()

        return label_tensor      


class IntrMARSHeadBodyAngleComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="IntrMARSHeadBodyAngleCompute",
            compute_function = self.head_body_angle_compute)

    def head_body_angle_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  
        
        angle_2 = interior_angle(keypoints_intruder[:, 0], keypoints_intruder[:, 3], keypoints_intruder[:,6])

        label_tensor = torch.from_numpy(np.array(angle_2)[:, np.newaxis])

        label_tensor = label_tensor.to(batch.device).float()

        return label_tensor      


class MARSNoseNoseDistanceComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="MARSNoseNoseDistCompute",
            compute_function = self.nose_nose_compute)

    def nose_nose_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  
        
        distance_1 = dist_nose_nose(keypoints_resident[:, :, 0], keypoints_resident[:, :, 1],
            keypoints_intruder[:, :, 0], keypoints_intruder[:, :, 1])

        label_tensor = torch.from_numpy(np.array(distance_1))[:, np.newaxis]

        label_tensor = label_tensor.to(batch.device).float()  

        return label_tensor              


class MARSNoseTailDistanceComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="MARSNoseTailDistCompute",
            compute_function = self.nose_tail_compute)

    def nose_tail_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  
        
        distance_2 = dist_nose_tail(keypoints_resident[:, :,0], keypoints_resident[:, :, 1],
            keypoints_intruder[:, :,0], keypoints_intruder[:, :, 1])

        label_tensor = torch.from_numpy(np.array(distance_2))[:, np.newaxis]

        label_tensor = label_tensor.to(batch.device).float()  

        return label_tensor        


class ResMARSSocialAngleComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="ResSocialAngleCompute",
            compute_function = self.social_angle_compute)

    def social_angle_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  
        
        angle_1  = social_angle(keypoints_resident[:, :, 0], keypoints_resident[:, :, 1],
            keypoints_intruder[:, :, 0], keypoints_intruder[:, :, 1])

        label_tensor = torch.from_numpy(np.array(angle_1))[:, np.newaxis]

        label_tensor = label_tensor.to(batch.device).float()

        return label_tensor            


class IntrMARSSocialAngleComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="IntrSocialAngleCompute",
            compute_function = self.social_angle_compute)

    def social_angle_compute(self, batch):
        # Batch has 18 states, 18 actions

        keypoints = transform_svd_to_keypoints(batch.cpu().numpy()[:, :18])
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[:, :14].reshape((-1, 7, 2))
        keypoints_intruder = keypoints[:, 14:].reshape((-1, 7, 2))  
        
        angle_2 = social_angle(keypoints_intruder[:, :, 0], keypoints_intruder[:, :, 1],
            keypoints_resident[:, :, 0], keypoints_resident[:, :, 1])

        label_tensor = torch.from_numpy(np.array(angle_2))[:, np.newaxis]

        label_tensor = label_tensor.to(batch.device).float()

        return label_tensor            


class ResMARSSpeedComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="ResSpeedCompute",
            compute_function = self.speed_compute)

    def speed_compute(self, batch):
        # Batch has 18 states, 18 actions

        return torch.norm(batch[:, 18:20], dim = -1, keepdim = True)

class IntrMARSSpeedComputation(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM 
        # This is just a size placeholder.       
        self.feature_tensor = torch.LongTensor([0])
        super().__init__(input_size, output_size, num_units, name="IntrSpeedCompute",
            compute_function = self.speed_compute)

    def speed_compute(self, batch):
        # Batch has 18 states, 18 actions

        return torch.norm(batch[:, 27:29], dim = -1, keepdim = True) 