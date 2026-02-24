from math import sqrt, degrees, radians, asin, acos, atan2, atan
from robot_kinematics.forward_kinematics import DHForwardKinematics
from scipy.spatial.transform import Rotation
import numpy as np

"""
joint_0 (base): 90 - self.j0
joint_1 (prim): 90 - self.j1
joint_2 (seco): 180 - self.j2 
joint_3 (seco-t): 90 - self.j3
joint_4 (ter): 180 - self.j4
joint_5 (ter-t): 90 - self.j5
"""


class InverseKinematics(object):
    def __init__(self):
        """
        This class is used to perform inverse kinematics on a 6DOF Robot Arm
        """
        # end-effector properties
        self.end_eff_pos = None
        self.end_eff_rot = None

        # Arm properties
        self.pri_arm_origin = [[0], [0], [163.2]]

        self.pri_arm_length = [[0], [0], [236.5]]
        self.sec_arm_length = [[0], [97.5], [236.5]]
        self.ter_arm_length = [[0], [-255], [0]]

        # joint angles
        self.j0 = 0
        self.j1 = 0
        self.j2 = 0
        self.j3 = 0
        self.j4 = 0
        self.j5 = 0

        # forward kinematics
        theta = [radians(90), radians(90), radians(0), radians(-90)]
        alpha = [radians(-90), radians(0), radians(-90), radians(180)]
        radius = [0, - self.sec_arm_length[2][0], - self.sec_arm_length[1][0], 0]
        distance = [self.pri_arm_origin[0][0], 0, 0, self.get_arm_length(self.sec_arm_length)]

        self.fk = DHForwardKinematics(theta, alpha, radius, distance)

    # set end-effector position and rotation
    def set_end_effector(self, position: [float, float, float], rotation: [float, float, float]):
        """
        This class updates the position/rotation of the end effector

        Args:
            position: new position of the end effector
            rotation: new rotation of the end effector
        """
        self.end_eff_pos = position
        self.end_eff_rot = rotation

    # return the joint angles for end-effector properties
    def calc_inverse_kinematics(self):
        """
        This function performs the calculations for the inverse kinematics

        Return:
            the joint angles required to reach the desired end effector position
        """
        # calculate rotation matrix for end-effector
        end_eff_rot = Rotation.from_euler('xyz', [radians(self.end_eff_rot[0]), radians(self.end_eff_rot[1]), radians(self.end_eff_rot[2])])
        end_eff_mat = end_eff_rot.as_matrix()

        # get j0, j1, j2 target position
        target_3_mat = end_eff_mat @ self.ter_arm_length
        target_3_x = target_3_mat.item(0) + self.end_eff_pos[0]
        target_3_y = target_3_mat.item(1) + self.end_eff_pos[1]
        target_3_z = target_3_mat.item(2) + self.end_eff_pos[2]

        # base joint angle
        self.j0 = degrees(atan2(target_3_x, target_3_y))

        # geometry calculations
        hyp = sqrt(target_3_x ** 2 + target_3_y ** 2)
        dz = self.pri_arm_origin[2][0] - target_3_z

        d_hyp = sqrt(dz ** 2 + hyp ** 2)
        d_hyp_rot = degrees(asin(dz / d_hyp))

        # calculate arm lengths
        pri_arm_length = self.get_arm_length(self.pri_arm_length)
        sec_arm_length = self.get_arm_length(self.sec_arm_length)

        # end effector position is not in reach
        if pri_arm_length + sec_arm_length < d_hyp:
            print("effector out of reach")
            d_hyp = pri_arm_length + sec_arm_length

        # calculate primary arm joint angle
        cos_sentence = degrees(acos((sec_arm_length ** 2 - pri_arm_length ** 2 - d_hyp ** 2) / (-2 * d_hyp * pri_arm_length)))
        self.j1 = 90 - (cos_sentence - d_hyp_rot)

        # calculate secondary arm joint angle
        self.j2 = degrees(acos((d_hyp ** 2 - sec_arm_length ** 2 - pri_arm_length ** 2) / (-2 * sec_arm_length * pri_arm_length)))
        self.j2 -= degrees(atan(self.sec_arm_length[1][0] / self.sec_arm_length[2][0]))

        # get relative rotation matrix (rot)
        self.fk.set_joint_angles(180 - self.j0, -self.j1, -(90 - self.j2), 0)
        # Compute the rotation from joint 3 to the end-effector in the joint's frame
        joint3_rot = self.fk.get_joint_rotation_matrix(3)
        rel_rot = np.matmul(np.linalg.inv(joint3_rot), end_eff_mat)

        # extract XYZ Euler angles to align rel_rot to identity
        rel_rot_obj = Rotation.from_matrix(rel_rot)
        angles_xyz = rel_rot_obj.as_euler('yxz', degrees=True)

        # calculate the remaining joint angles
        raw_j5, raw_j4, raw_j3 = angles_xyz
        self.j4 = - raw_j4 if - 90 <= raw_j3 <= 90 else 180 + raw_j4
        self.j3 = (raw_j3 + 90) % 180 - 90
        self.j5 = ((- raw_j5) + 90) % 180 - 90

        return [self.j0, self.j1, self.j2, self.j3, self.j4, self.j5]

    @staticmethod
    def get_arm_length(arm):
        """
        This function takes in an arm array and calculates its length

        Args:
            arm: Arm array

        Return:
            float of the Arm length
        """
        return sqrt(sum([value[0] ** 2 for value in arm]))