import rospy
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy
import numpy as np
import time


class DrawBoundingBox:
    def __init__(self) -> None:
        #rospy.init_node('livox_detector', anonymous=True)
        self.marker_pub = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=1)
        self.marker_pub_lable = rospy.Publisher('/detect_box3d_lable', MarkerArray, queue_size=1)
        self.marker_array = MarkerArray()
        self.marker_array_lable = MarkerArray()

        self.lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                      [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        self.max_arr_marker = 0  # 计算总的mark的数量(最大数量)
        self.frame_id = 'rslidar'
        #self.frame_id = 'velodyne'
        self.lablestr = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def rotx(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                        [0,  c,  -s],
                        [0, s,  c]])
    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])
    def rotz(self,t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                        [s,  c,  0],
                        [0, 0,  1]])

    def get_3d_box(self, box):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:heading_angle
            box_size: tuple of (l,w,h)
            : rad scalar, clockwise from pos z axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        #Rx = self.rotx(-box[6])
        #Ry = self.roty(-box[7])
        Rx = self.rotx(0)
        Ry = self.roty(0)
        Rz = self.rotz(-box[8])
        R = np.dot(Rz,np.dot(Ry,Rx))
        l, w, h = box[3],box[4],box[5]

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + box[0]
        corners_3d[1, :] = corners_3d[1, :] + box[1]
        corners_3d[2, :] = corners_3d[2, :] + box[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def display(self, boxs,k):
        self.marker_array.markers.clear()
        self.marker_array_lable.markers.clear()
        boxes = []
        usebox = []
        for b in boxs:
            if b[10] < k:
                continue
            box = self.get_3d_box(b)
            box = box.transpose(1, 0).ravel()
            boxes.append(box)
            usebox.append(b)


        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i], ob[i+8], ob[i+16]))

            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.id = obid
            #print('marker.id',marker.id)
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.lifetime = rospy.Duration(0)

            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 1
            marker.color.a = 0.7
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.points = []
            for line in self.lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])
            self.marker_array.markers.append(marker)

            #lable
            m = Marker()
            m.header.frame_id = self.frame_id
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.ns = "test"
            m.id = obid
            m.pose.orientation.w = 1.0
            m.color.r = 1
            m.color.b = 0
            m.color.g = 1
            m.color.a = 1.0  # 标签的透明度
            m.scale.x = 2.0
            m.scale.y = 1
            m.scale.z = 1
            m.pose.position.x = float(usebox[obid][0])
            m.pose.position.y = float(usebox[obid][1])
            m.pose.position.z = float(usebox[obid][2]) + 2.0
            # m.text = self.lablestr[int(usebox[obid][9])] + str(round(usebox[obid][10],2)) + '---' + str(round(usebox[obid][7],2))# text类型必须是string
            m.text =  str(int(usebox[obid][11])) + '--' +self.lablestr[int(usebox[obid][9])]

            self.marker_array_lable.markers.append(m)

        if len(self.marker_array.markers) < self.max_arr_marker:
            for sign_mark_over in range(len(boxes), self.max_arr_marker):
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD

                marker.id = sign_mark_over
                marker.lifetime = rospy.Duration(0)
                marker.color.r = 0
                marker.color.g = 0
                marker.color.b = 0
                marker.color.a = 0.1
                marker.scale.x = 0.1
                marker.pose.orientation.x = 0
                marker.pose.orientation.w = 1
                marker.points.append(Point(0, 0, 0))
                marker.points.append(Point(0, 0, 0))
                self.marker_array.markers.append(marker)
                #lable
                m = Marker()
                m.header.frame_id = self.frame_id
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.ns = "test"
                m.id = sign_mark_over
                m.pose.orientation.w = 1.0
                m.color.r = 0
                m.color.b = 0
                m.color.g = 0
                m.color.a = 1
                m.scale.x = 0
                m.scale.y = 0
                m.scale.z = 0.1
                m.pose.position.x = 0
                m.pose.position.y = 0
                m.pose.position.z = 0
                m.text = str("0")
                self.marker_array_lable.markers.append(m)

        if len(self.marker_array.markers) >= self.max_arr_marker:
            self.max_arr_marker = len(self.marker_array.markers)

        if len(self.marker_array.markers) is not 0:
            self.marker_pub.publish(self.marker_array)
            self.marker_array.markers.clear()

        if len(self.marker_array_lable.markers) is not 0:
            self.marker_pub_lable.publish(self.marker_array_lable)
            self.marker_array_lable.markers.clear()

        # print(self.max_arr_marker)


if __name__ == '__main__':
    boxs = numpy.loadtxt('/home/xgp/catkin_xgp/rosBag/r/0.txt')
    #print(boxs)

 #   rospy.init_node("pointcloud_subscriber")
  #  while 1:
  #      bs = BIN_TENSORFLOW_TO_ROS(boxs=boxs)
  #      bs.prediction_publish()
  #      time.sleep(0.1)


    detector = DrawBoundingBox()
    # [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center


    for i in range(10):
        print(i)
        detector.display(boxs,0.5)
        time.sleep(2)

