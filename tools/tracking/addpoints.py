import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

import time

class PointCloudPublisher():
    header = Header()
    header.frame_id = 'rslidar'
    dtype = PointField.FLOAT32
    point_step = 16
    fields = [PointField(name='x', offset=0, datatype=dtype, count=1),
              PointField(name='y', offset=4, datatype=dtype, count=1),
              PointField(name='z', offset=8, datatype=dtype, count=1),
              PointField(name='intensity', offset=12, datatype=dtype, count=1)]

    def __init__(self):
        self.publisher_ = rospy.Publisher('/test_cloud', PointCloud2, queue_size=10)

    def drawPoints(self,fn):
        # self.header.stamp = self.get_clock().now().to_msg()
        points = np.load(fn)
        points = points[:,0:4].tobytes()
        rowstep = len(points)
        pc2_msg = PointCloud2(header=self.header,
                              height=1,
                              width=int(rowstep/16),
                              is_dense=False,
                              is_bigendian=False,
                              fields=self.fields,
                              point_step=16,
                              row_step=rowstep,
                              data=points)
        self.publisher_.publish(pc2_msg)

    def drawPointsFromnp(self,points):
        points = points[:,0:4].tobytes()
        rowstep = len(points)
        pc2_msg = PointCloud2(header=self.header,
                              height=1,
                              width=int(rowstep/16),
                              is_dense=False,
                              is_bigendian=False,
                              fields=self.fields,
                              point_step=16,
                              row_step=rowstep,
                              data=points)
        self.publisher_.publish(pc2_msg)
