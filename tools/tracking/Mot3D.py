from tracking.tracker import Tracker3D
from tracking.config import cfg, cfg_from_yaml_file
import numpy as np
import time


class Mot3D:
    def __init__(self,yaml_file):
        self.yaml_file = yaml_file
        self.config = cfg_from_yaml_file(yaml_file, cfg)
        self.tracker = Tracker3D(box_type="Kitti", tracking_features=False, config=self.config)
        self.i = 1


    def track_one_frame(self,object_info):
        all_info = np.array([x for x in object_info if float(x[10]) > self.config.init_score])
        if len(all_info) < 1:
            all_info = np.zeros(11).reshape((1, -1))
        objects = all_info[:, [0, 1, 2, 3, 4, 5, 8]]
        det_features = all_info[:, [6, 7, 9]]
        det_scores = all_info[:, [10]]

        self.tracker.tracking(objects,
                              features=det_features,
                              scores=det_scores,
                              timestamp=self.i)
        self.i += 1

        savs = np.zeros(12).reshape((1, -1))
        for key in self.tracker.active_trajectories.keys():
            track = self.tracker.active_trajectories[key]
            track.filtering(self.config)
            id = np.array(key)
            kk = list(track.trajectory.keys())
            box = track.trajectory[kk[-1]].predicted_state
            box = np.array([box[0, 0], box[1, 0], box[2, 0], box[9, 0], box[10, 0], box[11, 0], box[12, 0]])
            box = box.reshape((1, -1))
            for j in range(len(kk)):
                index = -1 - j
                aa = track.trajectory[kk[index]]
                if aa.score is not None:
                    features = aa.features.reshape((1, -1))
                    scores = aa.score.reshape((1, -1))
                    sav = np.concatenate((box[:, [0, 1, 2, 3, 4, 5]], features[:, [0, 1]], box[:, 6].reshape((-1, 1)),
                                          features[:, 2].reshape((-1, 1)), scores, id.reshape((-1, 1))), axis=1)
                    savs = np.r_[savs, sav]
                    break

        return savs[1:len(savs),:]
