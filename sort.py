import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
import lap

def circum_radius(bbox):
    w = abs(bbox[2]-bbox[0])
    h = abs(bbox[3]-bbox[1])
    diameter = np.square(np.power(w,2) + np.power(h,2))
    return diameter / 2

def bbox_center(bb):
    return (bb[2]- bb[1], bb[3]-bb[0])
    
def edist(bb_1, bb_2):
    """
    Computes euclidean distance between two bboxes in the form [x1,y1,x2,y2]
    """
    return euclidean(bbox_center(bb_1), bbox_center(bb_2))

def iou(bb_1, bb_2):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_1[0], bb_2[0])
    yy1 = np.maximum(bb_1[1], bb_2[1])
    xx2 = np.minimum(bb_1[2], bb_2[2])
    yy2 = np.minimum(bb_1[3], bb_2[3])
    bb_1_area = (bb_1[2] - bb_1[0]) * (bb_1[3] - bb_1[1])
    bb_2_area = (bb_2[2] - bb_2[0]) * (bb_2[3] - bb_2[1]) 
    intersection = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)

    union = bb_1_area + bb_2_area - intersection

    return intersection / union

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0], 
                          [0,0,1,0,0,0,1],  
                          [0,0,0,1,0,0,0], 
                          [0,0,0,0,1,0,0],  
                          [0,0,0,0,0,1,0], 
                          [0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0], 
                          [0,0,1,0,0,0,0], 
                          [0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox,observed=True):
    """
    Updates the state vector with observed bbox.
    """
    if observed:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers, mode='iou', threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  mode_dct = {'iou': iou, 'edist': edist}
  metric = mode_dct[mode]

  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = metric(det,trk)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      _, x, y = lap.lapjv(-iou_matrix, extend_cost=True)
      matched_indices = np.array([[y[i],i] for i in x if i >= 0])
      # x, y = linear_sum_assignment(-iou_matrix)
      # matched_indices = np.array(list(zip(x, y)))
      
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=10, min_hits=2):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []

  def clear(self):
    self.trackers = []

  def update(self, dets=np.empty((0, 4))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 4))
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3]]

    matched, unmatched_dets, unmatched_trks = \
        associate_detections_to_trackers(dets, trks, threshold = 0.1)

    # update matched trackers with assigned detections
    # 
    for m in matched:
      self.trackers[m[1]].update(dets[m[0]])
    # for u in unmatched_trks:
    #   self.trackers[u].update(trks[u], False)

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    # 


    for trk in self.trackers:
        d = trk.get_state()[0]
        if (trk.hit_streak >= self.min_hits) and (trk.time_since_update <= self.max_age):
          ret.append(np.concatenate((d,[trk.id])).reshape(1,-1))
    # remove dead tracklet
    for k, trk in enumerate(self.trackers): 
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(k)
          # unmatched_trks = unmatched_trks[unmatched_trks != k]



    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
