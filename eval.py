from creare_track import create_track
from sort import Sort, associate_detections_to_trackers
import numpy as np
from fastapi_server import Metric
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    prev_frame_track_soft = None
    # sort = Sort(10,1)

    X = [] # amount
    Y = [] # range
    Z = [] # skip_chance
    acc = []

    for a in [5, 10, 15, 20]:
        for r in [0, 10, 25, 50]:
            for s in [.0, .05, .1, .25, .5]:
                metric = Metric()    
                sort = Sort(10, 1)
                country_balls_amount, track_data  = create_track(a, r, s)
                for el in track_data:
                    dets = [b['bounding_box'] for b in el['data'] if len(b['bounding_box']) > 0]
                    tracks = sort.update(np.array(dets))
            
                    for i in range(min(len(tracks),len(dets))):
                        el["data"][i]["track_id"] = tracks[i][-1]
                        # if len(el["data"][i]['bounding_box']) == 0:
                        #     el["data"][i]["bounding_box"] = list(tracks[i][:4])


                    metric.append(el)

                X.append(a)
                Y.append(r)
                Z.append(s)
                acc.append(metric.evaluate(False))
                metric = None
                sort = None

    
    data = {'Объектов': X,'Дрожание': Y, 'Пропуски': Z, 'Точность': acc}
    df = pd.DataFrame(data)
    # df.to_csv('metrics.csv')

    X,Y,Z,acc = map(np.array,(X,Y,Z,acc))
 
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    plot = ax.scatter(X, Y, Z, c=acc, s=50,  cmap=matplotlib.cm.coolwarm)

    fig.colorbar(plot, ax = ax,
        shrink = 1.0, aspect = 5)

    # ax.set_title('Surface plot')
    ax.set_xlabel('Объектов')
    ax.set_ylabel('Дрожание')
    ax.set_zlabel('Пропуски')
    plt.show()



