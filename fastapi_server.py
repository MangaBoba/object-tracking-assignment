from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np
from sort import Sort, associate_detections_to_trackers
from collections import Counter
from motpy import Detection, MultiObjectTracker


app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

class Metric:
    def __init__(self):
        self.frame_det_trk_ids = []
    
    def append(self, frame): 
        for el in frame['data']:
            if el['bounding_box']:
                self.frame_det_trk_ids.append((el['cb_id'], el['track_id']))

    def clear(self):
        self.frame_det_trk_ids = []

    def evaluate(self, disp = True):
        
        cb_ids = sorted(list(set([val[0] for val in self.frame_det_trk_ids])))
        real_ids = [None]*len(cb_ids)
        true_preds = [0]*len(cb_ids)
        total_preds = [0]*len(cb_ids)
        acc_per_cb = [0]*len(cb_ids)
        for k, cb_id in enumerate(cb_ids):
            preds = [p[1] for p in self.frame_det_trk_ids if p[0]==cb_id]
            mc = Counter(preds).most_common(2)
            if mc[0][0] is not None:
                real_ids[k] = mc[0][0]
            elif len(mc) == 2:
                real_ids[k] = mc[1][0]
            else:
                real_ids[-1]
            # real_ids[k] = mc[0][0] if mc[0][0] else mc[1][0]
            total_preds[k] = len(preds)
            true_preds[k] = sum([1 for p in preds if p == real_ids[k]])
            acc_per_cb[k] = true_preds[k] / total_preds[k]
            if disp:
                print(f'Cb_id {cb_id} real_id {real_ids[k]} acc: {acc_per_cb[k]:.2f}')
        
        overall_acc = (sum(acc_per_cb)/len(acc_per_cb))
        if disp:
            print(f'Overall acc: {overall_acc:.2f}')
        return overall_acc
        
def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """

    dets = [b['bounding_box'] for b in el['data'] if len(b['bounding_box']) > 0]
    tracks = sort.update(np.array(dets))

    for i in range(min(len(tracks),len(dets))):
        el["data"][i]["track_id"] = tracks[i][-1]
        if len(el["data"][i]['bounding_box']) == 0:
            el["data"][i]["bounding_box"] = list(tracks[i][:4])


    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    return el

metric = Metric()
sort = Sort(10, 1)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))

    for el in track_data:
        await asyncio.sleep(0)

        el = tracker_soft(el)
        metric.append(el)

        # TODO: part 2
        # tracker.step(detections=[Detection(box=bb) for bb in dets])
        # tracks = tracker.active_tracks()
        # print(tracks)
        # exit(0)
        # el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    metric.evaluate()
    print('Bye..')
