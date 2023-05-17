import pandas as pd
import json


def json2csv(labelmap, json_dir, csv_dir=None):
    if csv_dir is None:
        csv_dir = json_dir.replace(".json", ".csv")
    data = json.load(open(json_dir))
    csv_list = []
    for classification in data:
        width, height = classification['width'], classification['height']
        image = classification['image']
        for item in classification['tags']:
            name = item['name']
            id_name = labelmap[name]
            xmin = int(item['pos']['x'])
            ymin = int(item['pos']['y'])
            # Hacemos esto porque el formato PASCAL VOC NO utiliza x, w,
            # por el contrario usa xmin y xmax siendo xmax = xmin+w
            xmax = int(item['pos']['x'] + item['pos']['w'])
            # Caso similar con ymax, es y + h
            ymax = int(item['pos']['y'] + item['pos']['h'])
            value = (image, width, height, xmin, ymin, xmax, ymax, name, id_name)
            csv_list.append(value)

    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'class_id']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    print(csv_df)
    csv_df.to_csv(csv_dir, index=False)


if __name__ == '__main__':
    root = "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean"
    paths = {"train": f"{root}/train.json", "test": f"{root}/test.json"}
    lb_map = {"carro": 1, "motos": 2}
    # Converting train.json 2 train.csv
    json2csv(lb_map, paths["train"])
    # Converting test.json 2 test.csv
    json2csv(lb_map, paths["test"])
