import numpy as np

def clean_data(data):
    # data = data[data["測項"] != "RAINFALL"]
    # data = data[data["測項"] != "THC"]
    # data = data[data["測項"] != "WD_HR"]
    # data = data[data["測項"] != "WIND_DIREC"]

    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip("x*#A"))
    data = data.values
    data = np.delete(data, [0,1], 1)
    data[data == 'NR'] = 0
    data[data == ''] = 0
    data[data == 'nan'] = 0
    data = data.astype(np.float)
    return data
