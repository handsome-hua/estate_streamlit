import pandas as pd
import pickle
import requests
from sklearn.preprocessing import StandardScaler
import streamlit as st
'''
                 Feature  Importance
18                 scale    0.666622
1           coordinate_y    0.117623
24        district_label    0.083288
0           coordinate_x    0.071390
15                   apt    0.009470
4               elevator    0.007750
2   decoration_condition    0.005613
20                  bath    0.005476
19             structure    0.005265
16                  lift    0.004764
22                  room    0.004192
11                 level    0.003038
12             framework    0.002686
23                saloon    0.002231
13            house_term    0.002177
14             ownership    0.001611
3                   deed    0.001140
5              facility0    0.001072
10             facility5    0.000931
7              facility2    0.000902
9              facility4    0.000795
21               kitchen    0.000774
6              facility1    0.000513
8              facility3    0.000347
17                rights    0.000334

'''
'''
我使用了
18                 scale    0.666622
1           coordinate_y    0.117623
24        district_label    0.083288
0           coordinate_x    0.071390
15                   apt    0.009470
2   decoration_condition    0.005613
20                  bath    0.005476
19             structure    0.005265
16                  lift    0.004764
22                  room    0.004192
11                 level    0.003038
'''
df = pd.read_csv("data_last.csv")
def download_file_from_google_drive(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
    else:
        st.error("Failed to download file")

def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model
  
# Google Drive 文件链接
url = "https://drive.google.com/uc?export=download&id=1SJ4pl2OYKwYoq-cuFk13Ho_jBLBPPOb6"
destination = 'random_forest_model_cut.pkl'

# 下载文件
st.write("Downloading file...")
download_file_from_google_drive(url, destination)
st.write("Download completed. File saved as random_forest_model_cut.pkl")

# 加载模型
try:
    reg = load_model(destination)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
district_codes = {
    '黄埔': 0,
    '徐汇': 1,
    '长宁': 2,
    '静安': 3,
    '普陀': 4,
    '虹口': 5,
    '杨浦': 6,
    '闵行': 7,
    '宝山': 8,
    '嘉定': 9,
    '浦东': 10,
    '金山': 11,
    '松江': 12,
    '青浦': 13,
    '奉贤': 14,
    '崇明': 15
}
FLOOR_LEVEL = {
    '低': 0,
    '中': 1,
    '高': 2,
    '地下室': 4,
}
CONDITION_CODE = {
    '毛坯': 0,
    '简装': 1,
    '精装': 2,
}
STRUCTURE_CODE = {
                '复式' : 0,
                '平层' : 1,
                '跃层' : 2,
                '错层' : 3,
            }

filled_processed_data = pd.read_csv('data_last.csv')

filled_processed_data = filled_processed_data.drop(columns=['elevator','framework','saloon','house_term','ownership','deed','facility0','facility5','facility2','facility4','kitchen','facility1','facility3','rights'])
columns_to_standardize = ['coordinate_x', 'coordinate_y', 'scale']
scaler = StandardScaler()
filled_processed_data[columns_to_standardize] = scaler.fit_transform(filled_processed_data[columns_to_standardize])

def predict_price(input_data):
    # 创建输入数据框
    input_df = pd.DataFrame([input_data])
    # 标准化
    input_df[columns_to_standardize] = scaler.transform(input_df[columns_to_standardize])
    st.write(input_df)
    # 预测
    prediction = reg.predict(input_df)

    return prediction[0]

# district_labels = pd.DataFrame(list(district_codes.items()), columns=['Division', 'Code'])
def run_random_forest_app():
    st.subheader('Please enter the required details :')

    # with st.expander("District"):
    #     st.dataframe(district_labels)
    area= st.number_input("Enter Total Area", 10)

    coordinate_x = st.slider("Select Longitude", float(df['coordinate_x'].min())-1, float(df['coordinate_x'].max())+1)
    coordinate_y = st.slider("Select Latitude", float(df['coordinate_y'].min())-1, float(df['coordinate_y'].max())+1)
    apt = st.slider("Select  Apartment", 1, int(max(df['apt'])) +1)
    lift = st.slider("select lift", 1, int(max(df['lift'])) +1)
    room = st.number_input("Enter the number of rooms", 1, 10)
    bath = st.number_input("Enter the number of bath", 1, 5)

    st.sidebar.write('----')
    Location = st.sidebar.selectbox('Select the Location', list(district_codes.keys()))
    location_label = district_codes[Location]  # 使用标签编码后的值 location 映射到 标签
    level_name = st.sidebar.selectbox("Select the floor", list(FLOOR_LEVEL.keys()))
    level = FLOOR_LEVEL[level_name]  # 楼层编码
    decoration = st.sidebar.selectbox('Select the decoration', list(CONDITION_CODE.keys()))
    decoration_label = CONDITION_CODE[decoration]
    structure = st.sidebar.selectbox('Select the structure', list(STRUCTURE_CODE.keys()))
    structure_label = STRUCTURE_CODE[structure]

    input_data = {
        'coordinate_x': coordinate_x,
        'coordinate_y': coordinate_y,
        'decoration_condition': decoration_label,
        'level': level,
        'apt': apt,
        'lift': lift,
        'scale': area,
        'structure': structure_label,
        'bath': bath,
        'room': room,
        'district_label': location_label,
    }
    if st.button("Calculate Price"):
        result = predict_price(input_data)
        formatted_result = "{:,.1f}".format(result)  # 保留一位小数，并使用千位分隔符
        st.success('Total Price: {} '.format(formatted_result))
if __name__ == '__main__':
    run_random_forest_app()
