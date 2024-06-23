import streamlit as st
import pandas as pd
import pickle
import requests
import os
from sklearn.preprocessing import StandardScaler

# 合并分割文件
def merge_files(part_dir, output_file):
    with open(output_file, 'wb') as output_f:
        for part_name in sorted(os.listdir(part_dir)):
            part_path = os.path.join(part_dir, part_name)
            with open(part_path, 'rb') as part_f:
                output_f.write(part_f.read())

# 加载模型
def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# 下载文件函数
def download_file_from_github(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"Failed to download file from {url}")

# 创建临时目录
temp_dir = "temp_parts"
os.makedirs(temp_dir, exist_ok=True)

# GitHub 文件链接列表
file_urls = [
    "https://raw.githubusercontent.com/handsome-hua/estate_streamlit/master/part_aa",
    "https://raw.githubusercontent.com/handsome-hua/estate_streamlit/master/part_ab",
    "https://raw.githubusercontent.com/handsome-hua/estate_streamlit/master/part_ac",
    "https://raw.githubusercontent.com/handsome-hua/estate_streamlit/master/part_ad"
]

# 下载分割文件
for url in file_urls:
    file_name = url.split('/')[-1]
    file_path = os.path.join(temp_dir, file_name)
    download_file_from_github(url, file_path)
st.success("All parts downloaded successfully.")

# 合并并加载模型
merged_file = "random_forest_model_cut.pkl"

if st.button("Merge and Load Model"):
    merge_files(temp_dir, merged_file)
    try:
        global reg  # 确保 reg 变量全局可用
        reg = load_model(merged_file)
        st.success("Model merged and loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")

# 以下部分保持不变，实现你的预测功能
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
    '复式': 0,
    '平层': 1,
    '跃层': 2,
    '错层': 3,
}

df = pd.read_csv("data_last.csv")

filled_processed_data = df.drop(columns=['elevator', 'framework', 'saloon', 'house_term', 'ownership', 'deed', 'facility0', 'facility5', 'facility2', 'facility4', 'kitchen', 'facility1', 'facility3', 'rights'])
columns_to_standardize = ['coordinate_x', 'coordinate_y', 'scale']
scaler = StandardScaler()
filled_processed_data[columns_to_standardize] = scaler.fit_transform(filled_processed_data[columns_to_standardize])

def predict_price(input_data):
    global reg  # 确保函数内使用全局变量 reg
    input_df = pd.DataFrame([input_data])
    
    # 标准化输入数据
    input_df[columns_to_standardize] = scaler.transform(input_df[columns_to_standardize])
    
    # 检查输入数据是否有缺失值
    if input_df.isnull().any().any():
        st.error("Input data contains missing values.")
        return None
    
    st.write(input_df)
    
    # 预测
    try:
        prediction = reg.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def run_random_forest_app():
    st.subheader('Please enter the required details :')

    area = st.number_input("Enter Total Area", 10)
    coordinate_x = st.slider("Select Longitude", float(df['coordinate_x'].min())-1, float(df['coordinate_x'].max())+1)
    coordinate_y = st.slider("Select Latitude", float(df['coordinate_y'].min())-1, float(df['coordinate_y'].max())+1)
    apt = st.slider("Select Apartment", 1, int(max(df['apt'])) + 1)
    lift = st.slider("select lift", 1, int(max(df['lift'])) + 1)
    room = st.number_input("Enter the number of rooms", 1, 10)
    bath = st.number_input("Enter the number of baths", 1, 5)

    st.sidebar.write('----')
    Location = st.sidebar.selectbox('Select the Location', list(district_codes.keys()))
    location_label = district_codes[Location]
    level_name = st.sidebar.selectbox("Select the floor", list(FLOOR_LEVEL.keys()))
    level = FLOOR_LEVEL[level_name]
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
        if result is not None:
            formatted_result = "{:,.1f}".format(result)
            st.success('Total Price: {} '.format(formatted_result))

if __name__ == '__main__':
    run_random_forest_app()
