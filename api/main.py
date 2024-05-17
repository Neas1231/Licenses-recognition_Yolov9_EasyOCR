import os
import shutil

import cv2
import streamlit as st
from EasyOCR import easyocr
from glob import glob
from PIL import Image
from io import FileIO

if os.path.exists('./data'):
    shutil.rmtree('./data')
st.header('Приложение для распознавания автомобильных номеров')


def get_picture(path):
    import cv2
    img = cv2.imread(path)
    return img


def get_file_paths(directory):
    import os
    return [os.path.join(root, filename) for root, _, files in os.walk(directory) for filename in files if
            filename[-3:] in ['png', 'jpg']]

def write_file(file):
    file_path = './data/'+file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

def predicts(reader):
    import subprocess
    import shutil
    import os
    console_output = str(subprocess.run(
        'python ./yolov9/detect.py \
        --img 640 \
        --device 0 \
        --source ./data/  \
        --weights ./yolov9_weights/first_weights/best.pt \
        --save-crop \
        --project licenses/detect',
        capture_output=True))
    exp = console_output[console_output.rfind('exp'):console_output.rfind('exp') + console_output[
                                                                                   console_output.rfind(
                                                                                       'exp'):].find('\\')]
    print(exp)

    cropped = glob(f'./licenses/detect/{exp}/crops/license_plate/*')
    for file in cropped:
        img = get_picture(file)
        st.image(img, channels="BGR")
        result = reader.recognize(img, detail=0)
        st.write(result[0])
    if len(exp) > 3:
        exp_num = int(exp[3:])
        if exp_num > 10:
            shutil.rmtree(f'./licenses/detect/')


reader = easyocr.Reader(['en'], recog_network="licenses_model1", model_storage_directory="./EasyOCR/model",
                        user_network_directory="./EasyOCR/user_network", )
path = st.text_input('Введите путь к изображениям: ')
uploaded_file = st.file_uploader('Загрузите картинку')
if path:
    os.makedirs('./data/', exist_ok=True)
    print(path)
    img_paths = get_file_paths(path)
    st.write('Мы получили ', len(img_paths), ' картинок')
    for img_path in img_paths:
        img = get_picture(img_path)
        cv2.imwrite('./data/img.png', img)
        st.image(img, channels="BGR")
        st.write('В ходе распознавания мы получи такой результат:')
        predicts(reader)
    shutil.rmtree('./data/')
if uploaded_file is not None:
    os.makedirs('./data/', exist_ok=True)
    write_file(uploaded_file)
    try:
        st.image(uploaded_file, channels="BGR")
    except:
        try:
            st.video(uploaded_file)
        except:
            st.write('Что-то пошло не так')
    st.write('В ходе распознавания мы получи такой результат:')
    predicts(reader)
    shutil.rmtree('./data/')