import streamlit as st
import zipfile
from io import BytesIO
from PIL import Image
import pytesseract
import re
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from math import radians, sin, cos, sqrt, atan2
from streamlit_folium import st_folium
from PIL.ExifTags import TAGS, GPSTAGS

st.title("Извлечение координат из изображений и отображение на карте")

# Шаг 1: Выбор приложения
option = st.selectbox(
    "Выберите приложение из списка:",
    ("NoteCam @ iOS", "GPS MAP CAMERA", "Бесплатная версия GPS Камера 55")
)

# Шаг 2: Загрузка архива изображений
uploaded_file = st.file_uploader("Загрузите архив изображений (ZIP-файл)", type="zip")

def get_exif_data(img):
    exif_data = {}
    info = img._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'GPSInfo':
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data['GPSInfo'] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def get_lat_lon(exif_data):
    lat = None
    lon = None
    gps_info = exif_data.get('GPSInfo')
    if gps_info:
        gps_latitude = gps_info.get('GPSLatitude')
        gps_latitude_ref = gps_info.get('GPSLatitudeRef')
        gps_longitude = gps_info.get('GPSLongitude')
        gps_longitude_ref = gps_info.get('GPSLongitudeRef')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            if gps_latitude_ref != 'N':
                lat = -lat

            lon = convert_to_degrees(gps_longitude)
            if gps_longitude_ref != 'E':
                lon = -lon

    return lat, lon

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def extract_lat_lon(text):
    lat = None
    lon = None

    # Регулярные выражения для поиска координат
    lat_pattern = r'Latitude[:\s]*([-+]?\d{1,3}\.\d+)'
    lon_pattern = r'Longitude[:\s]*([-+]?\d{1,3}\.\d+)'

    lat_match = re.search(lat_pattern, text, re.IGNORECASE)
    lon_match = re.search(lon_pattern, text, re.IGNORECASE)

    if lat_match:
        lat = float(lat_match.group(1))
    if lon_match:
        lon = float(lon_match.group(1))

    return lat, lon

def calculate_distances(df):
    distances = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            distances.append({
                'Точка 1': i+1,
                'Точка 2': j+1,
                'Расстояние (м)': round(distance, 2)
            })
    return pd.DataFrame(distances)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Радиус Земли в метрах
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi/2)**2 + cos(phi1)*cos(phi2)*sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    meters = R * c
    return meters

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        image_names = [name for name in zip_ref.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Найдено {len(image_names)} изображений в архиве.")

        coords_list = []

        for name in image_names:
            try:
                img_data = zip_ref.read(name)
                img = Image.open(BytesIO(img_data))

                # Попытка извлечь координаты из метаданных EXIF
                exif_data = get_exif_data(img)
                lat, lon = get_lat_lon(exif_data)

                # Если не удалось, используем OCR для извлечения координат
                if lat is None or lon is None:
                    text = pytesseract.image_to_string(img)
                    lat, lon = extract_lat_lon(text)

                if lat is not None and lon is not None:
                    coords_list.append({'latitude': lat, 'longitude': lon})
                else:
                    st.write(f"Не удалось извлечь координаты из изображения {name}")
            except Exception as e:
                st.write(f"Ошибка при обработке изображения {name}: {e}")

        if coords_list:
            df = pd.DataFrame(coords_list)
            st.write("Извлеченные координаты:")
            st.dataframe(df)

            # Отображение точек на карте
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
            marker_cluster = MarkerCluster().add_to(m)

            for idx, row in df.iterrows():
                folium.Marker(location=[row['latitude'], row['longitude']],
                              popup=f"Точка {idx+1}").add_to(marker_cluster)

            # Расчет и отображение расстояний между точками
            distances_df = calculate_distances(df)
            st.write("Расстояния между точками (в метрах):")
            st.dataframe(distances_df)

            st.write("Карта с отмеченными точками:")
            st_folium(m, width=700, height=500)
        else:
            st.write("Не удалось извлечь координаты из загруженных изображений.")
