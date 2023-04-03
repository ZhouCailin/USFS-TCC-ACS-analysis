import os
import requests
import os
from zipfile import ZipFile

# 大陆州的FIPS代码列表
mainland_fips = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', '17', '18', '19', '20', '21', '22',
                 '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                 '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56']

work_dir = os.path.dirname(os.path.abspath(__file__))
# 下载链接
url_prefix = 'https://www2.census.gov/geo/tiger/TIGER2021/TRACT/'
# 存储文件夹
data_dir = work_dir + r"\shapefiles"
# 新建文件夹
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

for state_fips in mainland_fips:
    # 文件名
    file_name = f'tl_2021_{state_fips}_tract.zip'
    # 下载链接
    file_url = url_prefix + file_name
    # 本地保存路径
    save_path = os.path.join(data_dir, file_name)

    # 发送请求并下载文件
    response = requests.get(file_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

    # 解压文件
    with ZipFile(save_path, 'r') as zipObj:
        # 提取所有文件
        zipObj.extractall(data_dir)
