import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import os
import pandas as pd
import chardet

workdir = os.path.dirname(os.path.abspath(__file__))

mainland_fips = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', '17', '18', '19', '20', '21', '22',
                 '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                 '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56']


def getunicode(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
        print(result['encoding'])


def basicinfo(filename):
    # 打开TIFF文件
    with rasterio.open(filename) as src:
        # 获取波段数
        num_bands = src.count
        print(f"Number of bands: {num_bands}")

        # 获取数据类型
        data_type = src.dtypes[0]
        print(f"Data type: {data_type}")

        # 获取分辨率
        resolution = src.res
        print(f"Resolution: {resolution}")

        # 获取空间参考系统
        crs = src.crs
        print(f"Coordinate reference system: {crs}")

        # 获取大小
        width = src.width
        height = src.height
        print(f"Width: {width}, Height: {height}")


def count(image):
    # 读取数据并展平为一维数组
    data = image.flatten()

    # 计算每个值的count和percentage
    unique, counts = np.unique(data, return_counts=True)
    percentages = counts / len(data) * 100

    # 打印结果
    for value, count, percentage in zip(unique, counts, percentages):
        print("Value: {}, Count: {}, Percentage: {:.2f}%".format(value, count, percentage))


def hist(filename):
    with rasterio.open(filename) as src:
        data = src.read(1)

        # 定义掩膜
        mask_arr = (data != src.nodata).astype('uint8')

        # 计算统计信息
        stats = []
        with rasterio.Env():
            for i in range(1, src.count + 1):
                out_stats = rasterio.mask.mask(src, mask_arr, i, nodata=src.nodata, filled=True, all_touched=True,
                                               crop=True, indexes=i, stats='min max mean std median')
                stats.append(out_stats[0])

        # 打印统计信息
        print(stats)


def getstate(jsonfiles, state):
    jsonfile = jsonfiles + r"\cb_2018_" + state + "_tract_500k.json"

    return jsonfile


def normid(geoid):
    """insert 000 in middle of geoid"""
    geoid = geoid[:5] + "00" + geoid[5:]
    return geoid


def itercsv(rasterfile, jsonfiles, csvfile):
    # use a dataframe to store the result
    df = pd.DataFrame(columns=['GEOID', 'NormID', 'State', 'MeanValue'])
    now_state = -1
    count = 0

    tracts = pd.read_csv(csvfile, index_col=None, encoding='ISO-8859-1', dtype={'geoid': str})

    files = os.listdir(jsonfiles)

    with rasterio.open(rasterfile) as src:
        dst_crs = src.crs

        for jsonfile in files:
            state = jsonfile.split('_')[2]
            polygons = gpd.read_file(jsonfiles + '\\' + jsonfile)
            polygons_4326 = polygons.to_crs(epsg=4326)
            polygons_reproj = polygons_4326.copy()
            polygons_reproj.geometry = polygons_4326.geometry.to_crs(dst_crs)

            for val in polygons_reproj.values[:, 3]:
                target_polygon = polygons_reproj[polygons_reproj.values[:, 3] == val]
                target_shape = [mapping(target_polygon.geometry.iloc[0])]
                try:
                    out_image, out_transform = mask(dataset=src, shapes=target_shape, crop=True, nodata=-1)
                    out_image = np.squeeze(out_image)
                    out_image = np.moveaxis(out_image, 0, -1)
                    meanval = np.mean(out_image[out_image != 255])

                except:
                    print("not overlap:" + val + "---" + str(count))
                    count += 1
                    continue

                geoid = val
                normid = '14000US' + geoid.split('US')[1]
                df = pd.concat([df, pd.DataFrame([[geoid, normid, state, meanval]],
                                                 columns=['GEOID', 'NormID', 'State', 'MeanValue'])], ignore_index=True)

    # with rasterio.open(rasterfile) as src:
    #     dst_crs = src.crs
    #
    #     for index, row in tracts.iterrows():
    #         if count:
    #             count = False
    #             continue
    #         geoid = normid(row['geoid'])
    #         state = row['State']
    #
    #         # new state then load jsonfile of state
    #         if now_state != state:
    #             if now_state != -1:
    #             jsonfile = getstate(jsonfiles, state)
    #             try:
    #                 polygons = gpd.read_file(jsonfile)
    #             except:
    #                 print("cant find jsonfile of state " + state)
    #                 continue
    #             now_state = state
    #             polygons_4326 = polygons.to_crs(epsg=4326)
    #             polygons_reproj = polygons_4326.copy()
    #             polygons_reproj.geometry = polygons_4326.geometry.to_crs(dst_crs)
    #
    #         # get the polygon of geoid
    #         try:
    #             target_polygon = polygons_reproj[polygons_reproj.values[:, 3] == geoid]
    #             target_shape = [mapping(target_polygon.geometry.iloc[0])]
    #         except:
    #             print("cant find polygon of geoid " + geoid)
    #             df = pd.concat([df, pd.DataFrame([[geoid, state, np.nan]], columns=['GEOID', 'State', 'MeanValue'])],
    #                            ignore_index=True)
    #             continue
    #
    #         out_image, out_transform = mask(dataset=src, shapes=target_shape, crop=True, nodata=-1)
    #         out_image = np.squeeze(out_image)
    #         out_image = np.moveaxis(out_image, 0, -1)
    #         meanval = np.mean(out_image[out_image != 255])
    #
    #         df = pd.concat([df, pd.DataFrame([[geoid, state, meanval]], columns=['GEOID', 'State', 'MeanValue'])], ignore_index=True)

    return df


def maskbyjson(jsonfile, rasterfile):
    # json->gdf
    polygons = gpd.read_file(jsonfile)
    polygons_4326 = polygons.to_crs(epsg=4326)
    polygons_reproj = polygons_4326.copy()

    # reprojection
    with rasterio.open(rasterfile) as src:
        dst_crs = src.crs
        polygons_reproj.geometry = polygons_4326.geometry.to_crs(dst_crs)

        # 筛选geoid为特定值的polygon
        target_geoid = "1400000US01001020100"
        target_polygon = polygons_reproj[polygons_reproj.values[:, 3] == target_geoid]

        # 将目标polygon转换为shape对象
        target_shape = [mapping(target_polygon.geometry.iloc[0])]

        # shapes = [mapping(geom) for geom in polygons_reproj.geometry]

        out_image, out_transform = mask(dataset=src, shapes=target_shape, crop=True, nodata=-1)
        out_image = np.squeeze(out_image)
        out_image = np.moveaxis(out_image, 0, -1)
        meanval = np.mean(out_image[out_image != 255])

        # # 可视化图像
        # plt.imshow(out_image)
        # plt.show()


def maskbyshp(shpfile, filename, out_file):
    with rasterio.open(filename) as src:
        dst_crs = src.crs
        transform = src.transform
        nodata = src.nodata

        # gdf
        polygons = gpd.read_file(shpfile)
        polygons_4326 = polygons.to_crs(epsg=4326)
        polygons_reproj = polygons_4326.copy()
        polygons_reproj.geometry = polygons_4326.geometry.to_crs(dst_crs)

        shapes = [mapping(geom) for geom in polygons_reproj.geometry]

        # 裁剪栅格影像
        out_image, out_transform = mask(dataset=src, shapes=shapes, crop=True)

        out_image = np.squeeze(out_image)
        out_image = np.moveaxis(out_image, 0, -1)
        # 可视化图像
        plt.imshow(out_image)
        plt.show()

        return out_image


def crop_raster_get_mean(rasterfile):
    """
    get mean value of raster data in each polygon(tract) of shpfiles
    :param rasterfile: directory of rasterfile
    :return: result dataframe
    """
    df = pd.DataFrame(columns=['GEOID', 'NormID', 'State', 'MeanValue'])
    with rasterio.open(rasterfile) as src:
        dst_crs = src.crs

        for state_fips in mainland_fips:

            # get shapefile and reproject
            shpfile = workdir + "\\shapefiles\\" + f'tl_2021_{state_fips}_tract.shp'
            polygons = gpd.read_file(shpfile)
            polygons_reproj = polygons.copy()
            polygons_reproj.geometry = polygons.geometry.to_crs(dst_crs)

            print(f"working on state {state_fips}")

            # state level crop
            # shapes = [mapping(geom) for geom in polygons_reproj.geometry]

            # polygon level crop
            for geoid in polygons_reproj.values[:, 3]:
                target_polygon = polygons_reproj[polygons_reproj.values[:, 3] == geoid]
                target_shape = [mapping(target_polygon.geometry.iloc[0])]

                try:
                    out_image, out_transform = mask(dataset=src, shapes=target_shape, crop=True, nodata=-1)
                except:
                    print("no overlap " + geoid)
                    df = pd.concat([df, pd.DataFrame([[geoid, "14000US" + geoid, state_fips, meanval]],
                                                     columns=['GEOID', 'NormID', 'State', 'MeanValue'])],
                                   ignore_index=True)
                    continue

                # get mean value and save to dataframe
                out_image = np.squeeze(out_image)
                out_image = np.moveaxis(out_image, 0, -1)
                meanval = np.mean(out_image[out_image != 255])
                df = pd.concat([df, pd.DataFrame([[geoid, "14000US" + geoid, state_fips, meanval]],
                                                 columns=['GEOID', 'NormID', 'State', 'MeanValue'])], ignore_index=True)
        return df


def main():
    filename = "nlcd_tcc_conus_2021_v2021-4.tif"
    rasterfile = os.path.join(workdir, filename)

    df = crop_raster_get_mean(rasterfile)
    df.to_pickle(workdir + r"\tree_cover_tract_data.pkl")
    # df.to_csv(work_dir + r"\tree_cover_tract_data.csv", index=False)

if __name__ == '__main__':
    main()
