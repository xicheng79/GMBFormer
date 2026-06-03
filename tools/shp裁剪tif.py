import os
import fiona.path  # noqa: F401 - ensure geopandas can access fiona.path
import rasterio
import geopandas as gpd
from rasterio.mask import mask


# ============================================================
# 1. 输入输出路径
# ============================================================

# 输入 tif，例如 ESA WorldCover、绿地分布 tif、遥感影像 tif
INPUT_TIF = r"C:\Users\admin\Downloads\Chengdu_Green_WithCropland_ESA_WorldCover_2021_10m.tif"

# 裁剪边界 shp，例如成都市边界
CLIP_SHP = r"G:\DATA\绿地论文\研究区域图\potsdam_brandenburg_boundary_generator_no_fiona_v4\potsdam_boundary_output_v4\shp\04_potsdam_boundary\04_potsdam_boundary.shp"

# 输出裁剪后的 tif
OUTPUT_TIF = r"G:\DATA\绿地论文\研究区域图\Potsdam_green_binary_0_1_2021_cut.tif"


# ============================================================
# 2. 根据 shp 裁剪 tif
# ============================================================

def clip_tif_by_shp(input_tif, clip_shp, output_tif):
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    print("正在读取矢量边界...")
    shp_gdf = gpd.read_file(clip_shp)

    if shp_gdf.empty:
        raise ValueError("裁剪 shp 为空，请检查 shp 文件。")

    print("正在读取栅格影像...")
    with rasterio.open(input_tif) as src:
        raster_crs = src.crs
        raster_meta = src.meta.copy()

        print("TIF CRS:", raster_crs)
        print("SHP CRS:", shp_gdf.crs)

        # 如果 shp 没有 CRS，直接报错，避免裁剪错位
        if shp_gdf.crs is None:
            raise ValueError(
                "你的 shp 没有坐标系信息，请先在 ArcGIS/QGIS 中定义投影，"
                "或者确认它的真实 CRS 后再运行。"
            )

        # 如果 shp 和 tif 坐标系不同，自动投影到 tif 坐标系
        if shp_gdf.crs != raster_crs:
            print("SHP 与 TIF 坐标系不一致，正在自动重投影 shp...")
            shp_gdf = shp_gdf.to_crs(raster_crs)

        # 去除空几何
        shp_gdf = shp_gdf[~shp_gdf.geometry.is_empty]
        shp_gdf = shp_gdf[shp_gdf.geometry.notnull()]

        if shp_gdf.empty:
            raise ValueError("重投影后 shp 几何为空，请检查 shp 数据。")

        # 获取几何
        geometries = [geom for geom in shp_gdf.geometry]

        print("正在裁剪 TIF...")
        out_image, out_transform = mask(
            dataset=src,
            shapes=geometries,
            crop=True,
            nodata=src.nodata
        )

        # 更新元数据
        raster_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": raster_crs,
            "compress": "lzw"
        })

        print("正在保存裁剪结果...")
        with rasterio.open(output_tif, "w", **raster_meta) as dst:
            dst.write(out_image)

    print("裁剪完成！")
    print("输出文件：", output_tif)


# ============================================================
# 3. 主程序
# ============================================================

if __name__ == "__main__":
    clip_tif_by_shp(INPUT_TIF, CLIP_SHP, OUTPUT_TIF)
