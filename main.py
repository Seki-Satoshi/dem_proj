from dem_concatinator import concat_tiles
from dem_convert import convert_tiles, init_db
from dem_enabler import select_upstream
from dem_utils import DEM_DIR, dem_path_generator, offset
from dem_visualize import visualize_db


def main():
    print("Hello from dem-proj!")
    TILE_XML = dem_path_generator(DEM_DIR)
    OFFSETS = offset(DEM_DIR)
    DB_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db.json"
    OUTPUTS = "/Users/fukaya/Documents/pythons/dem_proj/output"
    OUTPUT_DIR = f"{OUTPUTS}/tiles"
    DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/dem.npy"
    OBJ_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/objection.npy"
    VALID_DATA = "/Users/fukaya/Documents/pythons/dem_proj/output/valid.npy"

    if False:
        print("init db")
        init_db(TILE_XML, OFFSETS, DB_PATH)
        visualize_db(DB_PATH, f"{OUTPUTS}/init.png")
        # exit(0)
        print("converting to npy")
        convert_tiles(DB_PATH, OUTPUT_DIR)
        visualize_db(DB_PATH, f"{OUTPUTS}/converted.png")

        print("concatinating all npy to one file")
        concat_tiles(DB_PATH, DEM_PATH)
    
    select_upstream(DEM_PATH, OBJ_PATH, VALID_DATA)

    print("done")


if __name__ == "__main__":
    main()
