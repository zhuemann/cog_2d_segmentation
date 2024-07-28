import pandas as pd
from data_prepocessing.get_max_pixel import get_max_pixel_step3

def external_get_max_pixel():

    df = pd.read_excel('/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/swedish_dataframe_test.xlsx')
    df = get_max_pixel_step3(df)

    df.to_excel('/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/swedish_dataframe_max_pixels.xlsx')
    print(df)
