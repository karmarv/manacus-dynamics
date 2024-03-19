import os, time
import pandas as pd
from boxsdk import OAuth2, Client

"""
Data file configurations
"""
data_dir  = "/mnt/c/workspace/eebio/manacus-dynamics/box_data/PROJECT MANACUS/Camera Traps 1 -- Dec 2021 to Jan 2022" 
data_file = "Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx"

class BoxNavigator():
    def __init__(self, home_dir):
        
        self.home_dir = home_dir
        self.load_credentials_from_file()
        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)
            print(f"Data folder: {self.home_dir}.")

    def load_credentials_from_file(self):
        credentials_file_path = 'box_credentials.pass'
        with open(credentials_file_path, 'r') as file:
            lines = file.readlines()
            self.client_id = lines[0].strip()
            self.client_secret = lines[1].strip()
            self.access_token = lines[2].strip()

        self.auth = OAuth2(client_id=self.client_id, client_secret=self.client_secret, access_token=self.access_token)
        self.client = Client(self.auth)

def read_metadata_file(data_file):
    xls = pd.ExcelFile(data_file)
    # Read first sheet in excel
    df = pd.read_excel(data_file, sheet_name=xls.sheet_names[0])
    # Snaps	quietSnaps	Grunts	Rolls	Calls	FemVisitation	courtshipSuccess	Copulation	JuvPresent	Gardening	vidLength
    #df = df.astype({'FemVisitation': 'str', 'Copulation': 'str'})
    # Read content
    print(xls.sheet_names)
    print(df.describe())
    return df

def filter_metadata_df(m_df):
    # Filter criteria for processing
    fv_df = m_df[m_df["FemVisitation"].isin([1, 2]) | m_df["Copulation"].isin([0])]
    print(fv_df)
    return fv_df

def is_available(x):
    file_path = os.path.join(data_dir, x["Lek"], x["Pista"], x["DateRange_Folder"], x["FileName"])
    if os.path.isfile(file_path):
        return pd.Series([True, file_path])
    else:
        return pd.Series([False, file_path])

def verify_available(df):
    #lambda x: np.square(x) if x.name in ['x', 'y'] else x
    df[["is_available", "local_path"]] = df.apply(is_available, axis=1)
    print("Verify file availability:\n", df)
    return df

if __name__ == "__main__":
    # Read metadata from file 
    m_df = read_metadata_file(data_file)
    f_df = verify_available(filter_metadata_df(m_df))
    f_df.to_csv("local_available.csv", index_label='Index')
    
    # Correct the filenames workflow (Spanish to English mapping)

    # Download worklow
    #box_files = BoxNavigator()
    