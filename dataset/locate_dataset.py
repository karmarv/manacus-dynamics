import os, time
import shutil
import pandas as pd


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

def is_available(x):
    file_path = os.path.join(data_dir, str(x["Lek"]), str(x["Pista"]), str(x["DateRange_Folder"]), str(x["FileName"]))
    if os.path.isfile(file_path):
        return pd.Series([True, file_path])
    else:
        return pd.Series([False, file_path])

def verify_available(df):
    # Corrected the filenames manually in source file (Spanish to English mapping)
    #lambda x: np.square(x) if x.name in ['x', 'y'] else x
    df[["is_available", "local_path"]] = df.apply(is_available, axis=1)
    return df

def filter_metadata_df(m_df):
    # A. courtship success [female visitation]
    fv_df = m_df[m_df["FemVisitation"].isin([1, 2]) | m_df["courtshipSuccess"].isin([0,1,2])]
    
    # B. Local video file is available
    fv_df = fv_df[fv_df["is_available"] == True]
    print(fv_df)
    return fv_df

def save_video_df(f_df):
    count = 0
    for index, x in f_df.iterrows():
        src_path = os.path.join(data_dir, str(x["Lek"]), str(x["Pista"]), str(x["DateRange_Folder"]), str(x["FileName"]))
        tgt_name = "{}-{}-{}-{}".format(str(x["Lek"]), str(x["Pista"]), str(x["DateRange_Folder"]), str(x["FileName"]))
        tgt_path = os.path.join("videos", tgt_name)
        try:
            shutil.copy(src_path, tgt_path)
        except Exception as e:
            print(e)
        count = count+1
    print("Copied {} items to target".format(count))
    return

"""
Read the human-annotated video spreadsheets for sampling videos containing male/female birds
- Both females and juvenile males have the same green plumage. So, they can't be reliably distinguished visually, but they can be distinguished based on behavioral cues. 
- In these spreadsheets, the column 'courtshipSuccess' denotes videos that contain females (based on their participation in the courtship display with an adult male), whereas the column 'JuvPresent' indicates videos that contain juvenile males (based on their performance of male display components and/or attendance on the court without an adult male present).
"""
if __name__ == "__main__":

    # Data file configurations
    data_dir  = "/mnt/c/workspace/eebio/manacus-dynamics/box_data/PROJECT MANACUS/Camera Traps 1 -- Dec 2021 to Jan 2022" 
    data_file = "./spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx"
    
    # Availability of video as per metadata from file 
    m_df = verify_available(read_metadata_file(data_file))
    f_df = filter_metadata_df(m_df)
    print("Filtered available video file:\n", f_df)
    f_df.to_csv(os.path.join("videos", "local_available.csv"), index_label='Index')    
    save_video_df(f_df)


    # Download worklow
    #box_files = BoxNavigator()
    