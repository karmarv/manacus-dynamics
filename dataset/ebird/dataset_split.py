import os
import cv2
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


"""
Read the labeled ebird metadata file
Headers:
  - ML Catalog Number,Format,Common Name,Scientific Name,Background Species,Recordist,
  - Date,Year,Month,Day,Time,Country,Country-State-County,State,County,Locality,Latitude,Longitude,
  - Age/Sex,Behaviors,Playback,Captive,Collected,Specimen ID,Home Archive Catalog Number,Recorder,Microphone,Accessory,
  - Partner Institution,eBird Checklist ID,Unconfirmed,Air Temp(°C),Water Temp(°C),Media notes,Observation Details,Parent Species,
  - eBird Species Code,Taxon Category,Taxonomic Sort,Recordist 2,Average Community Rating,Number of Ratings,Asset Tags,
  - Original Image Height,Original Image Width
"""
def read_metadata_file(data_file):
    # read CSV keeping encoding - https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas
    df = pd.read_csv(data_file)
    print(df.describe())
    return df

def is_available(x):
    file_path = os.path.join("./images", str(x["ML Catalog Number"])+".jpg")
    # Check sex
    sex = "Unknown"
    if "Female" in str(x["Age/Sex"]):
        sex = "Female"
    elif "Male" in str(x["Age/Sex"]):
        sex = "Male"
    
    # Check age
    age = "Unknown"
    if "Adult" in str(x["Age/Sex"]):
        age = "Adult"
    elif "Juvenile" in str(x["Age/Sex"]):
        age = "Juvenile"
    elif "Immature" in str(x["Age/Sex"]):
        age = "Juvenile"

    # Check file availability
    is_available = False
    if os.path.isfile(file_path):
        is_available = True

    return pd.Series([age, sex, is_available, file_path])

def verify_available(df):
    # Corrected the filenames manually in source file (Spanish to English mapping)
    #lambda x: np.square(x) if x.name in ['x', 'y'] else x
    df[["age", "sex", "is_available", "local_path"]] = df.apply(is_available, axis=1)
    return df

def convert_images(data_dir, glob_ext='*.png'):
    # path to search file
    path = os.path.join(data_dir, glob_ext)
    images_path = Path(data_dir)
    for filepath in images_path.glob(glob_ext):
        image = cv2.imread(str(filepath))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filepath_new = "images/{}.jpg".format(filepath.stem)
        print(filepath, "\t", filepath_new) 
        cv2.imwrite(filepath_new, image)

def split_dataset(df):
    X, y = df["local_path"], df["sex"]
    X_train, X_vt, y_train, y_vt = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True, stratify=y)
    print("Train split - 80%")
    print(y_train.value_counts())
    train_df = pd.DataFrame({'image': X_train,'sex':y_train, 'idx_col':X_train.index})
    train_df.to_csv("train_images.csv", index=False)

    X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, train_size=0.5, random_state=0, shuffle=True, stratify=y_vt)
    print("Val/Test split - 10%")
    print(y_val.value_counts())
    val_df = pd.DataFrame({'image': X_val,'sex':y_val, 'idx_col':X_val.index})
    val_df.to_csv("val_images.csv", index=False)
    print(y_test.value_counts())
    test_df = pd.DataFrame({'image': X_test,'sex':y_test, 'idx_col':X_test.index})
    test_df.to_csv("test_images.csv", index=False)
    return (X_train, y_train),  (X_val, y_val), (X_test, y_test)

def verify_split_dataset(data_file):
    # Availability of video as per metadata from file 
    m_df = verify_available(read_metadata_file(data_file))
    print("Sex - ", m_df["sex"].unique())
    print("Age - ", m_df["age"].unique())
    print(m_df.head())

    # Split the dataset
    split_dataset(m_df[["ML Catalog Number", "Age/Sex", "age", "sex", "is_available", "local_path"]])




"""
Read the human-annotated video spreadsheets for sampling videos containing male/female birds
- Both females and juvenile males have the same green plumage. So, they can't be reliably distinguished visually, but they can be distinguished based on behavioral cues. 
- In these spreadsheets, the column 'courtshipSuccess' denotes videos that contain females (based on their participation in the courtship display with an adult male), whereas the column 'JuvPresent' indicates videos that contain juvenile males (based on their performance of male display components and/or attendance on the court without an adult male present).
"""
if __name__ == "__main__":

    # Data file configurations
    data_file = "./ML__2024-04-12T18-59_whbman1_photo.csv"
    
    # Convert all the images to JPG type
    #convert_images("/home/rahul/workspace/eeb/manacus-project/data-ebird-manacus/images")

    # Split dataset as train=80, val=10, test=10 keeping sex distribution similar
    verify_split_dataset(data_file)




