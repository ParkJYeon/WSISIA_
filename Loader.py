import PIL
from torch.utils import data
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, dataframe, split = None, transfer = None):
        self.dataframe = dataframe
        if split != None:
            index_split = self.dataframe[dataframe['Split'] == split].index
            self.dataframe = self.dataframe.loc[index_split, :]
        self.transfer = transfer
        self.length = len(self.dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        patch = self.dataframe.iloc[item, :]
        patientID = patch['PatientID']
        patchID = str(patch['PatchID'])
        path = "./patchextract/" + patientID + "/" + patchID + ".png"
        label = patch[['Month', 'Event']].values.astype('float')
        img = PIL.Image.open(path)
        if self.transfer != None:
            img = self.transfer(img)

        return img, label

    @property
    def numofPatients(self):
        return len(self.dataframe['PatientID'].value_counts())

    @property
    def numofPatch(self):
        return self.dataframe['PatientID'].value_counts()

    @property
    def numoftotalPatch(self):
        return self.length

    def split_cluster(file, split, transfer = None):
        df = pd.read_csv(file)
        index0 = df[df['Cluster'] == 0].index
        index1 = df[df['Cluster'] == 1].index
        index2 = df[df['Cluster'] == 2].index
        index3 = df[df['Cluster'] == 3].index
        index4 = df[df['Cluster'] == 4].index
        index5 = df[df['Cluster'] == 5].index
        index6 = df[df['Cluster'] == 6].index
        index7 = df[df['Cluster'] == 7].index
        index8 = df[df['Cluster'] == 8].index
        index9 = df[df['Cluster'] == 9].index

        return PatchData(df.loc[index0, :], split=split, transfer=transfer),\
               PatchData(df.loc[index1, :], split=split, transfer=transfer),\
               PatchData(df.loc[index2, :], split=split, transfer=transfer),\
               PatchData(df.loc[index3, :], split=split, transfer=transfer),\
               PatchData(df.loc[index4, :], split=split, transfer=transfer),\
               PatchData(df.loc[index5, :], split=split, transfer=transfer),\
               PatchData(df.loc[index6, :], split=split, transfer=transfer),\
               PatchData(df.loc[index7, :], split=split, transfer=transfer),\
               PatchData(df.loc[index8, :], split=split, transfer=transfer),\
               PatchData(df.loc[index9, :], split=split, transfer=transfer)

    def load_split(file, split, transfer = None):
        df = pd.read_csv(file)
        return PatchData(df, split = split, transfer=transfer)

    def split_train_valid(self):
        onedf = self.dataframe[['PatientID', 'Event']].drop_duplicates()
        onesize = len(onedf)
        test_size = round(onesize * 0.2)
        train_idx, valid_idx = train_test_split(onedf['PatientID'].values, test_size=test_size, stratify=onedf['Event'].values, shuffle = True, random_state=0)

        transfers = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.57806385, 0.57806385, 0.57806385], [0.00937135, 0.00937135, 0.00937135])
        ])

        training = self.dataframe[self.dataframe['PatientID'].isin(train_idx)]
        valid = self.dataframe[self.dataframe['PatientID'].isin(valid_idx)]

        return PatchData(training, transfer=transfers), PatchData(valid, transfer=transfers)

    def get_label(self):
        label = self.dataframe[['Month', 'Event']].values.astype('float')
        return label
