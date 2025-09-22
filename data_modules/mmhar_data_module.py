from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class MMHarDataset2(Dataset, metaclass=ABCMeta):
    """
    Generic Dataset implementation for multimodal human activity datasets.
    Data for each instance is loaded from disk on demand (in the __getitem__ method).
    To implement a new dataset, simply extend this class and implement the included abstract methods.
    """

    @staticmethod
    @abstractmethod
    def _supported_modalities() -> List[str]:
        """
        Must return a list of supported modalities for the dataset (using the same set of values as the modality
        column in the dataset manager's dataframe).
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _get_data_for_instance(modality, path):
        """
        Must return the data for the instance of the specified modality found at the given path.
        """
        pass

    def __init__(self,
            modalities: List[str],
            dataset_manager: Any,
            split = {},
            transforms = {},
            ssl=False,
            n_views=2,
            limited_k=None):
        """
        Initializes the data set by splitting the dataset manager's data frame into one data frame
        for each specified modality and keeping only the data for the specified subjects.

        The implementation relies on dataset_manager having a data_files_df property consisting of a Pandas
        DataFrame which contains the "subject", "path", "modality", "label", "trial" columns.

        transforms must be a dictionary where each (optional) key is a modality label, and the associated
        value is a Transform that can be applied to the data for that modality.
        """
        assert(set(modalities).issubset(set(self._supported_modalities())))
        assert(len(modalities) != 0)

        super().__init__()
        self.modalities = modalities
        self.transforms = transforms
        self.ssl = ssl
        self.n_views = n_views

        selected_df = dataset_manager.data_files_df[dataset_manager.data_files_df["modality"].isin(modalities)]
        SUBJECT_SHUFFLE_SEED = 42  # You can choose any fixed integer

        if True:
            # Create a separate random number generator with fixed seed
            rng = np.random.RandomState(SUBJECT_SHUFFLE_SEED)
            
            # Create a copy of the dataframe to avoid modifying the original during iteration
            df_copy = selected_df.copy()
            
            # Get unique activity labels
            unique_activities = df_copy['label'].unique()
            
            # For each activity, shuffle subjects within that activity only
            for activity in unique_activities:
                # Get indices for this activity
                activity_mask = df_copy['label'] == activity
                
                # Get unique subjects for this activity
                activity_subjects = df_copy.loc[activity_mask, 'subject'].unique()
                
                if len(activity_subjects) > 1:  # Only shuffle if we have multiple subjects
                    # Create shuffled mapping within this activity
                    shuffled_subjects = activity_subjects.copy()
                    rng.shuffle(shuffled_subjects)
                    subject_map = dict(zip(activity_subjects, shuffled_subjects))
                    
                    # Apply mapping only to rows with this activity
                    for orig_subject, new_subject in subject_map.items():
                        mask = (df_copy['label'] == activity) & (df_copy['subject'] == orig_subject)
                        selected_df.loc[mask, 'subject'] = new_subject
            
            print(f"Shuffled subject labels within each activity class")


        # Find all complete samples (with all modalities present) and extract their indices.
        join_columns = ["label", "subject", "trial","label2"]
        if "scene" in list(selected_df.columns):
            join_columns = ["label", "subject", "scene", "session", "label2"] # workaround for MMAct
        modalities_per_sample = selected_df.value_counts(subset=join_columns, sort=False)
        complete_samples_index = modalities_per_sample[modalities_per_sample == len(self.modalities)].index

        # Split data into separate data frames for each modality, then filter and sort the data frames.
        self.data_tables = {}
        for i, modality in enumerate(modalities):
            df = selected_df[selected_df["modality"] == modality]
            temp_idx = df.set_index(complete_samples_index.names).index
            df = df[temp_idx.isin(complete_samples_index)]
            df = df.reset_index(drop=True)

            # Filter based on the desired split.
            for key in split:
                df = df[df[key].isin(split[key])]

            df = df.sort_values(["modality", "label", "subject", "trial", "label2"], ascending=[True, True, True, True, True])
            if limited_k is not None:
                if i == 0:
                    df = self._limit_df(df, limited_k)
                    label_subject_trial = df[['label', 'subject', 'trial', "label2"]]
                else:
                    df = pd.merge(df, label_subject_trial, how='inner', on=['label', 'subject', 'trial', "label2"])
            self.data_tables[modality] = df


    @staticmethod
    def _limit_df(df, k):
        unique_labels = df.label.unique()
        limited_df = pd.DataFrame(columns=df.columns)
        for unique_l in unique_labels:
            df_l = df.loc[df['label'] == unique_l]
            num_to_sample = int(k * df_l.shape[0])
            if num_to_sample == 0:
                num_to_sample = 1
            df_l = df_l.sample(num_to_sample)
            limited_df = limited_df.append(df_l)
        return limited_df

    def __len__(self):
        return self.data_tables[self.modalities[0]].shape[0]

    def __getitem__(self, idx):
        """
        Loads the data samples for all of the specified modalities from the disk and applies the relevant transforms.
        Returns a dictionary with one key-value pair for each modality, and one key-value pair with the label of the instance.
        """
        data_sample = {}
        label = self.data_tables[self.modalities[0]].iloc[idx].loc["label"]
        data_sample["label"] = label
        data_sample["idx"] = idx
        data_sample["label2"] = self.data_tables[self.modalities[0]].iloc[idx].loc["label2"]
        subject = self.data_tables[self.modalities[0]].iloc[idx].loc["subject"]
        data_sample["subject"] = subject
        data_sample["scene"] = self.data_tables[self.modalities[0]].iloc[idx].loc["trial"]


        for modality in self.modalities:
            path = self.data_tables[modality].iloc[idx].loc["path"]
            data = self._get_data_for_instance(modality, path)

            if self.ssl and self.n_views > 1 and modality in self.transforms:
                views = [self.transforms[modality](data) for _ in range(self.n_views)]
                data_sample[modality] = views

            elif modality in self.transforms:
                data = self.transforms[modality](data)
                data_sample[modality] = data

            else:
                data_sample[modality] = data

        return data_sample


class MMHarDataset(Dataset, metaclass=ABCMeta):
    """
    Generic Dataset implementation for multimodal human activity datasets.
    Data for each instance is loaded from disk on demand (in the __getitem__ method).
    To implement a new dataset, simply extend this class and implement the included abstract methods.
    """

    @staticmethod
    @abstractmethod
    def _supported_modalities() -> List[str]:
        """
        Must return a list of supported modalities for the dataset (using the same set of values as the modality
        column in the dataset manager's dataframe).
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _get_data_for_instance(modality, path):
        """
        Must return the data for the instance of the specified modality found at the given path.
        """
        pass

    def __init__(self,
            modalities: List[str],
            dataset_manager: Any,
            split = {},
            transforms = {},
            ssl=False,
            n_views=2,
            limited_k=None):
        """
        Initializes the data set by splitting the dataset manager's data frame into one data frame
        for each specified modality and keeping only the data for the specified subjects.

        The implementation relies on dataset_manager having a data_files_df property consisting of a Pandas
        DataFrame which contains the "subject", "path", "modality", "label", "trial" columns.

        transforms must be a dictionary where each (optional) key is a modality label, and the associated
        value is a Transform that can be applied to the data for that modality.
        """
        assert(set(modalities).issubset(set(self._supported_modalities())))
        assert(len(modalities) != 0)

        super().__init__()
        self.modalities = modalities
        self.transforms = transforms
        self.ssl = ssl
        self.n_views = n_views

        selected_df = dataset_manager.data_files_df[dataset_manager.data_files_df["modality"].isin(modalities)]
        
        SUBJECT_SHUFFLE_SEED = 42  # You can choose any fixed integer

        if False:
            # Create a separate random number generator with fixed seed
            rng = np.random.RandomState(SUBJECT_SHUFFLE_SEED)
            
            # Create a copy of the dataframe to avoid modifying the original
            df_copy = selected_df.copy()
            
            # Create a composite key of subject and session
            if "session" in df_copy.columns:
                df_copy['subject_session'] = df_copy['subject'].astype(str) + '_' + df_copy['session'].astype(str)
                
                # Get unique subject-session pairs
                unique_subject_sessions = df_copy['subject_session'].unique()
                
                # Create shuffled version of these pairs
                shuffled_subject_sessions = unique_subject_sessions.copy()
                rng.shuffle(shuffled_subject_sessions)
                
                # Create mapping dictionary
                subject_session_map = dict(zip(unique_subject_sessions, shuffled_subject_sessions))
                
                # Apply mapping to create new subject and session columns
                for orig_pair, new_pair in subject_session_map.items():
                    # Find indices with the original pair
                    mask = (df_copy['subject_session'] == orig_pair)
                    
                    # Extract new subject and session from the new pair
                    new_subject, new_session = new_pair.split('_')
                    
                    # Update values
                    selected_df.loc[mask, 'subject'] = int(new_subject)
                    selected_df.loc[mask, 'session'] = int(new_session)
            else:
                # If no session column, just shuffle subjects
                unique_subjects = df_copy['subject'].unique()
                shuffled_subjects = unique_subjects.copy()
                rng.shuffle(shuffled_subjects)
                subject_map = dict(zip(unique_subjects, shuffled_subjects))
                
                for orig_subject, new_subject in subject_map.items():
                    mask = (df_copy['subject'] == orig_subject)
                    selected_df.loc[mask, 'subject'] = new_subject
            
            print("Shuffled subject-session pairs while keeping them together")

        # TODO check here 

        # Find all complete samples (with all modalities present) and extract their indices.
        join_columns = ["label", "subject", "trial"]
        


        #####################

        if "scene" in list(selected_df.columns):
            join_columns = ["label", "subject", "scene", "session"] # workaround for MMAct
        modalities_per_sample = selected_df.value_counts(subset=join_columns, sort=False)
        complete_samples_index = modalities_per_sample[modalities_per_sample == len(self.modalities)].index

        # Split data into separate data frames for each modality, then filter and sort the data frames.
        self.data_tables = {}
        for i, modality in enumerate(modalities):
            df = selected_df[selected_df["modality"] == modality]
            temp_idx = df.set_index(complete_samples_index.names).index
            df = df[temp_idx.isin(complete_samples_index)]
            df = df.reset_index(drop=True)

            # Filter based on the desired split.
            for key in split:
                df = df[df[key].isin(split[key])]

            df = df.sort_values(["modality", "label", "subject", "trial"], ascending=[True, True, True, True])
            if limited_k is not None:
                if i == 0:
                    df = self._limit_df(df, limited_k)
                    label_subject_trial = df[['label', 'subject', 'trial']]
                else:
                    df = pd.merge(df, label_subject_trial, how='inner', on=['label', 'subject', 'trial'])
            self.data_tables[modality] = df


    @staticmethod
    def _limit_df(df, k):
        unique_labels = df.label.unique()
        limited_df = pd.DataFrame(columns=df.columns)
        for unique_l in unique_labels:
            df_l = df.loc[df['label'] == unique_l]
            num_to_sample = int(k * df_l.shape[0])
            if num_to_sample == 0:
                num_to_sample = 1
            df_l = df_l.sample(num_to_sample)
            limited_df = limited_df.append(df_l)
        return limited_df

    def __len__(self):
        return self.data_tables[self.modalities[0]].shape[0]

    def __getitem__(self, idx):
        """
        Loads the data samples for all of the specified modalities from the disk and applies the relevant transforms.
        Returns a dictionary with one key-value pair for each modality, and one key-value pair with the label of the instance.
        """

        data_sample = self.data_tables[self.modalities[0]].iloc[idx].to_dict()
        data_sample["idx"] = idx
        """
        data_sample = {}
        label = self.data_tables[self.modalities[0]].iloc[idx].loc["label"]
        subject = self.data_tables[self.modalities[0]].iloc[idx].loc["subject"]


        data_sample["label"] = label
        data_sample["idx"] = idx
        data_sample["subject"] = subject
        data_sample["scene"] = self.data_tables[self.modalities[0]].iloc[idx].loc["scene"]  #TODO change this 
        """
        for modality in self.modalities:
            path = self.data_tables[modality].iloc[idx].loc["path"]
            data = self._get_data_for_instance(modality, path)

            if self.ssl and self.n_views > 1 and modality in self.transforms:
                views = [self.transforms[modality](data) for _ in range(self.n_views)]
                data_sample[modality] = views

            elif modality in self.transforms:
                data = self.transforms[modality](data)
                data_sample[modality] = data

            else:
                data_sample[modality] = data

        return data_sample







class MMHarDataModule(LightningDataModule, metaclass=ABCMeta):
    """
    Generic LightningDataModule implementation for multimodal human activity datasets.
    To implement a new dataset, simply extend this class and implement the included abstract methods.
    """

    @abstractmethod
    def _create_dataset_manager(self):
        """
        Must return an instance of a dataset manager which contains information about all of the instances
        included in the dataset.

        The dataset manager must have a data_files_df property consisting of a Pandas DataFrame which contains
        the "subject", "path", "modality", "label", "trial" columns, in order to be used with a MMHarDataset.
        """
        pass

    @abstractmethod
    def _create_train_dataset(self) -> MMHarDataset:
        """
        Must return a MMHarDataset to be used for training. This dataset will be wrapped into the train DataLoader.
        """
        pass

    @abstractmethod
    def _create_val_dataset(self) -> MMHarDataset:
        """
        Must return a MMHarDataset to be used for testing. This dataset will be wrapped into the val and test DataLoaders.
        """
        pass


    @abstractmethod
    def _create_test_dataset(self) -> MMHarDataset:
        """
        Must return a MMHarDataset to be used for testing. This dataset will be wrapped into the val and test DataLoaders.
        """
        pass

    def __init__(self,
            path: str,
            modalities: List[str],
            batch_size: int,
            split = {},
            train_transforms = {},
            test_transforms = {},
            ssl = False,
            n_views = 2,
            num_workers = 1,
            limited_k=None):
        super().__init__()
        self.path = path
        self.modalities = modalities
        self.batch_size = batch_size
        self.split = split
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.ssl = ssl
        self.n_views = n_views
        self.num_workers = num_workers
        self.limited_k = limited_k

    def _init_manager(self):
        self.dataset_manager = self._create_dataset_manager()

    def _init_dataloaders(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()
        drop_last_ssl = bool(self.ssl)
        self.train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last_ssl, num_workers=self.num_workers, pin_memory=True)
        self.test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=True)
        if "val" in self.split:
            val_dataset = self._create_val_dataset()
            self.val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last_ssl, num_workers=self.num_workers, pin_memory=True)
        else:
            self.val = None

    def setup(self, stage: Optional[str] = None):
        self._init_manager()
        self._init_dataloaders()
        
    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        if self.val is not None:
            return self.val
        return None

    def test_dataloader(self):
        return self.test

@dataclass
class MMHarDatasetProperties:
    """
    Simple data class used to bind together several related classes for easier programatic access.
    """
    manager_class: Any
    dataset_class: MMHarDataset
    datamodule_class: MMHarDataModule

if __name__ == '__main__':
    # Can't instantiate these abstract classes.
    try:
        dataset = MMHarDataset(["Depth"], {"Depth": None}, [1,2,3,4], {})
    except TypeError as err:
        print(err)

    try:
        dataset = MMHarDataModule("/home/data", ["Depth"], 64, {})
    except TypeError as err:
        print(err)