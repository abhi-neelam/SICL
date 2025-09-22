from datasets.utd_mhad import UTDDatasetManager

from datasets.l2 import l2Manager
from datasets.mmact import MMActDatasetManager
from data_modules.utd_mhad_data_module import UTDDataModule, UTDDataset
from data_modules.l2_data_module import l2DataModule
from data_modules.mmact_data_module import MMActDataModule, MMActDataset
from data_modules.mmhar_data_module import MMHarDatasetProperties

DATASET_PROPERTIES = {
    "utd_mhad": MMHarDatasetProperties(
        manager_class=UTDDatasetManager,
        dataset_class=UTDDataset,
        datamodule_class=UTDDataModule
    ),
    "mmact": MMHarDatasetProperties(
        manager_class=MMActDatasetManager,
        dataset_class=MMActDataset,
        datamodule_class=MMActDataModule
    ),
    "l1": MMHarDatasetProperties(
        manager_class=UTDDatasetManager,
        dataset_class=UTDDataset,
        datamodule_class=UTDDataModule
    ),
    "l2": MMHarDatasetProperties(
        manager_class=l2Manager,
        dataset_class=UTDDataset,
        datamodule_class=l2DataModule
    ),
    "l3": MMHarDatasetProperties(
    manager_class=l2Manager,
    dataset_class=UTDDataset,
    datamodule_class=l2DataModule
    )

}