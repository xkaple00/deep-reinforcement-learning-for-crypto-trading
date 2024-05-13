import os
import numpy as np
from typing import Tuple

class DiskDataLoader:
    def __init__(self, dataset_name: str = 'dataset') -> None:
        self.dataset_name = dataset_name
        self.path_root = 'data'

        self.path_dataset = os.path.join(
            self.path_root,
            self.dataset_name
        )

        try:
            self.price_array_total = np.load(
                os.path.join(self.path_dataset, 'price_outfile.npy'), mmap_mode='r'
            )
        except:
            print('price_outfile.npy does not exist in {}'.format(self.path_dataset))
            self.price_array_total = None

        try:
            self.tech_array_total = np.load(
                os.path.join(self.path_dataset, 'metrics_outfile.npy'),
                mmap_mode="r",
            )
        except:
            print('metrics_outfile.npy does not exist in {}'.format(self.path_dataset))
            self.tech_array_total = None

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.price_array_total, self.tech_array_total