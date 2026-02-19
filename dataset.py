import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import glob
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
import config

class QuickDrawDataset(Dataset):
    def __init__(self, npy_dir, ndjson_dir, max_samples_per_class=None, 
                 split=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Args:
            npy_dir (str): Path to folder containing .npy files
            ndjson_dir (str): Path to folder containing .ndjson files
            max_samples_per_class (int): Optional limit for debugging (e.g., 1000)
            split (str): One of 'train', 'val', 'test', or None for full dataset
            train_ratio (float): Proportion of data for training (default 0.7)
            val_ratio (float): Proportion of data for validation (default 0.15)
            test_ratio (float): Proportion of data for testing (default 0.15)
            random_state (int): Random seed for reproducible splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        self.ndjson_files = sorted(glob.glob(os.path.join(ndjson_dir, "*.ndjson")))

        # Verify alignment
        assert len(self.npy_files) == len(self.ndjson_files), "Mismatch in number of .npy and .ndjson files!"

        self.classes = [os.path.basename(f).replace('.npy', '') for f in self.npy_files]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self._all_indices = []  # Maps global_idx -> (class_idx, local_idx)
        self._all_metadata = []  # Stores compact metadata
        self.image_maps = []  # Stores memory-mapped numpy arrays

        print(f"Initializing Dataset with {len(self.classes)} classes...")

        for class_idx, (npy_path, ndjson_path) in enumerate(zip(self.npy_files, self.ndjson_files)):
            cls_name = self.classes[class_idx]

            # 1. Load Images (Memory Mapped - Instant & Low RAM)
            # This creates a "window" to the file on disk without reading it all
            img_data = np.load(npy_path, mmap_mode='r')
            self.image_maps.append(img_data)

            # 2. Load Metadata (Line by Line)
            count = 0
            print(f"Processing metadata for {cls_name}...")

            with open(ndjson_path, 'r') as f:
                for line in f:
                    if max_samples_per_class and count >= max_samples_per_class:
                        break

                    try:
                        # Extract only what we need for error slicing
                        entry = json.loads(line)
                        # Parse timestamp to global Unix timestamp (UTC, seconds)
                        ts_val = entry.get('timestamp')
                        if ts_val is not None:
                            try:
                                s = str(ts_val).replace(' UTC', "")[:-6]
                                dt = datetime.fromisoformat(s)
                                timestamp = dt.timestamp() if dt.tzinfo else datetime(*dt.timetuple()[:6], tzinfo=timezone.utc).timestamp()
                            except (ValueError, TypeError):
                                timestamp = 0.0
                        else:
                            timestamp = 0.0
                        #convert timestamp to days since Jan 1, 1970
                        timestamp = timestamp // (60 * 60 * 24)
                        drawing = entry.get('drawing', [])
                        meta = {
                            'country': entry.get('countrycode', 'UNK'),
                            'recognized': entry.get('recognized', False),
                            'key_id': entry.get('key_id', ''),
                            'timestamp': timestamp,
                            'word': entry.get('word', ''),
                            'num_strokes': len(drawing) if isinstance(drawing, list) else 0,
                        }

                        # Store mapping info
                        self._all_indices.append((class_idx, count))
                        self._all_metadata.append(meta)
                        count += 1
                    except json.JSONDecodeError:
                        continue

            # Validation: Ensure .npy has enough rows for the metadata we read
            if img_data.shape[0] < count:
                print(f"Warning: {cls_name}.npy has fewer images than metadata rows! Truncating...")
                # Truncate metadata to match valid images
                truncate_len = img_data.shape[0]
                # Remove the extra entries we just added
                added_indices = count - truncate_len
                if added_indices > 0:
                    self._all_indices = self._all_indices[:-added_indices]
                    self._all_metadata = self._all_metadata[:-added_indices]

        # Create train/val/test splits
        all_idx = np.arange(len(self._all_indices))
        
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=test_ratio, random_state=random_state
        )
        
        # Second split: separate train and val from remaining
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adjusted, random_state=random_state
        )
        
        # Store split indices
        self._split_indices = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        # Set the active split
        self.split = split
        if split is not None:
            assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
            self._active_indices = self._split_indices[split]
        else:
            self._active_indices = all_idx
    
    def get_split_indices(self, split):
        """Return the indices for a specific split."""
        return self._split_indices[split]
    
    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, idx):
        # Map from split index to global index
        global_idx = self._active_indices[idx]
        
        # 1. Find which file and which row this index belongs to
        class_idx, local_idx = self._all_indices[global_idx]

        # 2. Retrieve Image from Disk (Lazy Loading)
        # Reshape flat vector (784) to (1, 28, 28)
        img_array = self.image_maps[class_idx][local_idx]
        img_tensor = torch.from_numpy(img_array).float().div(255.0).view(1, 28, 28)

        # 3. Retrieve Metadata
        meta = self._all_metadata[global_idx]
        cls_name = self.classes[class_idx]
        label = class_idx

        return {
            'image': img_tensor,  # Input for Model
            'label': label,  # Ground Truth
            'label_name': cls_name,  # Metadata for Slicing
            'country': meta['country'],  # Metadata for Slicing
            'recognized': meta['recognized'],  # Metadata for Slicing
            'key_id': meta['key_id'],  # Unique ID for tracking
            'timestamp': meta['timestamp'],  # Global Unix timestamp (UTC)
            'word': meta['word'],  # Prompt word (for validation)
            'num_strokes': meta['num_strokes'],  # Drawing complexity
        }



# Initialize (This takes about 30s-1min to parse 5M metadata lines)
# Set max_samples_per_class=1000 first to test quickly!
def load_dataset(max_samples_per_class=None, split=None):
    """
    Load the QuickDraw dataset.
    
    Args:
        max_samples_per_class (int): Optional limit for debugging
        split (str): One of 'train', 'val', 'test', or None for full dataset
    
    Returns:
        QuickDrawDataset: Dataset instance for the specified split
    """
    return QuickDrawDataset(config.NPY_PATH, config.NDJSON_PATH, max_samples_per_class, split=split)

def create_dataloader(max_samples_per_class=None, split='train', batch_size=64, shuffle=True):
    """
    Create a DataLoader for the specified split.
    
    Args:
        max_samples_per_class (int): Optional limit for debugging
        split (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
    """
    dataset = load_dataset(max_samples_per_class=max_samples_per_class, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
