"""tf.data pipeline builders for efficient data loading."""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union


class CosmologicalDataPipeline:
    """
    Efficient tf.data pipeline for cosmological H(z) data.
    
    Features:
    - Parallel data loading
    - Automatic batching
    - Prefetching for GPU utilization
    - Caching for faster subsequent epochs
    - Optional shuffling
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "data_{}.npy",
        batch_size: int = 32,
        validation_split: float = 0.2,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        seed: Optional[int] = None,
    ):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Directory containing .npy files
            file_pattern: Glob pattern for file names
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle data
            shuffle_buffer_size: Buffer size for shuffling
            cache: Whether to cache dataset after first epoch
            num_parallel_calls: Parallelism for data loading
            prefetch_buffer_size: Buffer size for prefetching
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache = cache
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_buffer_size = prefetch_buffer_size
        self.seed = seed
        
        self.file_paths = sorted(self.data_dir.glob(file_pattern.replace("{}", "*")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No files found in {data_dir} matching {file_pattern}")
    
    def _parse_single_file(self, file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parse a single .npy file into features and target."""
        def parse(path):
            data = np.load(path.numpy())
            features = data[:, :2].astype(np.float32)
            n_points = len(data)
            target = data[0, 2] / n_points
            target = np.array(target, dtype=np.float32)
            return features, target
        
        features, target = tf.py_function(
            func=parse,
            inp=[file_path],
            Tout=[tf.float32, tf.float32]
        )
        
        n_points = len(np.load(self.file_paths[0]))
        features.set_shape([n_points, 2])
        target.set_shape([])
        
        return features, target
    
    def _preprocess_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply preprocessing steps to dataset."""
        if self.shuffle:
            dataset = dataset.shuffle(
                self.shuffle_buffer_size,
                seed=self.seed,
                reshuffle_each_iteration=True
            )
        
        if self.cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        
        return dataset
    
    def get_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Get training and validation datasets."""
        dataset = tf.data.Dataset.from_tensor_slices(
            [str(p) for p in self.file_paths]
        )
        
        dataset = dataset.map(
            self._parse_single_file,
            num_parallel_calls=self.num_parallel_calls
        )
        
        dataset = self._preprocess_dataset(dataset)
        
        n_files = len(self.file_paths)
        n_train = int(n_files * (1 - self.validation_split))
        
        train_dataset = dataset.take(n_train)
        val_dataset = dataset.skip(n_train)
        
        return train_dataset, val_dataset
    
    def get_tf_dataset(self) -> tf.data.Dataset:
        """Get a single combined tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(
            [str(p) for p in self.file_paths]
        )
        
        dataset = dataset.map(
            self._parse_single_file,
            num_parallel_calls=self.num_parallel_calls
        )
        
        if self.shuffle:
            dataset = dataset.shuffle(
                self.shuffle_buffer_size,
                seed=self.seed,
                reshuffle_each_iteration=True
            )
        
        if self.cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        
        return dataset
    
    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.file_paths)
    
    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.n_samples * (1 - self.validation_split))
    
    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return int(self.n_samples * self.validation_split)


def create_training_pipeline(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    validation_split: float = 0.2,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Convenience function to create training pipeline."""
    pipeline = CosmologicalDataPipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        **kwargs
    )
    
    return pipeline.get_dataset()