
# --- path configuration (see .env.example) ---
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
import pyarrow.parquet as pq


class LAIONAestheticsLoader:
    """
    Loader for LAION Aesthetics V2 dataset with filtering capabilities.
    
    Args:
        data_dir: Path to directory containing parquet files
        min_aesthetic_score: Minimum aesthetic score threshold (default: 6.0)
        min_width: Minimum image width (default: 512)
        min_height: Minimum image height (default: 512)
        batch_size: Number of rows to process at once (default: 10000)
    """
    
    def __init__(
        self,
        data_dir: str,
        min_aesthetic_score: float = 6.0,
        max_aesthetic_score: Optional[float] = None,
        min_width: int = 512,
        min_height: int = 512,
        batch_size: int = 10000
    ):
        self.data_dir = Path(data_dir)
        self.min_aesthetic_score = min_aesthetic_score
        self.max_aesthetic_score = max_aesthetic_score
        self.min_width = min_width
        self.min_height = min_height
        self.batch_size = batch_size
        
        # Find all parquet files
        self.parquet_files = sorted(self.data_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")
        
        print(f"Found {len(self.parquet_files)} parquet files")
    
    def _filter_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering criteria to a batch of data."""
        # Filter by aesthetic score
        mask = df['AESTHETIC_SCORE'] >= self.min_aesthetic_score
        
        # Filter by resolution if width/height columns exist
        if self.max_aesthetic_score is not None:
            mask &= df['AESTHETIC_SCORE'] < self.max_aesthetic_score
        if 'width' in df.columns and 'height' in df.columns:
            mask &= (df['width'] >= self.min_width) & (df['height'] >= self.min_height)
        
        return df[mask]
    
    def iter_batches(self) -> Iterator[pd.DataFrame]:
        """
        Iterate through filtered data in batches.
        
        Yields:
            pd.DataFrame: Filtered batch of data
        """
        for parquet_file in self.parquet_files:
            print(f"Processing {parquet_file.name}...")
            
            # Read parquet file in batches
            parquet_file_obj = pq.ParquetFile(parquet_file)
            
            for batch in parquet_file_obj.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                filtered_df = self._filter_batch(df)
                
                if len(filtered_df) > 0:
                    yield filtered_df
    
    def get_filtered_dataframe(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load filtered data into a single DataFrame.
        
        Args:
            max_samples: Maximum number of samples to return (None for all)
        
        Returns:
            pd.DataFrame: Filtered dataset
        """
        dfs = []
        total_samples = 0
        
        for batch_df in self.iter_batches():
            if max_samples is not None:
                remaining = max_samples - total_samples
                if remaining <= 0:
                    break
                batch_df = batch_df.head(remaining)
            
            dfs.append(batch_df)
            total_samples += len(batch_df)
            
            if max_samples is not None and total_samples >= max_samples:
                break
        
        if not dfs:
            print("Warning: No samples matched the filtering criteria")
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(result)} samples")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the filtered dataset."""
        total_samples = 0
        filtered_samples = 0
        
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            total_samples += len(df)
            filtered_samples += len(self._filter_batch(df))
        
        return {
            'total_samples': total_samples,
            'filtered_samples': filtered_samples,
            'filter_rate': filtered_samples / total_samples if total_samples > 0 else 0,
            'num_files': len(self.parquet_files)
        }


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = LAIONAestheticsLoader(
        data_dir=DATA_ROOT + "/filtered_laion_aesthetics_parquet_recaptioned/",
        min_aesthetic_score=5.5,
        # max_aesthetic_score=5.65,
        min_width=256,
        min_height=256,
        batch_size=10000
    )
    
    # Option 1: Get statistics
    stats = loader.get_stats()
    print(f"Dataset statistics: {stats}")
    
    # # Option 2: Iterate through batches (memory efficient)
    # for batch in loader.iter_batches():
    #     print(f"Processing batch with {len(batch)} samples")
    #     # Process batch here (e.g., download images, training)
    #     break  # Remove this to process all batches
    
    # # Option 3: Load all filtered data into memory
    # df = loader.get_filtered_dataframe(max_samples=100000)
    # print(df.head())
    # print(f"\nColumns: {df.columns.tolist()}")

    # Save filtered data to 8 Parquet files
    df = loader.get_filtered_dataframe()
    output_dir = DATA_ROOT + "/filtered_laion_aesthetics_parquet_recaptioned_new"
    os.makedirs(output_dir, exist_ok=True)
    num_parts = 12

    val_size = 50000
    if len(df) > val_size:
        val_df = df.sample(n=val_size, random_state=42)
        train_df = df.drop(val_df.index)
        val_path = os.path.join(output_dir, "val.parquet")
        val_df.to_parquet(val_path, index=False)
        print(f"Saved validation set with {len(val_df)} rows to {val_path}")
        df = train_df
        print(f"Training set has {len(df)} rows after removing validation set")

    chunk_size = (len(df) + num_parts - 1) // num_parts  # ceil division

    for i in range(num_parts):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))
        part_df = df.iloc[start:end]
        if not part_df.empty:
            part_path = os.path.join(output_dir, f"part_{i+1}.parquet")
            part_df.to_parquet(part_path, index=False)
            print(f"Saved {len(part_df)} rows to {part_path}")