import os
import zstandard as zstd
from tqdm import tqdm
import io
import json
import argparse
import subprocess
import boto3
import tarfile
import webdataset as wds

class DCLMToWebDatasetConverter:
    def __init__(self, s3_input_bucket, s3_output_bucket, s3_output_prefix):
        self.s3_client = boto3.client('s3')
        self.s3_input_bucket = s3_input_bucket
        self.s3_output_bucket = s3_output_bucket
        self.s3_output_prefix = s3_output_prefix
    
    def read_jsonl_zst_from_s3(self, s3_path):
        """Read JSONL.zst file from S3 using streaming method"""
        cmd = f"aws s3 cp {s3_path} -"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            decompressor = zstd.ZstdDecompressor()
            stream_reader = decompressor.stream_reader(process.stdout)
            stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            
            for line in stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    
    def convert_to_liquid_format(self, data, sample_id):
        """Convert DCLM sample to Liquid format"""
        text = data.get('text', '')
        if len(text.strip()) < 10:
            return None
        
        return {
            'data_type': 'text_pretrain',
            'text': text,
            'length': len(text),
            'vqcode_512': 'no',
            'vqcode_multi768': 'no',
            'width': 'no',
            'height': 'no',
            'url': data.get('url', ''),
            'timestamp': data.get('timestamp', ''),
            'id': data.get('id', f'sample_{sample_id}')
        }
    
    def create_and_upload_tar(self, samples, tar_index):
        """Create and upload a TAR file"""
        tar_buffer = io.BytesIO()
        tar_name = f"{tar_index:06d}.tar"
        
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for sample_idx, sample in enumerate(samples):
                key = f"{sample_idx:08d}"
                
                # Add text file
                text_content = sample['text'].encode('utf-8')
                text_info = tarfile.TarInfo(name=f"{key}.txt")
                text_info.size = len(text_content)
                tar.addfile(text_info, io.BytesIO(text_content))
                
                # Add metadata JSON
                metadata = {k: v for k, v in sample.items() if k != 'text'}
                meta_content = json.dumps(metadata).encode('utf-8')
                meta_info = tarfile.TarInfo(name=f"{key}.json")
                meta_info.size = len(meta_content)
                tar.addfile(meta_info, io.BytesIO(meta_content))
        
        # Upload to S3
        s3_key = f"{self.s3_output_prefix}/{tar_name}"
        self.s3_client.put_object(
            Bucket=self.s3_output_bucket,
            Key=s3_key,
            Body=tar_buffer.getvalue(),
            ContentType='application/tar'
        )
        
        s3_url = f"s3://{self.s3_output_bucket}/{s3_key}"
        print(f"Uploaded {tar_name} with {len(samples)} samples")
        return s3_url
    
    def convert(self, s3_input_pattern, max_files=None, samples_per_tar=1000):
        """Convert DCLM files to WebDataset"""
        s3_files = self.get_s3_file_list(s3_input_pattern, self.s3_input_bucket, max_files)
        uploaded_files = []
        
        sample_id = 0
        current_samples = []
        tar_index = 0
        
        for s3_path in tqdm(s3_files, desc="Processing files"):
            print(f"Processing: {s3_path}")
            
            for data in self.read_jsonl_zst_from_s3(s3_path):
                liquid_sample = self.convert_to_liquid_format(data, sample_id)
                
                if liquid_sample:
                    current_samples.append(liquid_sample)
                    sample_id += 1
                    
                    if len(current_samples) >= samples_per_tar:
                        uploaded_url = self.create_and_upload_tar(current_samples, tar_index)
                        uploaded_files.append(uploaded_url)
                        current_samples = []
                        tar_index += 1
        
        # Upload remaining samples
        if current_samples:
            uploaded_url = self.create_and_upload_tar(current_samples, tar_index)
            uploaded_files.append(uploaded_url)
        
        # Create pattern
        pattern = self.create_pipe_pattern(uploaded_files)
        
        return uploaded_files, pattern
    
    def create_pipe_pattern(self, uploaded_files):
        """Create WebDataset pattern using pipe commands"""
        if not uploaded_files:
            return ""
        
        uploaded_files.sort()
        
        if len(uploaded_files) == 1:
            return f"pipe:aws s3 cp {uploaded_files[0]} -"
        
        # Create brace expansion pattern
        base_path = '/'.join(uploaded_files[0].split('/')[:-1])
        num_files = len(uploaded_files)
        s3_pattern = f"{base_path}/{{000000..{num_files-1:06d}}}.tar"
        
        return f"pipe:aws s3 cp {s3_pattern} -"

    def get_s3_file_list(self, base_path, bucket_name, max_files=None):
        """Get list of S3 files"""
        cmd = f"aws s3 ls {base_path} --recursive"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        files = []
        for line in result.stdout.split('\n'):
            if line.strip() and line.endswith('.jsonl.zst'):
                filename = line.split()[-1]
                full_path = f"s3://{bucket_name}/{filename}"
                files.append(full_path)
        
        if max_files:
            files = files[:max_files]
        
        return files

def test_webdataset_loading(pattern, num_samples=5):
    """Simplified test that handles WebDataset format properly"""
    print(f"Testing WebDataset loading with pattern: {pattern}")
    
    try:
        # Create WebDataset without automatic decoding
        dataset = wds.WebDataset(pattern)
        
        print("Loading raw samples...")
        sample_count = 0
        
        for sample in dataset:
            sample_count += 1
            print(f"\nSample {sample_count}:")
            print(f"  Keys: {list(sample.keys())}")
            
            # Handle the sample data
            if 'txt' in sample and 'json' in sample:
                # Extract text
                text_data = sample['txt']
                if isinstance(text_data, bytes):
                    text = text_data.decode('utf-8')
                else:
                    text = str(text_data)
                
                # Extract metadata
                json_data = sample['json']
                if isinstance(json_data, bytes):
                    metadata_str = json_data.decode('utf-8')
                else:
                    metadata_str = str(json_data)
                
                try:
                    metadata = json.loads(metadata_str)
                    
                    print(f"  Data type: {metadata.get('data_type', 'unknown')}")
                    print(f"  Text length: {len(text)}")
                    print(f"  Text preview: {text[:100]}...")
                    print(f"  URL: {metadata.get('url', 'none')}")
                    
                except json.JSONDecodeError as e:
                    print(f"  JSON decode error: {e}")
                    print(f"  Raw JSON data: {metadata_str[:200]}...")
            else:
                print(f"  Unexpected sample structure: {sample}")
            
            if sample_count >= num_samples:
                break
        
        if sample_count > 0:
            print(f"\n✓ Successfully loaded {sample_count} samples!")
            return True
        else:
            print("\n✗ No samples found!")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    # Parse S3 output path
    s3_output_path_clean = args.s3_output_path[5:]  # Remove 's3://'
    bucket_name = s3_output_path_clean.split('/')[0]
    s3_prefix = '/'.join(s3_output_path_clean.split('/')[1:])
    
    converter = DCLMToWebDatasetConverter(args.s3_bucket, bucket_name, s3_prefix)
    
    # Convert
    uploaded_files, pattern = converter.convert(
        s3_input_pattern=args.s3_pattern,
        max_files=args.max_files,
        samples_per_tar=args.samples_per_tar
    )
    
    # Save pattern
    with open(args.pattern_output_file, 'w') as f:
        f.write(pattern)
    
    print(f"Conversion complete!")
    print(f"Created {len(uploaded_files)} TAR files")
    print(f"Pattern: {pattern}")
    print(f"Saved to: {args.pattern_output_file}")

    if args.test_dataset:
        print("\nTesting WebDataset loading...")
        test_webdataset_loading(pattern, args.test_samples)

# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DCLM to WebDataset using pipe method')
    parser.add_argument('--s3-pattern', type=str, 
                       default='s3://your-s3-bucket/datasets/dclm-baseline-1.0/hf_hub_snapshot/',
                       help='S3 path pattern for DCLM files')
    parser.add_argument('--s3-bucket', type=str, 
                       default='your-s3-bucket',
                       help='S3 path bucket name for DCLM files')
    parser.add_argument('--s3-output-path', type=str, 
                       default='s3://your-s3-bucket/datasets/dclm_webdataset/',
                       help='S3 path to save WebDataset files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of DCLM files to process')
    parser.add_argument('--samples-per-tar', type=int, default=10000,
                       help='Number of samples per TAR file')
    parser.add_argument('--pattern-output-file', type=str, default='webdataset_pattern.txt',
                       help='File to save the WebDataset pattern')
    parser.add_argument('--test-dataset', action='store_true',
                       help='Test the created dataset after conversion')
    parser.add_argument('--test-samples', type=int, default=10,
                       help='Number of samples to test in streaming mode')
    
    args = parser.parse_args()
    main(args)


