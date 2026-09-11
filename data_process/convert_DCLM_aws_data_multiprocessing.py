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
import multiprocessing as mp


def create_and_upload_tar(s3_client, s3_output_bucket, s3_output_prefix, samples, tar_index):
        """Create and upload a TAR file"""
        tar_buffer = io.BytesIO()
        tar_name = f"{tar_index:06d}.tar"

        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for sample_idx, sample in enumerate(samples):
                key = f"{sample_idx:08d}"

                # Add text file
                text_content = sample["text"].encode("utf-8")
                text_info = tarfile.TarInfo(name=f"{key}.txt")
                text_info.size = len(text_content)
                tar.addfile(text_info, io.BytesIO(text_content))

                # Add metadata JSON
                metadata = {k: v for k, v in sample.items() if k != "text"}
                meta_content = json.dumps(metadata).encode("utf-8")
                meta_info = tarfile.TarInfo(name=f"{key}.json")
                meta_info.size = len(meta_content)
                tar.addfile(meta_info, io.BytesIO(meta_content))

        # Upload to S3
        s3_key = f"{s3_output_prefix}/{tar_name}"
        s3_client.put_object(
            Bucket=s3_output_bucket,
            Key=s3_key,
            Body=tar_buffer.getvalue(),
            ContentType="application/x-tar",
        )

        s3_url = f"s3://{s3_output_bucket}/{s3_key}"
        print(f"Uploaded {tar_name} with {len(samples)} samples")
        return s3_url

def read_jsonl_zst_from_s3(s3_path):
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


def convert_to_liquid_format(data, sample_id):
        """Convert DCLM sample to Liquid format"""
        text = data.get("text", "")
        # if len(text.strip()) < 10:
        #     return None

        return {
            "data_type": "text_pretrain",
            "text": text,
            "length": len(text),
            "vqcode_512": "no",
            "vqcode_multi768": "no",
            "width": "no",
            "height": "no",
            "url": data.get("url", ""),
            "timestamp": data.get("timestamp", ""),
            "id": data.get("id", f"sample_{sample_id}"),
        }


# ---------- NEW: Worker function ----------
def process_single_file(args):
    """Process a single S3 file -> return uploaded TAR URLs."""
    s3_client = boto3.client("s3")
    s3_path, samples_per_tar, start_tar_index, s3_output_bucket, s3_output_prefix = args
    uploaded_files = []
    sample_id = 0
    current_samples = []
    tar_index = start_tar_index

    print(f"[Worker {os.getpid()}] Processing {s3_path}")

    for data in read_jsonl_zst_from_s3(s3_path):
        liquid_sample = convert_to_liquid_format(data, f"{tar_index}_{sample_id}")
        if liquid_sample:
            current_samples.append(liquid_sample)
            sample_id += 1

            if len(current_samples) >= samples_per_tar:
                uploaded_url = create_and_upload_tar(s3_client, s3_output_bucket, s3_output_prefix, current_samples, tar_index)
                uploaded_files.append(uploaded_url)
                current_samples = []
                tar_index += 1
                sample_id = 0  # Reset for next TAR

    if current_samples:
        uploaded_url = create_and_upload_tar(s3_client, s3_output_bucket, s3_output_prefix, current_samples, tar_index)
        uploaded_files.append(uploaded_url)
        tar_index += 1

    return uploaded_files, tar_index

class DCLMToWebDatasetConverter:
    def __init__(self, s3_input_bucket, s3_output_bucket, s3_output_prefix):
        self.s3_client = boto3.client("s3")
        self.s3_input_bucket = s3_input_bucket
        self.s3_output_bucket = s3_output_bucket
        self.s3_output_prefix = s3_output_prefix

    def convert(self, s3_input_pattern, max_files=None, samples_per_tar=1000, num_workers=4):
        """Convert DCLM files to WebDataset in parallel"""
        s3_files = self.get_s3_file_list(s3_input_pattern, self.s3_input_bucket, max_files)

        # Assign disjoint ID ranges per worker to avoid collisions
        tasks = []
        tar_index = 0
        for s3_path in s3_files:
            tasks.append((s3_path, samples_per_tar, tar_index, self.s3_output_bucket, self.s3_output_prefix))
            tar_index += 10**6

        uploaded_files = []
        with mp.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_single_file, tasks),
                               total=len(tasks), desc="Processing files"):
                files, _ = result
                uploaded_files.extend(files)

        pattern = self.create_pipe_pattern(uploaded_files)
        return uploaded_files, pattern

    def create_pipe_pattern(self, uploaded_files):
        if not uploaded_files:
            return ""

        uploaded_files.sort()
        if len(uploaded_files) == 1:
            return f"pipe:aws s3 cp {uploaded_files[0]} -"

        # Return explicit brace pattern only if indices are sequential
        return [f"pipe:aws s3 cp {u} -" for u in uploaded_files]

    def get_s3_file_list(self, base_path, bucket_name, max_files=None):
        """Get list of S3 files"""
        cmd = f"aws s3 ls {base_path} --recursive"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)

        files = []
        for line in result.stdout.split("\n"):
            if line.strip() and line.endswith(".jsonl.zst"):
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
        dataset = wds.WebDataset(pattern)
        sample_count = 0

        for sample in dataset:
            sample_count += 1
            print(f"\nSample {sample_count}:")
            print(f"  Keys: {list(sample.keys())}")

            if "txt" in sample and "json" in sample:
                text_data = sample["txt"]
                text = text_data.decode("utf-8") if isinstance(text_data, bytes) else str(text_data)

                json_data = sample["json"]
                metadata_str = json_data.decode("utf-8") if isinstance(json_data, bytes) else str(json_data)

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
    bucket_name = s3_output_path_clean.split("/")[0]
    s3_prefix = "/".join(s3_output_path_clean.split("/")[1:])

    converter = DCLMToWebDatasetConverter(args.s3_bucket, bucket_name, s3_prefix)

    # Convert in parallel
    uploaded_files, pattern = converter.convert(
        s3_input_pattern=args.s3_pattern,
        max_files=args.max_files,
        samples_per_tar=args.samples_per_tar,
        num_workers=args.num_workers,
    )

    print(f"Conversion complete!")
    print(f"Created {len(uploaded_files)} TAR files")
    print(f"Pattern: {pattern}")
    print(f"Saved to: {args.pattern_output_file}")

    if args.test_dataset:
        print("\nTesting WebDataset loading...")
        test_webdataset_loading(pattern, args.test_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCLM to WebDataset using multiprocessing")
    parser.add_argument('--s3-pattern', type=str, 
                       default='s3://your-s3-bucket/datasets/dclm-baseline-1.0/hf_hub_snapshot/',
                       help='S3 path pattern for DCLM files')
    parser.add_argument('--s3-bucket', type=str, 
                       default='your-s3-bucket',
                       help='S3 path bucket name for DCLM files')
    parser.add_argument('--s3-output-path', type=str, 
                       default='s3://your-s3-bucket/datasets/dclm_webdataset',
                       help='S3 path to save WebDataset files')
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of DCLM files to process")
    parser.add_argument("--samples-per-tar", type=int, default=10000, help="Number of samples per TAR file")
    parser.add_argument("--pattern-output-file", type=str, default="webdataset_pattern.txt",
                        help="File to save the WebDataset pattern")
    parser.add_argument("--test-dataset", action="store_true", help="Test the created dataset after conversion")
    parser.add_argument("--test-samples", type=int, default=10, help="Number of samples to test in streaming mode")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes")

    args = parser.parse_args()
    main(args)
