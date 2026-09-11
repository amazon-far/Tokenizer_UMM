import argparse

from datasets import load_from_disk
import shutil
import os

def main(args):
    subset = load_from_disk(args.folder_path)
    sub_len = subset.num_rows
    # subset = subset.select(range(int(sub_len*0.0184)))

    def _to_gpt_format(batch):
        new_batch = dict(batch)
        texts = new_batch.get("text", [])
        out = []
        for t in texts:
            if isinstance(t, list):
                s = " ".join(map(str, t))
            else:
                s = "" if t is None else str(t)
            out.append([{"from": "gpt", "value": s}])
        new_batch["text"] = out
        return new_batch

    subset = subset.map(_to_gpt_format, batched=True)

    out_dir = args.save_path
    subset.save_to_disk(out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_path', type=str, default='/path/to/jsonl/files',help='path to folder containing jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    main(parser.parse_args())

# python pt_to_sft.py --folder_path /path/to/data/journeydb_sft_tokenized_llamagen_hf --save_path /path/to/data/journeydb_sft_tokenized_llamagen_hf_modified 