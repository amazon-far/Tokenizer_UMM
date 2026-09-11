import argparse

from datasets import load_from_disk

def main(args):
    dataset = load_from_disk(args.folder_path)
    n_val = args.val_sample
    total = len(dataset)
    start = total - n_val
    train_ds = dataset.select(range(0, start))
    val_ds = dataset.select(range(start, total))

    # train_ds.save_to_disk("/path/to/data/journeydb_sft_tokenized_unitok_hf_train")
    if args.train_save_path is not None:
        train_ds.save_to_disk(args.train_save_path)
    if args.val_save_path is not None:
        val_ds.save_to_disk(args.val_save_path)


    # blip3o val split : 9560

    # journeydb sft split : 81319

    # laion sft split : 917427

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_path', type=str, required=True, help='path to folder containing jsonl files')
    parser.add_argument('--train_save_path', type=str, default=None, help='path to save converted hf datasets files')
    parser.add_argument('--val_save_path', type=str, required=True, help='path to save converted hf datasets files')
    parser.add_argument('--val_sample', type=int, required=True, help='number of samples to use for validation')

    main(parser.parse_args())

# python tokenized_split.py --folder_path /path/to/data/laion_tokenized_unitok_hf --val_save_path /path/to/data/laion_sft_tokenized_unitok_hf --val_sample 917427