import os
import json
import argparse

from datasets import load_dataset

def main(args):
    
    file_list = os.listdir(args.temp_path)
    data_list = [os.path.join(args.temp_path,filename) for filename in file_list ]
    for file_path in data_list:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
            for i, sample in enumerate(data):
                for j, conversation in enumerate(sample["text"]):
                    if set(conversation.keys()) != {"from", "value"}:
                        print(f"Unexpected keys in conversation: {conversation.keys()} in file {file_path}")
                        # Remove unexpected keys from the conversation
                        s = {"from": conversation["from"], "value": conversation["value"]}
                        data[i]["text"][j] = s
        with open(file_path, 'w') as fw:
            for item in data:
                fw.write(json.dumps(item) + '\n')
    
    data_files = {args.fold:data_list }
    my_dataset = load_dataset("json", data_files=data_files, split=args.fold, streaming=False, num_proc=40)
 
    my_dataset = my_dataset.shuffle(seed=42)
    my_dataset.save_to_disk(args.save_path, num_shards=128)
    
# def main(args):
    
#     file_list = os.listdir(args.temp_path)
#     data_list = [os.path.join(args.temp_path,filename) for filename in file_list ]
#     data_files = {"train":data_list }
#     my_dataset = load_dataset("json", data_files=data_files, split='train', streaming=False, num_proc=40)
 
#     my_dataset = my_dataset.shuffle(seed=42)
#     my_dataset.save_to_disk(args.save_path, num_shards=256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fold', type=str, default='train',help='fold name')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    args = parser.parse_args()
    main(args)

