import datasets
import argparse


parser = argparse.ArgumentParser(description='Split Dataset')
parser.add_argument("--data_file", type=str, help = "The path to the CSV file", required=True)
parser.add_argument("--trainvalid_test_split", type=float, help = "The train+validation and test split", default=.80)
parser.add_argument("--train_valid_split", type=float, help = "The train valid split for the train+validation split", default=.80)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--write_to_disk", type=str, required=True)
args = parser.parse_args()

trainvalid_test_ds = datasets.load_dataset('csv', data_files=args.data_file).shuffle(args.seed)["train"].train_test_split(test_size=1-args.trainvalid_test_split)
train_valid_test_ds = trainvalid_test_ds["train"].train_test_split(test_size=1-args.train_valid_split)

train_ds = train_valid_test_ds["train"]
valid_ds = train_valid_test_ds["test"]
test_ds = trainvalid_test_ds["test"]

#import ipdb; ipdb.set_trace()

train_ds.to_csv(args.write_to_disk + "/train.csv", index_label=False, index=False)
valid_ds.to_csv(args.write_to_disk + "/valid.csv", index_label=False, index=False)
test_ds.to_csv(args.write_to_disk + "/test.csv", index_label=False, index=False)

print ("Done")
