import csv
import os
import sys

def main():
    assert len(sys.argv) == 3, "Specify infile and outfile"
    with open(sys.argv[1], "r") as f:
        with open(sys.argv[2], "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t')
            curr_line = f.readline()
            while curr_line:
                eval_acc_index = curr_line.find('\"eval_accuracy\": ')
                if eval_acc_index != -1:
                    accuracy = curr_line[eval_acc_index + len('\"eval_accuracy\": '):-2]
                    f1_line = f.readline()
                    f1_index = f1_line.find('\"eval_f1\": ')
                    f1_score = f1_line[f1_index + len('\"eval_f1\": '):-2]
                    loss_line = f.readline()
                    loss_label = '\"eval_loss\": '
                    loss_index = loss_line.find(loss_label)
                    loss_score = loss_line[loss_index + len(loss_label):-2]
                    for i in range(4):
                        step_line = f.readline()
                        step_label = '\"step\": '
                        step_index = step_line.find(step_label)
                        step_num = step_line[step_index + len(step_label):-1]
                    csvwriter.writerow([loss_score, accuracy, f1_score, step_num])
                curr_line = f.readline()

if __name__ == "__main__":
    main()
