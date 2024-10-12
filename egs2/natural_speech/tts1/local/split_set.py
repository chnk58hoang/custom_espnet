import os
import argparse
import random


def shuffle_files(file_paths):
    lines = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines.append(f.readlines())

    indices = list(range(len(lines[0])))
    random.shuffle(indices)
    shuffled_lines = [[lines[i][j] for j in indices] for i in range(len(lines))]
    for i, path in enumerate(file_paths):
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(shuffled_lines[i])


def split_file(input_folder, filename, folder1, folder2, folder3):
    file_path = os.path.join(input_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total_lines = len(lines)
    part1_size = int(total_lines * 0.90)
    part2_size = int(total_lines * 0.05)

    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    os.makedirs(folder3, exist_ok=True)

    with open(os.path.join(folder1, filename), 'w', encoding='utf-8') as f1:
        f1.writelines(lines[:part1_size])

    with open(os.path.join(folder2, filename), 'w', encoding='utf-8') as f2:
        f2.writelines(lines[part1_size:part1_size + part2_size])

    with open(os.path.join(folder3, filename), 'w', encoding='utf-8') as f3:
        f3.writelines(lines[part1_size + part2_size:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--valid_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    args = parser.parse_args()
    files = ['text', 'utt2spk', 'wav.scp']
    filepaths = [os.path.join(args.i, file) for file in files]
    shuffle_files(filepaths)
    for file in files:
        split_file(args.i, file, args.train_dir, args.valid_dir, args.test_dir)
