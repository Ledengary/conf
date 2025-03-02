import jsonlines
import argparse

def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        data = [obj for obj in reader]
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--nshots', type=int, default=10, required=True, help='Number of shots to select from GSM8K')
args = parser.parse_args()
nshots = args.nshots

gsm8k_path = '../../data/GSM8K/'
train_data = read_jsonl(gsm8k_path + 'train.jsonl')
test_data = read_jsonl(gsm8k_path + 'test.jsonl')

print('train_data:', len(train_data))
print('test_data:', len(test_data))

# Sort the training data by the total number of words in the question and full_answer
sorted_train_data = sorted(train_data, key=lambda x: len(x['question'].split()) + len(x['full_answer'].split()))

gsm8k_shots_str = ""
for i in range(nshots):
    question = sorted_train_data[i]['question']
    answer = sorted_train_data[i]['full_answer']
    print('Question:', question, '\nAnswer:', answer, '\n')
    gsm8k_shots_str += 'Question: ' + question + '\nAnswer: ' + answer + '\n\n'

# Save the shots to a file
output_path = gsm8k_path + f'{nshots}-shots.txt'
with open(output_path, 'w') as f:
    f.write(gsm8k_shots_str)
    print(f'Saved {nshots} shots to {output_path}')
print("Done!")