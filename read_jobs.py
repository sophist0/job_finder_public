import csv
import random
import pickle

from transformers import AutoTokenizer

def read_csv_data(fpath):
    data = []
    with open(fpath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0:
                title = row[1]
                desc = row[6]
                # drop rows with missing title or description
                if title and desc:
                    data.append((title, desc))
            i += 1
    return data

def get_datasets(data):

    # break data into training, validation, and testing sets
    n = len(data)
    n_train = int(0.7 * n)
    n_validate = int(0.15 * n)

    training = data[:n_train]
    validate = data[n_train:n_train + n_validate]
    testing = data[n_train + n_validate:]

    return training, validate, testing

def shuffle_tokenize_split(data, tokenizer):

    random.shuffle(data)

    # to batch but all the descriptions in a list and tokenize that list.
    tokenized_desc = []
    for title, desc in data:

        # max length is fixed by the standard Bert model
        desc_tokens = tokenizer(desc, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        tokenized_desc.append(desc_tokens)

    training_desc, validate_desc, testing_desc = get_datasets(tokenized_desc)
    split_desc_data = {"training": training_desc, "validate": validate_desc, "testing": testing_desc}

    training, validate, testing = get_datasets(data)
    split_data = {"training": training, "validate": validate, "testing": testing}

    return split_desc_data, split_data


def preprocess_data():

    data_path = "jobs_data/"
    fpath =  data_path + "companies_to_apply.csv"
    data = read_csv_data(fpath)

    n = len(data)
    print()
    print("-----------------------------------")
    print("Number of entries: ", n)
    print("-----------------------------------")
    print()

    # Load a pre-trained BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    split_desc_desc, split_data = shuffle_tokenize_split(data, tokenizer)

    with open(data_path + 'split_desc_data.pkl', 'wb') as file:
        pickle.dump(split_desc_desc, file)

    with open(data_path + 'split_data.pkl', 'wb') as file:
        pickle.dump(split_data, file)

#########################################################

preprocess_data()