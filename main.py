# This is a sample Python script.
import argparse
import itertools
import os
import random
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import torch.nn.functional as F
from torchmetrics import Accuracy
# from Transformer import CharacterPredictor
from hangman_utils import HangmanAPI
from LSTM import CharPredictor

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def top_k_categorical_accuracy(preds, target, k=5):
    _, top_k_preds = preds.topk(k, dim=-1)
    correct = top_k_preds.eq(target.view(-1, 1).expand_as(top_k_preds))
    top_k_acc_score = correct.float().mean()
    return top_k_acc_score


class CharPredictorDataset(Dataset):
    def __init__(self, input_data, target_data, char_to_idx, pad_id, max_length=30):
        self.input_data = input_data
        self.target_data = target_data
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.pad_id = pad_id

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        target_seq = self.target_data[idx]
        padding_length = self.max_length - len(input_seq)
        attention_mask = [1] * len(input_seq) + [0] * padding_length
        input_seq += [self.pad_id] * padding_length
        # target_seq += [-100] * padding_length
        input_x = torch.tensor(input_seq, dtype=torch.long)
        label = torch.tensor(target_seq, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        return input_x, label, attention_mask


class MyHangman(HangmanAPI):
    def __init__(self, access_token=None, session=None, timeout=None, embedding_dim=32, hidden_dim=128, num_layers=5, max_length=30):
        super().__init__(access_token, session, timeout)
        self.special_tokens = ["<pad>", "_"]
        self.vocab = ALPHABET + self.special_tokens
        vocab_size = len(self.vocab)
        self.char_to_idx = {c:idx for idx, c in enumerate(self.vocab)}
        self.idx_to_char = {idx:c for idx, c in enumerate(self.vocab)}
        self.pad_id = self.char_to_idx["<pad>"]
        self.mask_id = self.char_to_idx["_"]
        self.max_length = max_length

        self.model = CharPredictor(vocab_size, embedding_dim, hidden_dim, num_layers=num_layers, pad_id=self.pad_id).cuda()

    def encode_word(self, word):
        encoded_word = []
        for cha in word:
            encoded_word.append(self.char_to_idx[cha])
        return encoded_word

    def generate_all_combinations(self, n, limit=40):
        count = 0
        # Start with a list of n zeros
        for zeros in range(n, -1, -1):  # from n to 0
            for positions in itertools.combinations(range(n), n - zeros):
                if count >= limit:
                    return
                combination = [0] * n
                for position in positions:
                    combination[position] = 1
                yield combination
                count += 1

    def generate_all_combinations_reverse(self, n, limit=40):
        count = 0
        # Start with a list of n ones
        for ones in range(n, -1, -1):  # from n to 0
            for positions in itertools.combinations(range(n), n - ones):
                if count >= limit:
                    return
                combination = [1] * n
                for position in positions:
                    combination[position] = 0
                yield combination
                count += 1

    def create_training_data(self):
        input_data, target_data = [], []
        max_len = 0
        for word_idx, word in enumerate(self.full_dictionary):
            # word: apple
            # input: a _ _ l  e
            # output: p
            word = word.strip()
            encoded_word_ids = self.encode_word(word)
            leng = len(set(encoded_word_ids))
            max_len = max(leng, max_len)
            # masking
            word_id2pos = defaultdict(list)
            for idx, word_id in enumerate(encoded_word_ids):
                word_id2pos[word_id].append(idx)

            sorted_word_id2pos = sorted(word_id2pos.items(), key=lambda item: item[0], reverse=True)
            if word_idx < 10:
                print("word:", word)
                print("sorted_word_id2pos:", sorted_word_id2pos)
                print("===============================")

            all_mask_strategies = list(self.generate_all_combinations(len(sorted_word_id2pos)))
            all_mask_strategies_reverse = list(self.generate_all_combinations_reverse(len(sorted_word_id2pos)))
            # Convert lists to tuples
            all_mask_strategies = [tuple(lst) for lst in all_mask_strategies]
            all_mask_strategies_reverse = [tuple(lst) for lst in all_mask_strategies_reverse]

            # Combine and remove duplicates using set
            combined_mask_strategies = list(set(all_mask_strategies + all_mask_strategies_reverse))

            times = len(combined_mask_strategies)
            for i in range(times):
                this_mask_strategy = combined_mask_strategies[i]
                masked_word_ids = deepcopy(encoded_word_ids)
                recorded_targets = []
                for sorted_idx in range(len(sorted_word_id2pos)):
                    word_id, pos_list = sorted_word_id2pos[sorted_idx]
                    if this_mask_strategy[sorted_idx] == 0:
                        for pos in pos_list:
                            masked_word_ids[pos] = self.mask_id
                        recorded_targets.append(word_id)
                if len(recorded_targets) != 0:
                    input_data.append(masked_word_ids.copy())
                    target_data.append(random.sample(recorded_targets, 1)[0])
        print("input_data:", input_data[:100])
        print("target_data:", target_data[:100])
        print("max len:", max_len)
        return input_data, target_data

    # def create_training_data(self):
    #     input_data, target_data = [], []
    #     for word in self.full_dictionary:
    #         # word: apple
    #         # input: a _ _ l  e
    #         # output: p
    #         word = word.strip()
    #         encoded_word_ids = self.encode_word(word)
    #         # masking
    #         word_id2pos = defaultdict(list)
    #         for idx, word_id in enumerate(encoded_word_ids):
    #             word_id2pos[word_id].append(idx)
    #
    #         for word_id, pos_list in word_id2pos.items():
    #             masked_word_ids = deepcopy(encoded_word_ids)
    #             for pos in pos_list:
    #                 masked_word_ids[pos] = self.mask_id
    #             input_data.append(masked_word_ids)
    #             target_data.append(word_id)
    #
    #             char_lis = list(word_id2pos.keys())
    #             times = 0
    #             seen = [word_id]
    #             new_masked_word = deepcopy(masked_word_ids)
    #             while times < len(char_lis):
    #                 j = random.randint(0, len(char_lis) - 1)
    #                 times += 1
    #                 if char_lis[j] in seen:
    #                     continue
    #                 seen.append(char_lis[j])
    #                 for pos in word_id2pos[char_lis[j]]:
    #                     new_masked_word[pos] = self.mask_id
    #                 input_data.append(new_masked_word.copy())
    #                 target_data.append(word_id)
    #
    #     return input_data, target_data

    def train(self, data_loader, eval_dataloader, loss_fn, optimizer, num_epochs):
        acc_scorer = Accuracy(num_classes=len(self.vocab), top_k=6)
        k = 6
        all_eval_loss, eval_acc, eval_topk_acc = [], [], []

        for epoch in trange(num_epochs):
            all_loss, train_acc, train_topk_acc = [], [], []
            self.model.train()
            with tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]') as t:
                for input_batch, target_batch, attention_mask in t:
                    # Convert input and target sequence to tensors
                    input_tensor = input_batch.cuda()
                    target_tensor = target_batch.cuda()
                    attention_mask = attention_mask.cuda()
                    # Forward pass
                    output = self.model(input_tensor)
                    out_dim = output.shape[-1]
                    loss = loss_fn(output.view(-1, out_dim), target_tensor.view(-1))
                    preds = torch.argmax(output, dim=-1)
                    acc_score = (preds == target_tensor).float().mean()
                    topk_acc_score = acc_scorer(output.cpu(), target_tensor.cpu())

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    all_loss.append(loss.item())
                    train_acc.append(acc_score.item())
                    train_topk_acc.append(topk_acc_score.item())
                    if len(all_eval_loss) == 0:
                        t.set_postfix(train_loss=sum(all_loss)/len(all_loss), train_acc=sum(train_acc)/len(train_acc),
                                      train_topk_acc=sum(train_topk_acc)/len(train_topk_acc),
                                      eval_loss=0.0, eval_acc=0.0, eval_topk_acc=0.0)
                    else:
                        t.set_postfix(train_loss=sum(all_loss)/len(all_loss), train_acc=sum(train_acc)/len(train_acc),
                                      train_topk_acc=sum(train_topk_acc)/len(train_topk_acc),
                                      eval_loss=sum(all_eval_loss)/len(all_eval_loss),
                                      eval_acc=sum(eval_acc)/len(eval_acc),
                                      eval_topk_acc=sum(eval_topk_acc)/len(eval_topk_acc))

            all_eval_loss, eval_acc, eval_topk_acc = [], [], []
            self.model.eval()
            with torch.no_grad():
                for input_batch, target_batch, attention_mask in eval_dataloader:
                    input_tensor = input_batch.cuda()
                    target_tensor = target_batch.cuda()
                    attention_mask = attention_mask.cuda()
                    # Forward pass
                    output = self.model(input_tensor)
                    out_dim = output.shape[-1]
                    loss = loss_fn(output.view(-1, out_dim), target_tensor.view(-1))
                    preds = torch.argmax(output, dim=-1)
                    acc_score = (preds == target_tensor).float().mean()
                    topk_acc_score = acc_scorer(output.cpu(), target_tensor.cpu())
                    all_eval_loss.append(loss.item())
                    eval_acc.append(acc_score.item())
                    eval_topk_acc.append(topk_acc_score.item())

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {sum(all_loss) / len(all_loss)}, Average Train Acc: {sum(train_acc) / len(train_acc)}, Average Train Top-{k} Acc: {sum(train_topk_acc) / len(train_topk_acc)}")
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Average Eval Loss: {sum(all_eval_loss) / len(all_eval_loss)}, Average Eval Acc: {sum(eval_acc) / len(eval_acc)}, Average Eval Top-{k} Acc: {sum(eval_topk_acc) / len(eval_topk_acc)}")

    def select_index(self, probs):
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)
        for idx in sorted_indices:
            guess_letter = self.idx_to_char[int(idx)]
            if guess_letter not in self.guessed_letters and guess_letter not in self.special_tokens:
                return guess_letter

    def guess(self, word):
        clean_word = word[::2]

        input_ids = self.encode_word(clean_word)
        input_ids = input_ids + (self.max_length - len(input_ids)) * [self.pad_id]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
        output = self.model(input_tensor)
        output = output.squeeze(0)
        probs = torch.nn.functional.softmax(output, dim=-1)
        guess_letter = self.select_index(probs)
        return guess_letter

    def start_my_game(self):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []

        while True:
            word = input("Type a word to start game: e.g, a p p l _\n")
            if word == "exit":
                return

            if word == "new":
                self.guessed_letters = []
                word = input("Type a word to start game: e.g, a p p l _\n")
            # get guessed letter from user code
            guess_letter = self.guess(word)

            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)

            print("Guessing letter: {0}".format(guess_letter))


def main():
    parser = argparse.ArgumentParser(description='Parameters for Hangman game')
    parser.add_argument('--model_path', type=str, default="model.pth",
                        help='Path to the model')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--bsz', type=int, default=10240)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--do_train', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    batch_size = args.bsz
    num_epoch = args.num_epoch
    lr = args.lr
    num_layers = args.num_layers

    hangman = MyHangman(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    files = os.listdir(".")
    if "model.pth" in files and not args.do_train:
        print("Loading model...")
        hangman.model.load_state_dict(torch.load(args.model_path))
    else:
        print("Creating training data...")
        input_data, target_data = hangman.create_training_data()
        # Create a Dataset
        train_x, test_x, train_y, test_y = train_test_split(input_data, target_data, test_size=0.1, random_state=42)
        train_dataset = CharPredictorDataset(train_x, train_y, hangman.char_to_idx, hangman.pad_id)
        eval_dataset = CharPredictorDataset(test_x, test_y, hangman.char_to_idx, hangman.pad_id)
        # Create a DataLoader
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        print("Len input data:", len(input_data))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(hangman.model.parameters(), lr=lr)

        print("Start training...")
        hangman.train(data_loader, eval_dataloader, loss_fn, optimizer, num_epochs=num_epoch)

        print("Saving model...")
        torch.save(hangman.model.state_dict(), args.model_path)

    print("Testing...")
    hangman.model.eval()
    hangman.start_my_game()


if __name__ == "__main__":
    main()

