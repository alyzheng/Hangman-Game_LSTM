import argparse

import torch
from tqdm import trange
import os

from main import MyHangman

KEY = os.getenv("KEY")

def main():
    parser = argparse.ArgumentParser(description='Parameters for Hangman game')
    parser.add_argument('--num_runs', type=int, default=20,
                        help='Number of games to run')
    parser.add_argument('--model_path', type=str, default="model.pth",
                        help='Path to the model')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=5)

    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    timeout = 2000
    print(args)
    print(KEY)
    hangman = MyHangman(access_token=KEY, timeout=timeout, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    print("Loading model...")
    hangman.model.load_state_dict(torch.load(args.model_path))

    prev_practice_runs, prev_recorded_runs, prev_recorded_successes, prev_practice_successes = hangman.my_status()

    num_runs = args.num_runs
    update_interval = 1  # adjust this value as needed

    with trange(num_runs) as t:
        for i in t:
            hangman.start_game(practice=1, verbose=args.verbose)
            if (i + 1) % update_interval == 0 or i == num_runs - 1:
                total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes = hangman.my_status()
                practice_success_rate = (total_practice_successes - prev_practice_successes) / (
                            total_practice_runs - prev_practice_runs)
                t.set_postfix(success_rate='{:.3f}'.format(practice_success_rate))

    total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes = hangman.my_status()
    practice_success_rate = (total_practice_successes - prev_practice_successes) / (
                total_practice_runs - prev_practice_runs)

    print('Run %d practice games out of an allotted 100,000. Final practice success rate = %.3f' % (
    num_runs, practice_success_rate))


if __name__ == "__main__":
    main()
