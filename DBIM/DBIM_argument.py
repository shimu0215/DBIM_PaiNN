import argparse
import torch

def parse_opt_DBIM():
    # Settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="auto", help='Computation device.')

    parser.add_argument('--data_path', type=str, default="../data_rd/data_rd_test", help='Computation device.')
    parser.add_argument('--max_atom_number', type=int, default=29, help='Computation device.')
    parser.add_argument('--max_atom_id', type=int, default=10, help='Computation device.')

    parser.add_argument('--batch_size', type=int, default=256, help='Computation device.')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Computation device.')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Computation device.')

    parser.add_argument('--T', type=int, default=1000, help='Computation device.')
    parser.add_argument('--eta', type=int, default=1, help='Computation device.')
    parser.add_argument('--epochs', type=int, default=500, help='Computation device.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Computation device.')
    parser.add_argument('--patience', type=int, default=50, help='Computation device.')
    parser.add_argument('--atom_type_scaling', type=int, default=0.25, help='Computation device.')

    parser.add_argument('--training', type=bool, default=True, help='Computation device.')

    parser.add_argument('--sample_steps', type=int, default=20, help='Computation device.')

    args, unknowns = parser.parse_known_args()

    if args.device == "auto":
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    args.dtype = torch.float32

    return args