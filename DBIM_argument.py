import argparse
import torch

def parse_opt():
    # Settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="auto", help='Computation device.')

    parser.add_argument('--data_path', type=str, default="../data/tmp", help='Computation device.')
    parser.add_argument('--max_atom_number', type=int, default=29, help='Computation device.')
    parser.add_argument('--max_atom_id', type=int, default=10, help='Computation device.')

    parser.add_argument('--batch_size', type=int, default=512, help='Computation device.')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Computation device.')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Computation device.')

    parser.add_argument('--num_layers', type=int, default=2, help='11.')
    parser.add_argument('--m_dim', type=int, default=16, help='256.')

    parser.add_argument('--T', type=int, default=1000, help='Computation device.')
    parser.add_argument('--eta', type=int, default=0, help='Computation device.')
    parser.add_argument('--epochs', type=int, default=6, help='500.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Computation device.')
    parser.add_argument('--patience', type=int, default=10, help='Computation device.')
    parser.add_argument('--atom_type_scaling', type=int, default=0.25, help='Computation device.')
    parser.add_argument('--noise_level', type=int, default=0.5, help='Computation device.')

    parser.add_argument('--train', type=bool, default=True, help='Computation device.')
    parser.add_argument('--eval', type=bool, default=True, help='Computation device.')
    parser.add_argument('--eval_with_sampling', type=bool, default=True, help='Computation device.')

    parser.add_argument('--sample_steps', type=int, default=5, help='Computation device.')

    parser.add_argument('--load_DBIM', type=bool, default=False, help='Computation device.')
    parser.add_argument('--DBIM_path', type=str, default='saved_model/DBIM_model_gamma_3.pth',
                        help='Computation device.')
    parser.add_argument('--load_PaiNN', type=bool, default=False, help='Computation device.')
    parser.add_argument('--PaiNN_path', type=str, default='saved_model/PaiNN-0525-3',
                        help='Computation device.')

    parser.add_argument('--DBIM_save_path', type=str, default='saved_model/DBIM_model_gamma_3.pth',
                        help='Computation device.')

    args, unknowns = parser.parse_known_args()

    if args.device == "auto":
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    args.dtype = torch.float32

    return args