from DBIM_argument import parse_opt
from train import train
from eval import evaluate
from utils import read_list, load_model

import torch

args = parse_opt()

train_loader, val_loader, test_loader = read_list(args.data_path, args)

if args.train:
    train(args, train_loader, val_loader)

if args.eval:
    generative_model = load_model(model_path=args.DBIM_path, device=args.device, dtype=torch.float32)
    loss = evaluate(val_loader, generative_model, args)
    print(loss)

