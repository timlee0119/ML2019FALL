import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from hw3_common import load_data, hw3_dataset, Resnet18

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Required. Path to a folder with testing images.",
                        required=True, type=str)
    parser.add_argument('-o', '--output', help="Required. Path to the result .csv file",
                        required=True, type=str)
    parser.add_argument('-m', '--model', help="Required. Path to the trained model",
                        required=True, type=str)
    return parser

def write_ans(ans, ansfile):
    print("Writing answers to %s" % ansfile)
    if len(ansfile.split('/')) > 1:
        dirname = os.path.dirname(ansfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    import csv
    with open(ansfile, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        for i in range(len(ans)):
            writer.writerow([i, ans[i]])

if __name__ == '__main__':
    args = build_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_set = load_data(args.input)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5], inplace=False)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    test_dataset = hw3_dataset(test_set, transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = Resnet18()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    
    model.eval()
    results = []
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            predict = torch.max(outputs, 1)[1]
            results += predict.tolist()
    write_ans(results, args.output)
