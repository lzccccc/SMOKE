import torch


def main():
    dp_model = torch.load("../tools/logs/model_last.pth")
    ddp_model = torch.load("../tools/logs/model_final.pth")

    dp_model['state_dict'].pop('base.fc.weight')
    dp_model['state_dict'].pop('base.fc.bias')
    l_ddp = list(ddp_model['model'].keys())

    for i, (k, v) in enumerate(dp_model["state_dict"].items()):
        ddp_v = ddp_model['model'][l_ddp[i]]
        assert v.shape == ddp_v.shape
        ddp_model['model'][l_ddp[i]] = v
        print("Fit {} into {}".format(k, l_ddp[i]))

    torch.save(ddp_model, "model_servilized.pth")


if __name__ == '__main__':
    main()
