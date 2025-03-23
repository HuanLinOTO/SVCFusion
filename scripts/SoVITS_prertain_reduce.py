import torch

def g_model(input_path, output_path):
    model = torch.load(input_path)

    del model["model"]["emb_g.weight"]

    model["optimizer"] = None

    model["iteration"] = 0

    model["learning_rate"] = 0.0001

    torch.save(model, output_path)


def d_model(input_path, output_path):
    model = torch.load(input_path)
    # del model["model"]["emb_d.weight"]
    model["optimizer"] = None
    model["iteration"] = 0
    model["learning_rate"] = 0.0001
    torch.save(model, output_path)

if __name__ == "__main__":
    # arg
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="model.pth")
    # output_path
    parser.add_argument("--output_path", type=str, default="model_reduced.pth")

    arg = parser.parse_args()

    input_path = arg.input_path

    if input_path.startswith("g"):
        g_model(input_path, arg.output_path)
    else:
        d_model(input_path, arg.output_path)
