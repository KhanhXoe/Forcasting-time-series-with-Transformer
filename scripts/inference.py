import torch

def inference(model, data, input_len, pred_len, device):
    model.eval()
    x = torch.tensor(data[-input_len:], dtype=torch.float32).unsqueeze(0).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(pred_len):
            out = model(x)
            next_pred = out[:, -1].item()
            preds.append(next_pred)
            x = torch.cat([x[:, 1:], out[:, -1:]], dim=1)
    return preds
