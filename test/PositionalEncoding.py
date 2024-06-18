import torch

class PositionalEncoding:
    def __init__(self, d_model, max_len, device):
        self.encoding = torch.zeros(max_len, d_model, device=device)
        print("self.encoding (initial):", self.encoding)

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        print("pos:", pos)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        print("_2i:", _2i)
        print('sin')
        # 计算位置编码的sin部分
        for i in range(0, d_model, 2):
            print(f"i = {i}")
            sin_input = pos / (10000 ** (_2i[i//2] / d_model))
            self.encoding[:, i] = torch.sin(sin_input).squeeze()
            for j in range(max_len):
                print(f"self.encoding[{j}, {i}] = sin({pos[j].item()} / (10000 ** ({_2i[i//2].item()} / {d_model}))) = {self.encoding[j, i].item()}")

        # 计算位置编码的cos部分
        print('cos')
        for i in range(1, d_model, 2):
            print(f"i = {i}")
            cos_input = pos / (10000 ** (_2i[i//2] / d_model))
            self.encoding[:, i] = torch.cos(cos_input).squeeze()
            for j in range(max_len):
                print(f"self.encoding[{j}, {i}] = cos({pos[j].item()} / (10000 ** ({_2i[i//2].item()} / {d_model}))) = {self.encoding[j, i].item()}")

    def get_encoding(self):
        return self.encoding

d_model = 8
max_len = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n')
pos_encoding = PositionalEncoding(d_model, max_len, device)

encoding = pos_encoding.get_encoding()
print("encoding:", encoding)