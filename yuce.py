import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import csv
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, extra_info_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size + extra_info_size, output_size)

    def forward(self, x, extra_info, role_id, lengths):
        # 使用 pack_padded_sequence 来打包输入序列
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        out, _ = self.lstm(x, (h0, c0))

        # 使用 pad_packed_sequence 来解包输出序列
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.dropout(out[:, -1, :])
        out = torch.cat((out, extra_info), dim=1)
        out = self.fc(out).squeeze()
        out = torch.relu(out)  # 添加ReLU激活函数
        return role_id, out

class MyDataset(Dataset):
    def __init__(self, result, n_min, n_max):
        self.dataset = self.create_dataset(result, n_min, n_max)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def create_dataset(self, result, n_min, n_max):
        dataset = []
        for role_data in result:
            role_id = role_data[0]
            data = role_data[1:]
            for n in range(n_min, n_max+1):
                if len(data) < n + 1:
                    continue
                for i in range(len(data) - n):
                    x = data[i:i+n]
                    y = data[i+n][0]  # 金额是第一个特征
                    extra_info = data[i+n][1:]  # 去除金额的其他特征
                    dataset.append((role_id, x, extra_info, y))
        return dataset

# 数据预处理
def preprocess_data(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 按照'role_id'和'date'进行分组
    grouped = df.groupby(['role_id', 'date'])

    # 初始化结果字典
    result_dict = {}

    # 对于每个分组
    for name, group in grouped:
        # 获取'role_id'
        role_id = name[0]
        # 获取'date'
        date = name[1]
        # 获取该分组的'pay'、'total_times'等参数，并将其转换为列表
        data = group[['pay', 'total_times', 'use_t4', 'use_t1', 'use_t2', 'use_t3', 'level', 'remain_t1', 'remain_t2', 'remain_t4']].values.tolist()[0]
        # 将数据添加到结果字典中
        if role_id not in result_dict:
            result_dict[role_id] = [data]
        else:
            result_dict[role_id].append(data)

    # 将结果字典转换为列表
    result = [[role_id] + data for role_id, data in result_dict.items()]
    return result

def collate_fn(batch):
    # 对batch中的数据按照x的长度进行排序
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    # 分离role_id、x、extra_info和y
    role_ids, xs, extra_infos, ys = zip(*batch)

    # 获取x的长度
    lengths = [len(x) for x in xs]

    # 获取batch_size和max_length
    batch_size = len(xs)
    max_length = max(lengths)

    # 初始化x_tensor和extra_info_tensor
    x_tensor = torch.zeros((batch_size, max_length, len(xs[0][0])))
    extra_info_tensor = torch.zeros((batch_size, len(extra_infos[0])))

    # 将x和extra_info转换为tensor，并进行填充
    for i, (x, extra_info) in enumerate(zip(xs, extra_infos)):
        x_tensor[i, :lengths[i]] = torch.tensor(x)
        x_tensor[i, lengths[i]:] = 0  # 填充部分设置为0
        extra_info_tensor[i] = torch.tensor(extra_info)

    # 将y转换为tensor
    y_tensor = torch.tensor(ys)

    return role_ids, x_tensor, extra_info_tensor, y_tensor, lengths

def split_train_test(data):
    # 划分训练集和测试集，测试集大小为原数据的20%
    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data

# 训练模型
def train_model(model, dataloader, test_dataloader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for role_id, x, extra_info, y, lengths in dataloader:
                x = x.to(device)
                extra_info = extra_info.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                _, output = model(x, extra_info, role_id, lengths)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # 在每个epoch结束后，调用预测函数
        test_loss = predict(model, test_dataloader)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss}")

        # 如果测试损失更低，保存模型权重
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')

def predict(model, dataloader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for role_id, x, extra_info, y, lengths in dataloader:
            x = x.to(device)
            extra_info = extra_info.to(device)
            y = y.to(device)

            _, output = model(x, extra_info, role_id, lengths)
            loss = criterion(output, y)
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader.dataset)

def save_predictions_to_csv(model, dataloader, filepath):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, x, extra_info, y, lengths in dataloader:
            x = x.to(device)
            extra_info = extra_info.to(device)
            role_id, output = model(x, extra_info, _, lengths)
            predictions.append((role_id, output.item()))
    
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['role_id', 'prediction'])
        writer.writerows(predictions)

# 主函数
if __name__ == '__main__':
    batch_size = 8
    input_size = 10
    hidden_size = 128
    num_layers = 2
    output_size = 1
    extra_info_size = 9
    num_epochs = 15
    learning_rate = 0.00003

    result = preprocess_data('sorted_result.csv')
    train_data, val_data = split_train_test(result)
    predict_data = preprocess_data('predict_data.csv')

    train_dataset = MyDataset(train_data, 4, 5)
    val_dataset = MyDataset(val_data, 4, 5)
    predict_dataset = MyDataset(predict_data, 6, 6)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    predict_loader = DataLoader(predict_dataset, batch_size=1, collate_fn=collate_fn)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, extra_info_size).to(device)

    train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    model.load_state_dict(torch.load('best_model.pth'))

    # 使用模型预测并保存结果到CSV文件
    save_predictions_to_csv(model, predict_loader, 'predictions.csv')
