#Download data

```python
!wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
```

#UnZip

```python
!unzip ml-20m.zip
```

# import

```python
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import math 

dev = torch.device("cuda")
dev
```

# read_csv all movie

```python
movie = pd.read_csv("/content/ml-20m/movies.csv")
movie1 = movie[movie["genres"] != "(no genres listed)"]
movie1['ai_id']  = list(range(len(movie1)))
movie1.head()
```

# create model

```python
class reccom(nn.Module):
  def __init__(self):
    super().__init__()
    # embeding
    self.user = nn.Embedding(138493,32)
    self.movie = nn.Embedding(27032,32)
    self.fuly = nn.Sequential(
          nn.Linear(64,32),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(32,12),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(12,1,bias=False),
        )


  def forward(self,data):

    data_users, data_movie= data[:,0], data[:,1]
    inp_user = self.user(data_users)
    inp_mov = self.movie(data_movie)
    input_layer1 = torch.cat((inp_user,inp_mov),-1)
    output = self.fuly(input_layer1)

    return output
```

```python
new_model = reccom()
new_model.to(dev) 
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(new_model.parameters(), lr=10**(-2))
scheduler = ExponentialLR(optimizer, gamma=0.9)
```

```python
# เช็คว่า ใช้ gpu run รึเปล่า
next(new_model.parameters()).is_cuda
```

# train-ai

```python
# จำนวนบรรทัดทั้งหมด
length = 20000264

# แบ่งข้อมูลทีละ 5000 สำหรับส่งข้อมูลให้ ai
chunksize = 5000

# จำนวน iteration
length = math.ceil(length/chunksize)

epoch = 2
# tqdm context
for i in range(epoch):
    with tqdm.auto.tqdm(total=length, desc="chunks read: ") as bar:  # download bar
        # โหลด csv file มาทีละ 5,000 ส่วน(ตามจำนวน batch size)
        for i, data in enumerate(pd.read_csv("/content/ml-20m/ratings.csv", chunksize=chunksize, low_memory=False)):
            data = data.merge(movie[['movieId', 'genres']], left_on='movieId',right_on='movieId', how='left') # เช็คว่าเป็นหนังแนวไหน
            data = data[data["genres"] != "(no genres listed)"] # ไม่เอา หนังที่ "(no genres listed)"
            data = data.merge(movie1[['movieId', 'ai_id']], left_on='movieId',right_on='movieId', how='left') # ใส่ movieId ใหม่ ที่ใช้ชื่อว่า ai_id

            userId = data["userId"].tolist() # แปลง ข้อมูลใน column userId ให้เป็น list
            movieId = data["ai_id"].tolist() # แปลง ข้อมูลใน column ai_id  ให้เป็น list

            userId = np.array([userId]) -1 # แปลง ให้เป็น array แล้ว -1 เพื่อเอาไปเข้าใน embeding layer
            movieId = np.array([movieId])  # แปลง ให้เป็น array
            d = np.concatenate((userId,movieId),axis=0) # เอามารวมกัน
            d = d.T                                     # transpose
            d = torch.from_numpy(d)  # แปลงให้เป็น tensor
            y = data["rating"].tolist() # แปลง ข้อมูลใน column rating ให้เป็น list
            y = torch.tensor(y)         # แปลงให้เป็น tensor
            y,d = y.to(dev),d.to(dev) # แปลงให้ มัน run ใน gpu
            pred = new_model(d)  # ให้ ai predict ออกมา
            loss = loss_fn(pred.squeeze(), y.type(torch.float32)) # คำนวน ค่า loss
 
            loss.backward() # ∂(loss)/∂w 
            optimizer.step() # เปลี่ยน ค่า weight ใน model ให้เรียบร้อย
            optimizer.zero_grad() # set ค่า grad ให้เป็น 0 เพราะ เวลา grad ค่าจะถูกสะสมไปเรื่อยๆ 
            bar.update(1) # เพิ่มค่า download เช่น 1/0 -> 2/10  
            bar.set_description(f"loss={loss}") # แสดงค่า loss ออกมา ใน download bar
    scheduler.step() # ลดค่า learning_rate โดยการ คูณ 0.9 (ค่า gamma ย้อนกลับไปดู ตอนประกาศตัวแปร scheduler)


```

#save-model

```python
torch.save(new_model.state_dict(), "use.pth") # save model ชื่อว่า  use.pth ไว้สำหรับใช้งาน
torch.save({
            'model_state_dict': new_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "train.pth") # save เหมือนกัน แต่จะเก็บ optimizer , learning rateด้วย
print("save")
```

#ลอง load model มาtrain ใหม่

```python
checkpoint = torch.load('/content/train.pth')

new_model = reccom()
new_model.load_state_dict(checkpoint['model_state_dict'])
new_model.to(dev)
```

```python
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(new_model.parameters(), lr=10**(-2))
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # load optimizer ใหม่ และ learning rate ใหม่
scheduler = ExponentialLR(optimizer, gamma=0.9)
```

# train-ai-again

```python
# จำนวนบรรทัดทั้งหมด
length = 20000264

# แบ่งข้อมูลทีละ 5000 สำหรับส่งข้อมูลให้ ai
chunksize = 5000

# จำนวน iteration
length = math.ceil(length/chunksize)

epoch = 2
# tqdm context
for i in range(epoch):
    with tqdm.auto.tqdm(total=length, desc="chunks read: ") as bar:  # download bar
        # โหลด csv file มาทีละ 5,000 ส่วน(ตามจำนวน batch size)
        for i, data in enumerate(pd.read_csv("/content/ml-20m/ratings.csv", chunksize=chunksize, low_memory=False)):
            data = data.merge(movie[['movieId', 'genres']], left_on='movieId',right_on='movieId', how='left') # เช็คว่าเป็นหนังแนวไหน
            data = data[data["genres"] != "(no genres listed)"] # ไม่เอา หนังที่ "(no genres listed)"
            data = data.merge(movie1[['movieId', 'ai_id']], left_on='movieId',right_on='movieId', how='left') # ใส่ movieId ใหม่ ที่ใช้ชื่อว่า ai_id

            userId = data["userId"].tolist() # แปลง ข้อมูลใน column userId ให้เป็น list
            movieId = data["ai_id"].tolist() # แปลง ข้อมูลใน column ai_id  ให้เป็น list

            userId = np.array([userId]) -1 # แปลง ให้เป็น array แล้ว -1 เพื่อเอาไปเข้าใน embeding layer
            movieId = np.array([movieId])  # แปลง ให้เป็น array
            d = np.concatenate((userId,movieId),axis=0) # เอามารวมกัน
            d = d.T                                     # transpose
            d = torch.from_numpy(d)  # แปลงให้เป็น tensor
            y = data["rating"].tolist() # แปลง ข้อมูลใน column rating ให้เป็น list
            y = torch.tensor(y)         # แปลงให้เป็น tensor
            y,d = y.to(dev),d.to(dev) # แปลงให้ มัน run ใน gpu
            pred = new_model(d)  # ให้ ai predict ออกมา
            loss = loss_fn(pred.squeeze(), y.type(torch.float32)) # คำนวน ค่า loss
 
            loss.backward() # ∂(loss)/∂w 
            optimizer.step() # เปลี่ยน ค่า weight ใน model ให้เรียบร้อย
            optimizer.zero_grad() # set ค่า grad ให้เป็น 0 เพราะ เวลา grad ค่าจะถูกสะสมไปเรื่อยๆ 
            bar.update(1) # เพิ่มค่า download เช่น 1/0 -> 2/10  
            bar.set_description(f"loss={loss}") # แสดงค่า loss ออกมา ใน download bar
    scheduler.step() # ลดค่า learning_rate โดยการ คูณ 0.9 (ค่า gamma ย้อนกลับไปดู ตอนประกาศตัวแปร scheduler)
```

# save-model

```python
torch.save(new_model.state_dict(), "use.pth")
torch.save({
            'model_state_dict': new_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "train.pth")
print("save")
```

```python

```
