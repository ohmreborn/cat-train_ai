import torch 
from model import reccom
device = torch.device('cpu')
new_model = reccom()
new_model.load_state_dict(torch.load("./use.pth",map_location=device))
torch.save(new_model.state_dict(),"./use2.pth")
print("save")
