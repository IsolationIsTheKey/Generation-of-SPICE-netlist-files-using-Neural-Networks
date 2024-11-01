Import os
from torch.utils.tensorboard import SummaryWriter

yolov5_dir = "C:/Users/Username/PycharmProjects/Yolov5Train/yolov5"
yaml_path = "E:/DatasetNr2/data.yaml"

writer = SummaryWriter(log_dir=os.path.join(yolov5_dir, 'runs', 'train'))
os.system(f"python {yolov5_dir}/train.py --img 640 --batch 16 --epochs 50 --data {yaml_path} –
weights yolov5s.pt")
writer.close()
