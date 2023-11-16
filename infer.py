import torch
from torchvision import transforms
import cv2
import numpy as np
import facenet_pytorch
from Backbone import DAL_model
from dataclasses import dataclass

class Config:
    path_to_model = "/home/booleanlabs/Desktop/EE_GAN/weights/9_0.0000.state.pt"
    transform=transforms.Compose([transforms.ToTensor(), 
                                #   transforms.ToPILImage(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), 
                                transforms.Resize((96, 112))])
    class_id_count = 24652


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device =  'cpu'
    
mtcnn = facenet_pytorch.MTCNN(image_size=160, margin=0, min_face_size=20, 
                              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

model = DAL_model('cosface', Config.class_id_count).to(device)
model.load_state_dict(torch.load(Config.path_to_model))

model.eval()

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, _ = mtcnn.detect(frame)
        
        if boxes is not None:    
            for box in boxes.astype(np.uint16):
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2RGB)

                cv2.rectangle(frame, (box[2], box[3]), (box[0], box[1]), (0, 255, 0), 2)    
                
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                face = Config.transform(face).unsqueeze(0).to(device)
                prediction = model(face, torch.tensor([12]).to(device), torch.tensor([12]).to(device), emb=True)
                print(prediction)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
            

    
if __name__ == '__main__':
    main()