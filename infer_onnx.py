from PIL import Image
import onnxruntime
from torchvision import transforms
import numpy as np

if __name__ == '__main__':
    onnx_model = onnxruntime.InferenceSession("path_img", providers=["CPUExecutionProvider"])

    img = Image.open("/home/hungpham/ONNX/test_images/1.jpg")
    img = img.convert('RGB')
    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std =[0.229, 0.224, 0.225])
                                 ])
    img_tensor = transform(img).unsqueeze(0)
    img_np = img_tensor.numpy()

    feed_dict = {onnx_model.get_inputs()[0].name: img_np}
    res = onnx_model.run(None, feed_dict)

    print(res)
    print(res[0].shape)
    
