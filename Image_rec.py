from imageai.Classification import ImageClassification
import os
 
exec_path = os.getcwd()
 
prediction = ImageClassification()

# Set up model
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()
 
 #Set up prediction 
predictions, probabilities = prediction.classifyImage(os.path.join(exec_path,'Godzilla.jpg'), result_count=5)
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')