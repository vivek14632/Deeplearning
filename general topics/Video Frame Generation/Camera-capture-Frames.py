
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os
cap = cv2.VideoCapture(0)

try:
    if not os.path.exists('cameraFrames'):
        os.makedirs('cameraFrames')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0    
while(True):

    ret, frame = cap.read()

    cv2.imshow('cameraCapture',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break;
    else:
        name = './cameraFrames/frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name, frame)
        currentFrame += 1
print(str(currentFrame) +' image frames created.');        
cap.release()
cv2.destroyAllWindows()

