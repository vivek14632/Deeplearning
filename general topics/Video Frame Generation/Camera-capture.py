
# coding: utf-8

# In[4]:


import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('cameraCapture',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        cap.release()
        cv2.destroyAllWindows()
        

cap.release()
cv2.destroyAllWindows()

