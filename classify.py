import cv2
from cvzone.ClassificationModule import Classifier

item = []

with open('bill.csv', 'w') as creating_new_csv_file: 
   pass 

def runprg():
    cap = cv2.VideoCapture(1)
    myClassifier = Classifier('MyModel/keras_model.h5', 'MyModel/labels.txt')
    price = [0,1000,1500,20,40]
    w = 1

    def bill(name, price):
        cnt = 0
        for i in range(0, len(item)):
            if item[i][0] == name:
                item[i][1] += 1
                item[i][2] += price
                cnt = 1
        if cnt == 0:
            item.append([name, 1, price])

    while True:
        _, img = cap.read()
        predictions, index = myClassifier.getPrediction(img, scale=1)
        name = myClassifier.list_labels[index]
        if (index == 0):
            w = 1
        elif (index != 0):
            if (w == 1):
                bill(name, price[index])
                w = 0
        cv2.imshow("Billing", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    sumprice = 0
    if(len(item)>0):
        for i in range(0, len(item)):
            sumprice += item[i][2]
    item.insert(0, ['Item', 'Quantity', 'Price'])
    item.append(['Total', '', sumprice])

    with open('bill.csv', 'r+') as f:
        myDataList = f.readline()
        for line in myDataList:
            entry = line.split(',')
        for i in range(0, len(item)):
            f.writelines(f'\n{item[i][0]},{item[i][1]},{item[i][2]}')

    
