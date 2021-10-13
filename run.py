import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from singleLayerPerceptron import Perceptron

#난수 생성에 사용되는 시드(random_state 값)가 고정되어있으므로 바꾸지 않는이상 
#매번 같은 과정을 거쳐 같은 결과를 내놓을 것이다.

#클러스터 두 개(각각 200개의 점으로 구성)
datasetX, datasetY = datasets.make_blobs(n_samples=400,n_features=2,centers=2,cluster_std=1.0,random_state=18)
X_train, X_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2, random_state=135)

#퍼셉트론 모델 만들기 + 예측
pLayer = Perceptron(float(input("Learning Rate: ")), int(input("Epoch: ")),input('Activation Function: '))
pLayer.fitting(X_train, y_train)
predictions = pLayer.prediction(X_test)

#정확도 출력
print("\nPerceptron classification accuracy", pLayer.evaluation(y_test, predictions), '\n')

#===================================
#여기부턴 그래프 출력 코드 걍 단순작업
#===================================
minimumX = np.amin(X_train[:,0])
maximumX = np.amax(X_train[:,0])
endpointMin = (-pLayer.weightList[0] * minimumX - pLayer.bias) / pLayer.weightList[1]
endpointMax = (-pLayer.weightList[0] * maximumX - pLayer.bias) / pLayer.weightList[1]
ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
t= np.arange(pLayer.epoch)
classes = ['Purple', 'Yellow']
print("PREDICTION\n============")
while True:
    fig = plt.figure()
    fig1 = fig.add_subplot(1,2,1)
    fig1.plot([minimumX, maximumX],[endpointMin, endpointMax], 'k', color='g', label='Model')
    try:
        xIn = float(input("X input: "))
        yIn = float(input("Y input: "))
    except:
        print('Error: Wrong Input')
        break
    fig1.plot(xIn, yIn, color='r', marker='o', label='User Input')
    plt.scatter(X_train[:,0], X_train[:,1],marker='.',c=y_train)
    fig1.set_ylim([ymin - 5, ymax + 5])
    plt.legend()
    plt.title("Perceptron Classifier")
    
    fig2 = fig.add_subplot(1,2,2)
    fig2.plot(t, pLayer.weightLog[:, 0], linestyle=':', color='b', label="X Weight")
    fig2.plot(t, pLayer.weightLog[:, 1], linestyle=':', color='g', label="Y Weight")
    fig2.plot(t, pLayer.biasLog, linestyle=':', color='r', label="Bias Log")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Weight Change Log")

    print(classes[round(pLayer.prediction([xIn, yIn]))], ' Class')
    plt.pause(0.5)
    input('Press Enter to pass')
    plt.close()
    