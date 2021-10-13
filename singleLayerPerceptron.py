import numpy as np

class Perceptron:
    def __init__(self, learningRate = 0.01, epoch = 20, activation = 'STEP'):
        self.learningRate = learningRate
        self.epoch = epoch
        print("Learning Rate: ", self.learningRate)
        print("Epoch: ", self.epoch)
        self.weightList = []
        self.bias = 0

        self.weightLog = None
        self.biasLog = None
        if activation == 'STEP':
            self.activation = self.stepActivation
            print('Step activation selected')
        if activation == 'SIGMOID':
            self.activation = self.sigmoidActivation
            print('Sigmoid activation selected')
        
    #유닛 스텝 함수를 이용한 Activation    
    def stepActivation(self, inputVector, threshold=0):
        return np.where(inputVector >= threshold, 1, 0)
    
    #시그모이드 함수를 이용한 Activation
    def sigmoidActivation(self, inputVector):
        return 1 / (1 + np.exp(-inputVector))
    
    #내적 계산
    def linearDotProduct(self, inputVector):
        return np.dot(inputVector, self.weightList) + self.bias
    
    def fitting(self, inputVector, answerVector):
        rows, columns = inputVector.shape
        #가중치와 Bias, 로그 초기화
        self.weightList = np.zeros(columns)
        self.weightLog = np.zeros((self.epoch, columns))
        self.biasLog = np.zeros(self.epoch)

        #이 부분은 첨부된 원노트 참고
        for n in range(self.epoch):
            for index, sample in enumerate(inputVector):
                linearOutput = self.linearDotProduct(sample)
                predictedOutput = self.activation(linearOutput)
                updateWeight = self.learningRate  * (answerVector[index] - predictedOutput)
                self.weightList += sample * updateWeight
                self.bias += updateWeight

            #로그 기록
            self.weightLog[n] = self.weightList
            self.biasLog[n] = self.bias

    #inputVector에 대응하는 y값 예측
    def prediction(self, inputVector):
        linearOutput = self.linearDotProduct(inputVector)
        activatedOutput = self.activation(linearOutput)

        return activatedOutput
    
    #정확도 측정
    def evaluation(self, answerVector, predictionVector):
        return np.sum(answerVector == predictionVector) / len(predictionVector)