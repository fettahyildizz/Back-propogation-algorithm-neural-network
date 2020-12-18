import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BackPropogation:
    def __init__(self,n_class, learning_rate1,learning_rate2,learning_rate3, n_epoch, n_layer, n_neuron1, n_neuron2, momentum):
        self.momentum = momentum
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.learning_rate3 = learning_rate3
        self.n_class = n_class
        self.n_epoch = n_epoch
        self.n_layer = n_layer
        self.n_neuron1 = n_neuron1
        self.n_neuron2 = n_neuron2
        self.bias = np.ones(1)
        self.error_list = list()
        self.activation_func = self.sigmfunc
        self.weight0 = None
        self.weight1 = None
        self.weight2 = None

    def sigmfunc(self, x):
        return 1 / (1 + np.exp(-x))

    def sigturev(self, x):
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

    def train(self, X, yd):
        self.weight1 = np.random.uniform(-0.15, 0.15, (self.n_neuron1, 5))  # İlk ağırlıkları -0.15,0.15 arasında rastgele seçeriz, X[0].size + 1
        self.weight2 = np.random.uniform(-0.15, 0.15, (self.n_neuron2, self.n_neuron1 + 1))
        self.weight0 = np.random.uniform(-0.15, 0.15, (self.n_class, self.n_neuron2 + 1))
        self.grad0 = None
        self.grad1 = None
        self.grad2 = None
        self.old_weight1 = self.weight1
        self.old_weight2 = self.weight2
        self.old_weight0 = self.weight0
        self.older_weight1 = self.weight1
        self.older_weight2 = self.weight2
        self.older_weight0 = self.weight0

        for i in range(self.n_epoch):
            for idx, image in enumerate(X):
                self.older_weight1 = self.old_weight1
                self.older_weight2 = self.old_weight2
                self.older_weight0 = self.old_weight0
                self.old_weight1 = self.weight1
                self.old_weight2 = self.weight2
                self.old_weight0 = self.weight0


                #image_flatten = image.flatten()
                #image_flatten = int(image_flatten)

                self.image_x_biased = np.concatenate((image, self.bias))  # AĞIRLIĞA BIAS EKLENIR
                #3print(self.image_x_biased)

                #print(self.image_x_biased[0]+self.image_x_biased[1])
                self.image_x = np.transpose(np.array([self.image_x_biased]))  # x inputları düzenli hale getirilir
                self.v1 = np.matmul(self.weight1, self.image_x)
                self.y1 = self.sigmfunc(self.v1)
                self.y1_x = np.vstack(
                    (self.y1, self.bias))  # ikinci katman için birinci katman çıkışlarına bias eklenir.
                self.y1_x_t = np.transpose(self.y1_x)

                self.v2 = np.matmul(self.weight2, self.y1_x)
                self.y2 = self.sigmfunc(self.v2)
                self.y2_x = np.vstack((self.y2, self.bias))
                self.y2_x_t = np.transpose(self.y2_x)

                self.v0 = np.matmul(self.weight0, self.y2_x)
                self.y0 = self.sigmfunc(self.v0)
                ydx = np.array([yd[idx]])
                ydx = np.transpose(ydx)
                self.e = ydx - self.y0
                self.e_temp = np.array([self.e])
                self.et = np.transpose(self.e)
                self.E = (1 / 2) * np.matmul(self.et, self.e)
                self.error_list.append(float(self.E))

                self.grad0 = self.e * self.sigturev(self.v0)

                self.weight0_c = np.delete(self.weight0, -1, 1)  # ağırlığın son sütununu çıkarıp tranzpozesi alınır
                self.weight0_c_t = np.transpose(self.weight0_c)
                self.grad2 = np.dot(self.weight0_c_t, self.grad0) * self.sigturev(self.v2)

                self.weight2_c = np.delete(self.weight2, -1, 1)
                self.weight2_c_t = np.transpose(self.weight2_c)
                self.grad1 = np.dot(self.weight2_c_t, self.grad2) * self.sigturev(self.v1)

                self.grad0_np = np.array([self.grad0])
                self.grad0_np = np.transpose(self.grad0_np)
                self.grad1_np = np.array([self.grad1])
                self.grad1_np = np.transpose(self.grad1_np)
                self.grad2_np = np.array([self.grad2])
                self.grad2_np = np.transpose(self.grad2_np)

                self.y2_x_t = np.transpose(self.y2_x)
                self.y1_x_t = np.transpose(self.y1_x)
                self.image_x_t = np.array([self.image_x_biased])

                ##################AĞIRLIK GÜNCELLEME
                self.weight0 = self.weight0 + self.learning_rate3 * np.matmul(self.grad0,
                                                                             self.y2_x_t) + self.momentum * (
                                           self.old_weight0 - self.older_weight0)  # y2_x_t
                self.weight2 = self.weight2 + self.learning_rate2 * np.matmul(self.grad2,
                                                                             self.y1_x_t) + self.momentum * (
                                           self.old_weight2 - self.older_weight2)
                self.weight1 = self.weight1 + self.learning_rate1 * np.matmul(self.grad1,
                                                                             self.image_x_t) + self.momentum * (
                                           self.old_weight1 - self.older_weight1)

    def fit(self, test):
        #image_flatten = test.flatten()
        image_x_biased = np.concatenate((test, self.bias))
        image_x_t = np.array([image_x_biased])
        image_x = np.transpose(np.array([image_x_biased]))  # x inputs
        v1 = np.matmul(self.weight1, image_x)
        y1 = self.sigmfunc(v1)
        y1_x = np.vstack((y1, self.bias))  # ikinci katman için birinci katman çıkışlarına bias eklenir.
        y1_x_t = np.transpose(y1_x)

        v2 = np.matmul(self.weight2, y1_x)
        y2 = self.sigmfunc(v2)
        y2_x = np.vstack((y2, self.bias))
        y2_x_t = np.transpose(y2_x)

        v0 = np.matmul(self.weight0, y2_x)
        y0 = self.sigmfunc(v0)

        print(y0)
        index_max = np.argmax(y0)
        if index_max == 0:
            print("Iris-setosa.")
        elif index_max == 1:
            print("Iris-versicolor.")  #
        elif index_max == 2:
            print("Iris-virginica.")


with open("iris.data","r") as f:    #We implemented the data file.
    iris_data = [line.strip() for line in f]

rows, cols = (150,5)#We create a 151x5 list for the datas.
rows2, cols2 = (150,1)#We create a 151x1 list for classification.
iris_data_list = [[0]*cols]*rows
class_data_list =[[0]*cols2]*rows2
class_data_list2 =[[0]*cols2]*rows2

for i in range(len(iris_data)): #We manipulate the data list during the following lines.
    iris_data_list[i] = iris_data[i].split(",")

for j in range(len(iris_data_list)):
    class_data_list[j] = iris_data_list[j][4]
    del iris_data_list[j][4]

for k in range(len(class_data_list)):   #We manipulate the classification list.
    if class_data_list[k]=="Iris-setosa":
        class_data_list2[k] = 1,0,0
    elif class_data_list[k]=="Iris-versicolor":
        class_data_list2[k] = [0, 1, 0]
    else:
        class_data_list2[k] = [0, 0, 1]

class_data_array = np.array(class_data_list2)
iris_data_array = np.array(iris_data_list,dtype=object)

arr1 = iris_data_array[0:35]    ############    We train the network with 35 samples for each classes.
arr2 = iris_data_array[50:85]
arr3 = iris_data_array[100:135]

arr4 = class_data_array[0:35]
arr5= class_data_array[50:85]
arr6 = class_data_array[100:135]

train_data_set = np.vstack((arr1,arr2,arr3))
train_class_set = np.vstack((arr4,arr5,arr6))

train_data_set = ([list(map(float,i)) for i in train_data_set])
train_class_set = ([list(map(int,i)) for i in train_class_set])

neuralNetwork = BackPropogation(n_class = 3,learning_rate1=0.4,learning_rate2=0.5,learning_rate3=0.6, n_epoch=200, n_layer=2, n_neuron1=150, n_neuron2=60, momentum=0.7) ##We create the network here.
neuralNetwork.train(train_data_set, train_class_set)    ## We train the network with train_data_set. train_class_set is classification list.

arr7 = iris_data_array[35:50]   ## We will test the remaining 15 samples for each classes.
arr8 = iris_data_array[85:100]

arr9 = iris_data_array[135:150]

test_data_set=np.vstack((arr7,arr8,arr9))

arr10 = class_data_array[35:50]
arr11 = class_data_array[85:100]
arr12 = class_data_array[135:150]

test_class_set=np.vstack((arr10,arr11,arr12))

test_data_set = ([list(map(float,i)) for i in test_data_set])
test_class_set = ([list(map(int,i)) for i in test_class_set])

for i in test_data_set:
    neuralNetwork.fit(i)
