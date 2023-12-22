import numpy as np

# У пользователя сети должна быть возможность указать: 
    # 1)Количество слоев. 
    # 2)Сколько нейронов в каждом слое. 
    # 3)Функцию активации нейронов каждого слоя. 
    # 4)Должна быть функция обучения, с настройкой коэффициента скорости обучения, количества эпох.
    # В качестве задачи взять перепод чисел из двоичной СС в десятчную или XOR.

# Создаем класс Layers для слоев нейронной сети
class Layers:

    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.1   # инициализация весов слоя
        self.bias = np.random.randn(output_size) * 0.1                  # инициализация смещения слоя
        self.activation = activation                                    # инициализация функции смещения слоя
        self.last_activation = None                                     # инициализация последней активации
        self.error = None                                               # инициализация ошибки
        self.delta = None                                               # инициализация дельты


    # Метод прямого распространения
    # Выполняет прямое распространение сигнала через слой
    # и применяет функцию активации.
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation


    # Функции активации
    def _apply_activation(self, r):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        elif self.activation == 'tanh':
            return (np.e ** r - np.e ** (-r)) / (np.e ** r + np.e ** (-r))
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        return r


    # Производные функций активации
    def apply_activation_derivative(self, r):
        if self.activation == 'sigmoid':
            return r * (1 - r)
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'relu':
            return np.greater(r, 0).astype(int)
        return r


class NNetwork:


    def __init__(self, learning_rate=0.2, epochs=40000):
        self.learning_rate = learning_rate                          # Инициализация коэффициента скорости обучения
        self.epochs = epochs                                        # Инициализация количества эпох обучения
        self._layers = []                                           # Инициализация списка слоёв
        np.random.seed(1)                                           # Задание зерна для выдачи тех же рандомных чисел


    # Метод добавления слоя
    def add_layer(self, layer):
        self._layers.append(layer)


    # Метод прямого распространения
    # Выполняет прямое распространение через все слои сети.
    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X


    # Метод обратного распространения ошибки
    # Выполняет обратное распространение ошибки и обновляет веса и смещения каждого слоя.
    def backpropagation(self, X, y):
        output = self.feed_forward(X) # Вычисление выходного значения нейронной сети для входных данных X.
        # Цикл по слоям нейронной сети в обратном порядке (от последнего к первому)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i] # Получение текущего слоя.

            # Если текущий слой - последний, то вычисление ошибки слоя 
            # и дельты слоя на основе этой ошибки и 
            # производной активационной функции выходного слоя.
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)

            # Если текущий слой - не последний, то получение следующего слоя и вычисление ошибки текущего слоя 
            # на основе весов следующего слоя и дельты следующего слоя, 
            # а также вычисление дельты текущего слоя на основе ошибки 
            # текущего слоя и производной активационной функции текущего слоя.
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Цикл по слоям нейронной сети в прямом порядке (от первого к последнему)
        for i in range(len(self._layers)):
            layer = self._layers[i]         #Получение текущего слоя.
            #Если текущий слой - первый, то входными данными для обновления весов являются X,
            # иначе - последнее значение активации предыдущего слоя.
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            #Обновление весов текущего слоя на основе дельты текущего слоя,
            # входных данных и коэффициента обучения.
            layer.weights += layer.delta * o_i.T * self.learning_rate
            layer.bias += self.learning_rate * layer.delta


    # Метод обучения сети
    # Обучает сеть на заданном наборе данных.
    def train(self, X, y):
        for i in range(self.epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j])


    # Метод предсказания
    # Выполняет предсказание на основе обученной сети.
    def predict(self, X):
        return self.feed_forward(X)



def mainXOR():

    # Создание сети
    nn = NNetwork(0.2, 35000)
    nn.add_layer(Layers(2, 5, 'relu'))
    nn.add_layer(Layers(5, 5, 'sigmoid'))
    nn.add_layer(Layers(5, 2, 'sigmoid'))
    nn.add_layer(Layers(2, 1, 'relu'))

    # Обучающая выборка
    train_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([0, 1, 1, 0])

    # Обучение сети
    nn.train(train_input, target)

    # Проверка работы сети
    for item in train_input:
        print(f"Predicting XOR {item} gave {bool(np.round((nn.predict(item))))} and real is {bool(item[0] ^ item[1])}")



def mainBin():

    # Создание сети
    nn = NNetwork(0.05, 15000)
    nn.add_layer(Layers(5, 7, 'relu'))
    nn.add_layer(Layers(7, 7, 'sigmoid'))
    nn.add_layer(Layers(7, 5, 'sigmoid'))
    nn.add_layer(Layers(5, 1, 'relu'))



    # Тренировочные данные
    inputs = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1]
    ])


    max_number = 31


    targets = np.array([
    [0/max_number],
    [1/max_number],
    [2/max_number],
    [4/max_number],
    [8/max_number],
    [16/max_number],
    [10/max_number],
    [18/max_number],
    [6/max_number],
    [31/max_number]
    ])


    # Обучение сети
    nn.train(inputs, targets)


    # Проверка работы сети
    for item in inputs:
        print(f"Predicting bin {item} to dec gave {np.round(nn.predict(item) * max_number)}")
    print("Null", max_number * nn.predict([[0, 0, 0, 0, 0]]))
    print("Three", max_number * nn.predict([[0, 0, 0, 1, 1]]))
    print("Seven", max_number * nn.predict([[0, 0, 1, 1, 1]]))
    print("Fifteen", max_number * nn.predict([[0, 1, 1, 1, 1]]))



if __name__ == "__main__":
    mainXOR()
    mainBin()