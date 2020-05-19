# Traffic Signs Recognition With 97% Accuracy Using Deep Learning

![](https://github.com/ak224001/Traffic-Signs-Recognition-deep-learning/blob/Img/trafficSigns.png?raw=true)
## Build Model
```python
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', activation='relu',input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(43, activation="softmax"))
```
## Train Model
```python
epochs = 100
history = model.fit(X_train,y_train, batch_size = 64, epochs=epochs,
		validation_data=(X_test,y_test), verbose=1)
```
```python
Epoch 1/100
491/491 [==============================] - 4s 8ms/step - loss: 2.4459 - accuracy: 0.2939 - val_loss: 0.9538 - val_accuracy: 0.6954
Epoch 2/100
491/491 [==============================] - 4s 8ms/step - loss: 0.7279 - accuracy: 0.7645 - val_loss: 0.1739 - val_accuracy: 0.9459
Epoch 3/100
491/491 [==============================] - 4s 8ms/step - loss: 0.3370 - accuracy: 0.8939 - val_loss: 0.0728 - val_accuracy: 0.9809
Epoch 4/100
491/491 [==============================] - 4s 7ms/step - loss: 0.2099 - accuracy: 0.9374 - val_loss: 0.0565 - val_accuracy: 0.9870
Epoch 5/100
491/491 [==============================] - 4s 7ms/step - loss: 0.1597 - accuracy: 0.9522 - val_loss: 0.0391 - val_accuracy: 0.9898
Epoch 6/100
491/491 [==============================] - 4s 7ms/step - loss: 0.1396 - accuracy: 0.9581 - val_loss: 0.0298 - val_accuracy: 0.9931
Epoch 7/100
491/491 [==============================] - 4s 7ms/step - loss: 0.1157 - accuracy: 0.9652 - val_loss: 0.0258 - val_accuracy: 0.9939
Epoch 8/100
491/491 [==============================] - 4s 7ms/step - loss: 0.1077 - accuracy: 0.9672 - val_loss: 0.0258 - val_accuracy: 0.9943
Epoch 9/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0940 - accuracy: 0.9719 - val_loss: 0.0225 - val_accuracy: 0.9952
Epoch 10/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0864 - accuracy: 0.9740 - val_loss: 0.0414 - val_accuracy: 0.9890
Epoch 11/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0899 - accuracy: 0.9736 - val_loss: 0.0212 - val_accuracy: 0.9950
Epoch 12/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0777 - accuracy: 0.9773 - val_loss: 0.0215 - val_accuracy: 0.9943
Epoch 13/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0663 - accuracy: 0.9807 - val_loss: 0.0259 - val_accuracy: 0.9935
Epoch 14/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0724 - accuracy: 0.9800 - val_loss: 0.0174 - val_accuracy: 0.9954
Epoch 15/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0738 - accuracy: 0.9782 - val_loss: 0.0219 - val_accuracy: 0.9944
Epoch 16/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0637 - accuracy: 0.9818 - val_loss: 0.0175 - val_accuracy: 0.9954
Epoch 17/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0565 - accuracy: 0.9834 - val_loss: 0.0144 - val_accuracy: 0.9971
Epoch 18/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0581 - accuracy: 0.9839 - val_loss: 0.0220 - val_accuracy: 0.9944
Epoch 19/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0604 - accuracy: 0.9829 - val_loss: 0.0266 - val_accuracy: 0.9936
Epoch 20/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0585 - accuracy: 0.9839 - val_loss: 0.0199 - val_accuracy: 0.9954
Epoch 21/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0540 - accuracy: 0.9849 - val_loss: 0.0147 - val_accuracy: 0.9974
Epoch 22/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0648 - accuracy: 0.9823 - val_loss: 0.0165 - val_accuracy: 0.9964
Epoch 23/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0511 - accuracy: 0.9856 - val_loss: 0.0122 - val_accuracy: 0.9973
Epoch 24/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0584 - accuracy: 0.9850 - val_loss: 0.0166 - val_accuracy: 0.9972
Epoch 25/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0495 - accuracy: 0.9863 - val_loss: 0.0173 - val_accuracy: 0.9967
Epoch 26/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0582 - accuracy: 0.9843 - val_loss: 0.0155 - val_accuracy: 0.9971
Epoch 27/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0516 - accuracy: 0.9859 - val_loss: 0.0167 - val_accuracy: 0.9962
Epoch 28/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0514 - accuracy: 0.9863 - val_loss: 0.0111 - val_accuracy: 0.9977
Epoch 29/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0452 - accuracy: 0.9876 - val_loss: 0.0135 - val_accuracy: 0.9974
Epoch 30/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0449 - accuracy: 0.9879 - val_loss: 0.0111 - val_accuracy: 0.9982
Epoch 31/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0498 - accuracy: 0.9864 - val_loss: 0.0170 - val_accuracy: 0.9971
Epoch 32/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0419 - accuracy: 0.9884 - val_loss: 0.0145 - val_accuracy: 0.9972
Epoch 33/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0417 - accuracy: 0.9892 - val_loss: 0.0117 - val_accuracy: 0.9968
Epoch 34/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0448 - accuracy: 0.9879 - val_loss: 0.0111 - val_accuracy: 0.9976
Epoch 35/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0367 - accuracy: 0.9896 - val_loss: 0.0148 - val_accuracy: 0.9968
Epoch 36/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0559 - accuracy: 0.9860 - val_loss: 0.0122 - val_accuracy: 0.9974
Epoch 37/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0568 - accuracy: 0.9856 - val_loss: 0.0160 - val_accuracy: 0.9973
Epoch 38/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0466 - accuracy: 0.9886 - val_loss: 0.0117 - val_accuracy: 0.9980
Epoch 39/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0390 - accuracy: 0.9898 - val_loss: 0.0184 - val_accuracy: 0.9966
Epoch 40/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0491 - accuracy: 0.9877 - val_loss: 0.0222 - val_accuracy: 0.9955
Epoch 41/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0484 - accuracy: 0.9876 - val_loss: 0.0154 - val_accuracy: 0.9976
Epoch 42/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0397 - accuracy: 0.9896 - val_loss: 0.0169 - val_accuracy: 0.9967
Epoch 43/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0432 - accuracy: 0.9886 - val_loss: 0.0139 - val_accuracy: 0.9974
Epoch 44/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0501 - accuracy: 0.9887 - val_loss: 0.0199 - val_accuracy: 0.9958
Epoch 45/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0459 - accuracy: 0.9886 - val_loss: 0.0147 - val_accuracy: 0.9981
Epoch 46/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0411 - accuracy: 0.9888 - val_loss: 0.0103 - val_accuracy: 0.9973
Epoch 47/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0464 - accuracy: 0.9893 - val_loss: 0.0190 - val_accuracy: 0.9971
Epoch 48/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0376 - accuracy: 0.9910 - val_loss: 0.0111 - val_accuracy: 0.9977
Epoch 49/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0493 - accuracy: 0.9876 - val_loss: 0.0210 - val_accuracy: 0.9971
Epoch 50/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0382 - accuracy: 0.9905 - val_loss: 0.0114 - val_accuracy: 0.9976
Epoch 51/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0491 - accuracy: 0.9891 - val_loss: 0.0188 - val_accuracy: 0.9980
Epoch 52/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0407 - accuracy: 0.9896 - val_loss: 0.0176 - val_accuracy: 0.9974
Epoch 53/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0406 - accuracy: 0.9900 - val_loss: 0.0152 - val_accuracy: 0.9972
Epoch 54/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0485 - accuracy: 0.9887 - val_loss: 0.0172 - val_accuracy: 0.9973
Epoch 55/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0490 - accuracy: 0.9887 - val_loss: 0.0110 - val_accuracy: 0.9985
Epoch 56/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0579 - accuracy: 0.9869 - val_loss: 0.0183 - val_accuracy: 0.9964
Epoch 57/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0386 - accuracy: 0.9903 - val_loss: 0.0124 - val_accuracy: 0.9983
Epoch 58/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0338 - accuracy: 0.9928 - val_loss: 0.0139 - val_accuracy: 0.9980
Epoch 59/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0406 - accuracy: 0.9903 - val_loss: 0.0140 - val_accuracy: 0.9976
Epoch 60/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0486 - accuracy: 0.9886 - val_loss: 0.0115 - val_accuracy: 0.9974
Epoch 61/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0460 - accuracy: 0.9895 - val_loss: 0.0116 - val_accuracy: 0.9980
Epoch 62/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0514 - accuracy: 0.9879 - val_loss: 0.0111 - val_accuracy: 0.9977
Epoch 63/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0404 - accuracy: 0.9912 - val_loss: 0.0160 - val_accuracy: 0.9978
Epoch 64/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0458 - accuracy: 0.9897 - val_loss: 0.0128 - val_accuracy: 0.9978
Epoch 65/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0389 - accuracy: 0.9910 - val_loss: 0.0126 - val_accuracy: 0.9974
Epoch 66/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0461 - accuracy: 0.9886 - val_loss: 0.0186 - val_accuracy: 0.9972
Epoch 67/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0415 - accuracy: 0.9905 - val_loss: 0.0099 - val_accuracy: 0.9978
Epoch 68/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0438 - accuracy: 0.9901 - val_loss: 0.0206 - val_accuracy: 0.9962
Epoch 69/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0400 - accuracy: 0.9904 - val_loss: 0.0201 - val_accuracy: 0.9966
Epoch 70/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0558 - accuracy: 0.9883 - val_loss: 0.0133 - val_accuracy: 0.9985
Epoch 71/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0539 - accuracy: 0.9896 - val_loss: 0.0102 - val_accuracy: 0.9982
Epoch 72/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0373 - accuracy: 0.9909 - val_loss: 0.0100 - val_accuracy: 0.9981
Epoch 73/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0485 - accuracy: 0.9899 - val_loss: 0.0126 - val_accuracy: 0.9981
Epoch 74/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0441 - accuracy: 0.9899 - val_loss: 0.0166 - val_accuracy: 0.9967
Epoch 75/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0505 - accuracy: 0.9892 - val_loss: 0.0145 - val_accuracy: 0.9974
Epoch 76/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0453 - accuracy: 0.9896 - val_loss: 0.0225 - val_accuracy: 0.9946
Epoch 77/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0406 - accuracy: 0.9905 - val_loss: 0.0126 - val_accuracy: 0.9983
Epoch 78/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0542 - accuracy: 0.9887 - val_loss: 0.0175 - val_accuracy: 0.9976
Epoch 79/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0480 - accuracy: 0.9898 - val_loss: 0.0143 - val_accuracy: 0.9980
Epoch 80/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0494 - accuracy: 0.9898 - val_loss: 0.0141 - val_accuracy: 0.9974
Epoch 81/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0430 - accuracy: 0.9904 - val_loss: 0.0120 - val_accuracy: 0.9977
Epoch 82/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0439 - accuracy: 0.9905 - val_loss: 0.0135 - val_accuracy: 0.9977
Epoch 83/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0406 - accuracy: 0.9912 - val_loss: 0.0140 - val_accuracy: 0.9980
Epoch 84/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0496 - accuracy: 0.9906 - val_loss: 0.0100 - val_accuracy: 0.9980
Epoch 85/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0329 - accuracy: 0.9919 - val_loss: 0.0158 - val_accuracy: 0.9976
Epoch 86/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0418 - accuracy: 0.9912 - val_loss: 0.0233 - val_accuracy: 0.9967
Epoch 87/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0486 - accuracy: 0.9894 - val_loss: 0.0261 - val_accuracy: 0.9971
Epoch 88/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0582 - accuracy: 0.9891 - val_loss: 0.0184 - val_accuracy: 0.9976
Epoch 89/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0429 - accuracy: 0.9914 - val_loss: 0.0159 - val_accuracy: 0.9978
Epoch 90/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0544 - accuracy: 0.9891 - val_loss: 0.0128 - val_accuracy: 0.9976
Epoch 91/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0459 - accuracy: 0.9913 - val_loss: 0.0140 - val_accuracy: 0.9982
Epoch 92/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0360 - accuracy: 0.9920 - val_loss: 0.0312 - val_accuracy: 0.9946
Epoch 93/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0496 - accuracy: 0.9909 - val_loss: 0.0174 - val_accuracy: 0.9974
Epoch 94/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0349 - accuracy: 0.9917 - val_loss: 0.0164 - val_accuracy: 0.9976
Epoch 95/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0449 - accuracy: 0.9912 - val_loss: 0.0298 - val_accuracy: 0.9938
Epoch 96/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0571 - accuracy: 0.9887 - val_loss: 0.0227 - val_accuracy: 0.9972
Epoch 97/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0439 - accuracy: 0.9912 - val_loss: 0.0116 - val_accuracy: 0.9978
Epoch 98/100
491/491 [==============================] - 4s 7ms/step - loss: 0.0498 - accuracy: 0.9901 - val_loss: 0.0172 - val_accuracy: 0.9976
Epoch 99/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0478 - accuracy: 0.9898 - val_loss: 0.0268 - val_accuracy: 0.9948
Epoch 100/100
491/491 [==============================] - 4s 8ms/step - loss: 0.0413 - accuracy: 0.9913 - val_loss: 0.0205 - val_accuracy: 0.9977
```
## Model Accurancy
![](https://github.com/ak224001/Traffic-Signs-Recognition-deep-learning/blob/Img/trafficModelAccuracy.png?raw=true)
## Model loss
![](https://github.com/ak224001/Traffic-Signs-Recognition-deep-learning/blob/Img/trafficModelLoss.png?raw=true)
## Test Accuracy
```python
accuracy_score(test_labels,predict)
```
0.9670625494853523

## Thank you
