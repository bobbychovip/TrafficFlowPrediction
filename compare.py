# -*- coding: utf-8 -*-
#!/usr/bin/env python

from matplotlib.font_manager import FontProperties
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as md
import matplotlib.ticker as mt
import pandas as pd
import datetime as dt
import data_preprocess
import input_data
import lstm
import sae
import cnn_lstm
import cnn_lstm_final

time_steps = 8 

flow, labels = input_data.create_data_sets()
x_test = flow[45152:]
labels_test = labels[45152:]

lstm_model = lstm.model
sae_model = sae.model
cnn_lstm_model = cnn_lstm.model
cnn_lstm_final_model = cnn_lstm_final.model

scaler = data_preprocess.scaler
y_test = scaler.inverse_transform(labels_test)


pred = lstm_model.predict(x_test)
lstm_pred = scaler.inverse_transform(pred)
print "LSTM MAE:", metrics.mean_absolute_error(y_test, lstm_pred)
print "LSTM RMSE:", np.sqrt(metrics.mean_squared_error(y_test, lstm_pred))

x_test = x_test.reshape((-1, time_steps*33))
pred = sae_model.predict(x_test)
sae_pred = scaler.inverse_transform(pred)
print "SAE MAE:", metrics.mean_absolute_error(y_test, sae_pred)
print "SAE RMSE:", np.sqrt(metrics.mean_squared_error(y_test, sae_pred)) 

x_test= x_test.reshape((x_test.shape[0], time_steps, 33, 1))
pred = cnn_lstm_model.predict(x_test)
cnn_lstm_pred = scaler.inverse_transform(pred)
print "CNN_LSTM MAE:", metrics.mean_absolute_error(y_test, cnn_lstm_pred)
print "CNN_LSTM RMSE:", np.sqrt(metrics.mean_squared_error(y_test, cnn_lstm_pred))

x_test= x_test.reshape((x_test.shape[0], time_steps, 33, 1))
pred = cnn_lstm_final_model.predict(x_test)
cnn_lstm_final_pred = scaler.inverse_transform(pred)
print "CNN_LSTM_Final MAE:", metrics.mean_absolute_error(y_test, cnn_lstm_final_pred)
print "CNN_LSTM_Final RMSE:", np.sqrt(metrics.mean_squared_error(y_test, cnn_lstm_final_pred))

#print(y_test[56:344, 15])
"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # for Chinese characters
#fig, ax = plt.subplots()
fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
date_1 = dt.datetime(2017, 6, 7, 0, 0, 0)
date_2 = dt.datetime(2017, 6, 8, 0, 0, 0)
delta = dt.timedelta(minutes=5)
dates = mpl.dates.drange(date_1, date_2, delta)

ax1.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
#plt.xticks(pd.date_range(date_1,date_2,freq='5min'))#时间间隔
ax2.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax2.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax3.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

#plt.ylim((0, 900))
#plt.yticks(np.linspace(0, 900, 10))
plt.sca(ax1)
plt.title(u'LCTFP')
plt.ylabel(u'车流量')
plt.ylim((0, 900))
plt.yticks(np.linspace(0, 900, 10))
plt.xticks(rotation=30)
l1, = ax1.plot(dates, y_test[56:344, 16], label=u"真实值")
l2, = ax1.plot(dates, cnn_lstm_pred[56:344, 16], color='red',label=u"LCTFP")
plt.legend(handles=[l1, l2], loc='upper right')

plt.sca(ax2)
plt.title(u'LSTMs')
plt.ylabel(u'车流量')
plt.ylim((0, 900))
plt.yticks(np.linspace(0, 900, 10))
plt.xticks(rotation=30)
l3, = ax2.plot(dates, y_test[56:344, 16], label=u"真实值")
l4, = ax2.plot(dates, lstm_pred[56:344, 16], color='red',label=u"LSTMs")
plt.legend(handles=[l3, l4], loc='upper right')

plt.sca(ax3)
plt.title(u'SAEs')
plt.ylabel(u'车流量')
plt.ylim((0, 900))
plt.yticks(np.linspace(0, 900, 10))
plt.xticks(rotation=30)
l5, = ax3.plot(dates, y_test[56:344, 16], label=u"真实值")
l6, = ax3.plot(dates, lstm_pred[56:344, 16], color='red',label=u"SAEs")
plt.legend(handles=[l5, l6], loc='upper right')

ax = plt.gca()
ax.set_xticks(np.linspace(0, 30, 11))


plt.plot(y_test[264:324][16], 'o-', color='blue', label='test')
plt.plot(cnn_lstm_pred[264:324][16], 'o-', color='red', label='predict')
plt.title('cnn_lstm')

plt.ylim((0, 900))
plt.yticks(np.linspace(0, 900, 10))

plt.xlabel(u'时间')
plt.savefig("threemodels.png", dpi=1200)
#plt.legend(handles=[l1, l2, l3, l4, l5, l6], loc='upper right')
plt.show()

np.savetxt("lstm_6_7.txt", lstm_pred[56:344, 15])
np.savetxt("sae_6_7.txt", sae_pred[56:344, 15])
np.savetxt("LCTFP_6_7.txt", cnn_lstm_pred[56:344, 15])
np.savetxt("true_6_7.txt", y_test[56:344, 15])
"""
