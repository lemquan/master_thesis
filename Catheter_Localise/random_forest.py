from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from lasange_test import load_data
from util import plot_images,simple_plot
import timeit
import cPickle

# load the feature maps from the network
# feature maps of the validation data set taken at the last convolutional layer
f_out = np.load('feature_maps_imagenet.npz', 'r')
feature_maps = f_out['arr_0']
# simple_plot(feature_maps[1][:6241], (79,79))



# load teh feature maps from the test set
f_out = np.load('fmaps_test_imagenet.npz', 'r') # the feature map of the test set- the input to fit the DT
test_fmaps = f_out['arr_0']
test_fmaps = np.reshape(test_fmaps, (test_fmaps.shape[0], 6272))

# load the data
pixel = (96,96)
path='/Users/quale/Dropbox/Project/Code/Catheter_Localise/catheter_images_96.npz'
#path='/home/quanle/Project_2015/Catheter_Localise/catheter_images_96.npz'
datasets = load_data(path, pixel)
X_val, y_val = datasets[1] # feature map is from validation set
X_test, y_test = datasets[2]

# fit the Decision Tree (SVM does not output the required outputs, and RandomForest took way too long to run)
print 'Training the Decision Tree Classifier'
start_time = timeit.default_timer()
clf = DecisionTreeClassifier(max_features='log2') # sqrt works the same
#clf = RandomForestClassifier(max_features='auto')
clf = clf.fit(feature_maps, y_val)

# save hte classifier
# with open('decision_tree_clf.pkl', 'wb') as fid:
#     cPickle.dump(clf, fid)

print 'Predictions.....'
all_preds = clf.predict(np.reshape(test_fmaps, (test_fmaps.shape[0], 6272)))
# all_preds = []
# for i in xrange(test_fmaps.shape[0]):
#     print 'working on test image', i
#     inpt = np.reshape(test_fmaps[i], (1, 6272))
#     pred = clf.predict(inpt)
#     all_preds.append(pred)
np.savez('dt_preds_log2.npz', np.asarray(all_preds)) # save prediction from test set
#
#
# # reload test predictions
# f = np.load('rd_preds_log2.npz')
# p = f['arr_0']
#
# # plot the outputs
# for i in xrange(p.shape[0]):
#     plot_images(X_test[i], y_test[i], p[i], i, pixels=(96,96))

# Check for accuracy
p = all_preds
yhats = np.reshape(p,(p.shape[0], pixel[0]*pixel[1]))
mse = np.sum(np.sum((y_test - yhats)**2, axis=1), axis=0) / y_test.shape[0]
end_time = timeit.default_timer()
print mse
print 'time to complete:', end_time-start_time








