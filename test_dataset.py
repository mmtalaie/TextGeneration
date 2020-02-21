import pickle
import  numpy

target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
for item in target_params:
    item = numpy.array(item)
    print('1')
    print(item.shape)
    if item.shape == (5000,):
        print(item)