import numpy as np

def validate_input_data_types(func):
    def wrapper(*args):
        def list2array(list_input):
            return np.array(list_input)

        def array2list(array_input):
            return array_input.tolist()

        def single2array(single_input):
            return np.array([single_input])

        def array2single(array_input):
            return array_input[0]

        treated_args = []
        for arg in args:
            if not isinstance(arg, (list, np.ndarray, float, int)):
                raise ValueError("Unsupported data type. Please use list, numpy array, float, or int.")
            if isinstance(arg, list):
                # temparray = func(list2array(arg))
                treated_args.append(array2list(func(list2array(arg))))
            elif isinstance(arg, (float, int)):
                treated_args.append(array2single(func(single2array(arg))))
            else:
                treated_args.append(func(arg))
        # print(treated_args)
        if len(args)==1:
            return treated_args[0]
        else:
            return treated_args
    return wrapper

@validate_input_data_types
def example_function(data):
    data2 = data * 2.0
    return data2

# Usage examples
mydata = [2, 3, 4]
print(mydata)
print(example_function(mydata))
mydata = np.array([1, 2, 3])
print(mydata)
print(example_function(mydata))
example_function(5.5)
