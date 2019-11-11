import pandas as pd

def normalized():
    data_headers = ['input1', 'input2', 'input3', 'output']

    dataset = pd.read_csv('traffic.csv', names=data_headers)
    dataset.head()
    #print(dataset)

    dicts = dataset.to_dict()

    print("Dataset in dictionary format: \n", dicts)

    max_value, max_key = max(((v, k) for inner_d in dicts.values() for k, v in inner_d.items()))
    print("Maximum value: ", max_value)

    min_value, min_key = min(((v, k) for inner_d in dicts.values() for k, v in inner_d.items()))
    print("Minimum value: ", min_value)

    normalized_values = {}
    normalized_values['input1'] = {}
    normalized_values['input2'] = {}
    normalized_values['input3'] = {}
    normalized_values['output'] = {}

    def normalize(dicts, key):
        for k, v in dicts.items():
            if k == key:
                x = v
                print(x)
                for i in x.items():
                    new_val = (i[1] - min_value) / (max_value - min_value)
                    normalized_values[key][i[0]] = new_val
                # print(normalized_values)
                return v
            elif isinstance(v, dict):
                found = normalize(v, key)
                if found is not None:
                    return found

    (normalize(dicts, 'input1'))
    (normalize(dicts, 'input2'))
    (normalize(dicts, 'input3'))
    (normalize(dicts, 'output'))

    #print(normalized_values)
    return normalized_values