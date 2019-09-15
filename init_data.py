from urllib.request import urlopen
import tensor
import pickle

def init():
    array = []
    table = []
    for line in urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"):
        decoded_line = line.decode('UTF-8').lower().strip()
        for i in range(0,17,2):
            if(decoded_line[i] == 'x'):
                array.extend([1.,0., 0.])
            elif(decoded_line[i] == 'o'):
                array.extend([0.,1., 0.])
            else:
                array.extend([0.,0.,0.])
                
        if(decoded_line[18] == 'p'):
            table.extend([1,0]) #one - hot
        else:
            table.extend([0,1])
    
    data_count = len(table) // 2
    if(len(table) != len(array)):
        print("error")
    train_count = data_count * 4 // 5
    test_count = data_count - train_count
    
    table = tensor.Tensor(table, [data_count,2])
    data = tensor.Tensor(array, [data_count,27])

    train_data = tensor.create_zeros([train_count, 27])
    train_table = tensor.create_zeros([train_count, 2])

    test_data = tensor.create_zeros([test_count, 27])
    test_table = tensor.create_zeros([test_count, 2])
    
    choice_list = tensor.create_arange(0, data_count)
    tensor.set_shuffle(choice_list)
    
    train_choice = tensor.create_zeros([train_count], int)
    test_choice = tensor.create_zeros([test_count], int)
    
    tensor.copy(choice_list, 0, train_count, train_choice)
    tensor.copy(choice_list, train_count, test_count, test_choice)
    
    tensor.copy_row(data, train_choice, train_data)
    tensor.copy_row(table, train_choice, train_table)
    tensor.copy_row(data, test_choice, test_data)
    tensor.copy_row(table, test_choice, test_table)

    with open('ttt_train_data.bin', 'wb') as f:
        pickle.dump(train_data, f)
    with open('ttt_train_table.bin', 'wb') as f:
        pickle.dump(train_table, f)
    with open('ttt_test_data.bin', 'wb') as f:
        pickle.dump(test_data, f)
    with open('ttt_test_table.bin', 'wb') as f:
        pickle.dump(test_table, f)
    print("done")
