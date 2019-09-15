import pickle
import tensor
import layer

def save():
    """레이어와 loss 저장."""
    with open("ttt_layers.bin", 'wb') as f:
        pickle.dump(layers, f)
    with open("ttt_loss_list.bin", 'wb') as f:
        pickle.dump(loss_list, f)
    print("저장 완료")

with open('ttt_train_data.bin', 'rb') as f:
    data = pickle.load(f)
with open('ttt_train_table.bin', 'rb') as f:
    table = pickle.load(f)
with open('ttt_test_data.bin', 'rb') as f:
    data_test = pickle.load(f)
with open('ttt_test_table.bin', 'rb') as f:
    table_test = pickle.load(f)

with open('ttt_layers.bin', 'rb') as f:
    layers = pickle.load(f)

with open('ttt_loss_list.bin', 'rb') as f:
    loss_list = pickle.load(f)

learning_rate = tensor.Tensor([0.1], [1]) # 나만의 옵티마이저에 의해 초기값은 상관 없음.
while(True):
    for e in range(10):
        layers.set_train_mode(True)
        layers.forward(data)
        train_acc = layers.accuracy(table)
        loss_list.append(train_acc)
        layers.backward(table)
        
        #나만의 optimizer 방법
        learning_rate.array[0] = layers.layers[-1].loss / 3
        layers.update(learning_rate)

        layers.set_train_mode(False)
        layers.forward(data_test)
        test_acc = layers.accuracy(table_test)
        loss_list.append(test_acc)
        print("epoch :", e,"loss", layers.layers[-1].loss,"acc :", train_acc, ",test acc :", test_acc)

    save()
        
    
print("끝")
print("결과가 만족스러우면 save()를 입력하세요. 저장됩니다.")
