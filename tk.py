from tkinter import *
import tensor
import layer
import pickle

with open('ttt_layers.bin', 'rb') as f:
    layers = pickle.load(f)
    layers.set_train_mode(False)

ttt_map = tensor.create_zeros([1,18])
다음수 = tensor.create_zeros([9, 18])
선택수 = tensor.create_zeros([1,2])
제외수 = tensor.create_zeros([9,2])
phase = 0

def click_player_vs_player(event):
    global player_sw
    event.widget.configure(background = buttons_color_player1 * (1 - player_sw) + buttons_color_player2 * player_sw)
    player_sw = (player_sw + 1) % 2
    print(event.widget.winfo_name())
    
def click_player_vs_ai(event):
    event.widget.configure(background = buttons_color_player1)
    select = int(event.widget.winfo_name())
    ttt_map.array[select * 2] = 1.0
    ttt_map.shape = [9,2]
    print(ttt_map)
    ttt_map.shape = [1,18]
    for i in range(9):
        if(ttt_map.array[i * 2] + ttt_map.array[i*2 + 1] > 0):
            for j in range(9):
                다음수.array[18 * j + i * 2] = ttt_map.array[i*2]
                다음수.array[18 * j + i * 2 + 1] = ttt_map.array[i*2 + 1]
            제외수.array[i*2 + 1] = 1.0
        else:# 빈 공간에 인공지능 수를 둠. (처음에는 8번이 이 else가 실행되고 다음에는 6번, 4,2실행 됨.)            
            다음수.array[18 * i + i * 2 + 1] = 1.0
            제외수.array[i*2 + 1] = 0.0
    dap = layers.forward(다음수)
    print(dap)
    tensor.sub(dap, 제외수, dap)
    print(dap)
    tensor.argmax(dap, 0, 선택수)
    ttt_map.array[선택수.array[1] * 2 + 1] = 1.0
    buttons[선택수.array[1]].configure(background = buttons_color_player2)
    
        
    
    

window = Tk()
window.title("삼목")

buttons = [None] * 9
buttons_height = 10
buttons_width = 20
buttons_color = "#ffffff"
buttons_color_player1 = "#0000FF"
buttons_color_player2 = "#FF0000"

player_sw = 0


#버튼 만들기
for i in range(9):
    buttons[i] = Button(window, height = buttons_height, width = buttons_width, bg = buttons_color, name = str(i))
    buttons[i].grid(row = i // 3, column = i % 3)
    buttons[i].bind('<Button-1>', click_player_vs_ai)

buttons[0].info
#window.mainloop()
