import os 

file_=[883]


for i in file_:
    os.system("python " + "predict.py " + str(i))
    print(str(i) + "运行完毕" + "\n\n")
    
'''    if os.system("python " + "predict.py " + str(i))!= 0:
    
        print("错误!检查!" + "\n\n")
        break
    else:
        os.system("python " + "predict.py " + str(i))
        print(str(i) + "运行完毕" + "\n\n") '''