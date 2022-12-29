from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import ttk
import numpy as np
from tkinter import messagebox
from tkinter import filedialog
import csv


main = Tk()
main.config(bg='#B2EBF2')
main.geometry('800x500')
main.title('Purchase Prediction')
photo = PhotoImage(file='customers.png')
main.iconphoto(False,photo)
main.resizable(False, False)

def train_model():
    dataset = pd.read_csv('D:\ML Codes\Data Set\car_data.csv')
    
    x = dataset.iloc[:,[2,3]]
    y = dataset.iloc[:, -1]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=0)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    classifier.score(x_train, y_train)
      
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn import metrics
    accu = metrics.accuracy_score(y_test, y_pred)
    Label(main, text=f'Model Trained upto : {accu*100}% Accuracy', bg='#B2EBF2').place(x=20, y = 450)
    
    def calculate():
        x_pred = sc.transform([[age.get(), salary.get()]])
        pred = classifier.predict(x_pred)
        if pred == 0:
            messagebox.showwarning('No Chances', 'Customer has No Chances to Buy!')
        else:
            messagebox.showinfo('Chances', 'Customer have Chances to Buy!')
            
    def import_customers():
        file = filedialog.askopenfilename(title='Select Dataset', filetypes=(('Csv Files', '*.csv'),('all files', '*.*')))
        data = pd.read_csv(file)
        x_features = data.iloc[:, [2,3]]
        
        x_converted = sc.transform(x_features)
        
        cust_pred = classifier.predict(x_converted)
        formatted_pred = []
        for i in cust_pred:
            if i == 0:
                formatted_pred.append('No Chances of buy')
            else:
                formatted_pred.append('Chances of buy')
        data['Purchase or Not'] = formatted_pred
        data.to_csv('D:\ML Codes\Project\Predicted_output.csv', index=False)
        
        main.destroy()
        top = Tk()
        top.geometry('1000x300')
        top.title('Purchase Prediction')
        photo = PhotoImage(file='customers.png')
        top.iconphoto(False,photo)
        
# =============================================================================
#         def getData(event):
#             selected_row = table.focus()
#             data = table.item(selected_row)
#             global row
#             row = data['values']
#             pred_x = sc.transform([[row[2], row[3]]])
#             
#             prediction = classifier.predict(pred_x)
#             
#             if prediction == 0:
#                 messagebox.showwarning('No Chances', 'Customer has no Chances of Buy')
#             else:
#                 messagebox.showinfo('Chances', 'Customer has Chances of Buy!')
# =============================================================================
        scrollbar = ttk.Scrollbar(top, orient=VERTICAL)
        table = ttk.Treeview(top, columns=(1,2,3,4,5), yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        scrollbar.config(command=table.yview)
        table['show'] = 'headings'
        #table.bind("<ButtonRelease-1>", getData)
        table.pack(fill=Y)
        csv_file = open('D:\ML Codes\Project\Predicted_output.csv', 'r')
        csv_reader = csv.reader(csv_file)
        csv_reader_list = list(csv_reader)
        
        for f in csv_reader_list:
            table.insert('', END, values=f)
            
        top.mainloop()
                
    
    Button(main, text='Check Customer', command=calculate).place(x=350, y = 260)
    
    Button(main, text='Import Customers', command=import_customers).place(x=345, y = 300)
    

Label(main, text='Customer Purchase Prediction', fg='black', bg='#B2EBF2', font=16).pack(pady=50)

main_image = PhotoImage(file='customer_predict.png')
Label(main, image=main_image, width=70, height=70, bg='#B2EBF2').place(x=360, y=80)

Label(main, text= 'Enter Customer Age : ', bg = '#B2EBF2').place(x=200, y=160)
age = StringVar()
Entry(main, textvariable=age, width=30).place(x=320, y= 162)

Label(main, text='Enter Customer Annual Income : ', bg='#B2EBF2').place(x=140, y=210)
salary = StringVar()
Entry(main, textvariable=salary, width=30).place(x=320, y=212)

train_model()


main.mainloop()