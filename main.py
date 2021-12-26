import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mbox
from os.path import splitext
import cv2
import numpy


def resizex(img, fixed_width=400):
    width_percent = fixed_width / float(img.shape[1])
    height_size = int((float(img.shape[0]) * float(width_percent)))
    new_image = cv2.resize(img, (fixed_width, height_size))
    return new_image


def insert_text():
    file = fd.askopenfilename(filetypes=(('jpg файлы', '*.jpg'),
                                         ('png файлы', '*.png'),
                                         ('jpeg файлы', '*.jpeg'),
                                         ('Все файлы', '*')))
    a = splitext(file)[1]
    if a == '.jpg' or a == '.jpeg' or a == '.png':
        import opencv as o
        import prdictofobrabotannimg as prd
        o.main(file)
        global greeting
        global img
        try:
            greeting.destroy()
        except Exception:
            pass
        greeting = tk.Label(root, text=prd.main(), font=('Times New Roman', 20))
        greeting.grid(row=2)

        photo = numpy.array(cv2.imread(file))
        # print(photo)
        photo = resizex(photo)
        cv2.imwrite('image.png', photo)
        canvas = tk.Canvas(root, height=600, width=900)
        img = tk.PhotoImage(file='image.png')
        image = canvas.create_image(0, 0, anchor='nw', image=img)
        canvas.grid(row=2, column=1)
    else:
        mbox.showerror("Ошибка", "Принимает только изображения")


root = tk.Tk()
root.title('EvklidPhoto')
root.geometry('800x900+200+100')
b1 = tk.Button(text="Загрузить изображение", command=insert_text)
b1.grid(row=1, sticky=tk.E)

root.mainloop()
