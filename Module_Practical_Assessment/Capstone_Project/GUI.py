from keras.models import load_model
import numpy as np
from PIL import ImageGrab
from tkinter import *
import pyttsx3
from PIL import ImageTk, Image
from tkinter import ttk
import os
from pygame import mixer
from gtts import gTTS
from datetime import datetime

e_model = load_model('mnist.h5')
h_model = load_model('hindi1.h5')

def e_predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = e_model.predict([img])[0]
    return np.argmax(res), max(res)


def h_predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((32, 32))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 32, 32, 1)
    img = img / 255.0
    # predicting the class
    res = h_model.predict([img])[0]
    return np.argmax(res), max(res)

# function to convert text to speech in Hindi
def speak(digit):
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice" + date_string + ".mp3"
    language = "hi"
    speech = gTTS(text=digit, lang=language, slow=False)
    speech.save(filename)
    # Generate an MP3 file of the spoken Hindi text
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

    while mixer.music.get_busy():
        pass

    mixer.music.stop()
    mixer.quit()

    # Delete the audio file
    os.remove(filename)



window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()
# window.geometry("1120x650")

canv = Canvas(window, width=1120, height=650, bg='white')
canv.grid(row=2, column=3)

img = ImageTk.PhotoImage(Image.open("material.jpg"))  # PIL solution
canv.create_image(0, 0, anchor=NW, image=img)


def MyProject():
    global l1

    widget = cv
    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    m1 = mycombo.get()

    if m1 == 'English':
        # Image is captured from canvas and is resized to (28 X 28) px
        img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

        # Converting rgb to grayscale image
        img = img.convert('L')

        # Extracting pixel matrix of image and converting it to a vector of (1, 784)
        x = np.asarray(img)

        # Calling function for prediction
        pred = e_predict_digit(img)

    elif m1 == 'Hindi':

        img = ImageGrab.grab().crop((x, y, x1, y1)).resize((32, 32))

        # Converting rgb to grayscale image
        img = img.convert('L')

        # Extracting pixel matrix of image and converting it to a vector of (1, 784)
        x = np.asarray(img)

        # Calling function for prediction
        pred = h_predict_digit(img)

    # Displaying the result
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Algerian', 20))
    l1.place(x=520, y=450)
    if m1 == "Hindi":
        digit = str(pred[0])
        speak(digit)

    else:
        txt = str(pred[0])
        spc = pyttsx3.init()
        spc.say(txt)
        spc.runAndWait()


lastx, lasty = None, None


# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()


# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Label
L1 = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="black")
L1.place(x=300, y=10)

# Button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=400, y=370)

# Button to predict digit drawn on canvas
b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=600, y=370)

# Button to predict digit drawn on canvas
mycombo = ttk.Combobox(window, justify='center', font='arial', width=25, state='readonly')
mycombo['value'] = ('English', 'Hindi')
mycombo.set('English')
mycombo.place(x=450, y=500)

# Setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=400, y=70)

cv.bind('<Button-1>', event_activation)

window.mainloop()
