import cv2, os, numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime
import paho.mqtt.publish as publish

# Konfigurasi
mqttHostname = "broker.hivemq.com"
mqttRootTopic = "artofkingmandaku"

eyeModel = "model/haarcascade_eye.xml"
frontalFaceModel = "model/haarcascade_frontalface_default.xml"
wajahDir = 'datawajah'
latihDir = 'latihwajah'

def selesai1():
    intructions.config(text="Rekam Data Telah Selesai!")
def selesai2():
    intructions.config(text="Training Wajah Telah Selesai!")
def selesai3():
    intructions.config(text="Sukses Keluar Parkir")
def rekamDataWajah():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier(frontalFaceModel)
    eyeDetector = cv2.CascadeClassifier(eyeModel)
    faceID = entry2.get()
    nama = entry1.get()
    nim = entry2.get()
    ambilData = 1
    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile = str(nim) +'_'+str(nama) +'_'+ str(ambilData) +'.jpg'
            cv2.imwrite(wajahDir + '/' + namaFile, frame)
            ambilData += 1
            roiabuabu = abuabu[y:y + h, x:x + w]
            roiwarna = frame[y:y + h, x:x + w]
            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)
        cv2.imshow('webcamku', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData > 30:
            break
    selesai1()
    cam.release()
    cv2.destroyAllWindows()  # untuk menghapus data yang sudah dibaca
    trainingWajah()

def trainingWajah():
    def getImageLabel(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        faceIDs = []
        for imagePath in imagePaths:
            PILimg = Image.open(imagePath).convert('L')
            imgNum = np.array(PILimg, 'uint8')
            faceID = int(os.path.split(imagePath)[-1].split('_')[0])
            faces = faceDetector.detectMultiScale(imgNum)
            for (x, y, w, h) in faces:
                faceSamples.append(imgNum[y:y + h, x:x + w])
                faceIDs.append(faceID)
            return faceSamples, faceIDs

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier(frontalFaceModel)
    faces, IDs = getImageLabel(wajahDir)
    faceRecognizer.train(faces, np.array(IDs))
    # simpan
    faceRecognizer.write(latihDir + '/training.xml')
    selesai2()

# Hapus Foto Training
def hapusVisitor(id):
    for filename in os.listdir("datawajah"):
        if(int(filename.split("_")[0]) == id): 
            try: os.remove(str(wajahDir + "/" + filename))
            except FileNotFoundError: print(filename, "notfound")
            trainingWajah()

def markAttendance(name):
    pass
    # mqttc.publish("artofkingmandaku/keluar", True)
    # with open("Attendance.csv",'r+') as f:
    #     namesDatalist = f.readlines()
    #     namelist = []
    #     yournim = entry2.get()
    #     for line in namesDatalist:
    #         entry = line.split(',')
    #         namelist.append(entry[0])
    #     if name not in namelist:
    #         now = datetime.now()
    #         dtString = now.strftime('%H:%M:%S')
    #         f.writelines(f'\n{name},{yournim},{dtString}')

def absensiWajah():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier(frontalFaceModel)
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(latihDir + '/training.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    yourname = entry1.get()
    names = []
    names.append(yourname)
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)
    sekali = False
    while not sekali :
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)
            id, confidence = faceRecognizer.predict(abuabu[y:y+h,x:x+w])
            if confidence < 70:
                id = names[0]
                confidence = "  {0}%".format(round(170 - confidence))
                publish.single("artofkingmandaku/keluar", True, hostname=mqttHostname)
                sekali=True

            elif confidence > 70:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(round(150 - confidence))

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('ABSENSI WAJAH', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
    selesai3()
    cam.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
# mengatur canvas (window tkinter)
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=8)
canvas.configure(bg="black")
# judul
judul = tk.Label(root, text="Art Of King", font=("Roboto",34),bg="#242526", fg="white")
canvas.create_window(350, 80, window=judul)
#credit
made = tk.Label(root, text="MAN 2 KUDUS", font=("Times New Roman",13), bg="black",fg="white")
canvas.create_window(360, 20, window=made)
# for entry data nama
entry1 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
label1 = tk.Label(root, text="Plat Nomor", font="Roboto", fg="white", bg="black")
canvas.create_window(90,170, window=label1)
# for entry data nim
entry2 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
label2 = tk.Label(root, text="ID", font="Roboto", fg="white", bg="black")
canvas.create_window(60, 210, window=label2)

global intructions

# tombol untuk rekam data wajah
intructions = tk.Label(root, text="Welcome", font=("Roboto",15),fg="white",bg="black")
canvas.create_window(370, 300, window=intructions)
Rekam_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Rekam_text, font="Roboto", bg="#00ff00", fg="black", height=1, width=15,command=rekamDataWajah)
Rekam_text.set("Masuk")
Rekam_btn.grid(column=0, row=7)

# tombol untuk training wajah
Rekam_text1 = tk.StringVar()
# Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto", bg="#20bebe", fg="white", height=1, width=15,command=trainingWajah)
Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto", bg="#20bebe", fg="white", height=1, width=15)
Rekam_text1.set("Training")
Rekam_btn1.grid(column=1, row=7)

# tombol absensi dengan wajah
Rekam_text2 = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Rekam_text2, font="Roboto", bg="#20bebe", fg="white", height=1, width=20, command=absensiWajah)
Rekam_text2.set("Keluar")
Rekam_btn2.grid(column=2, row=7)

root.title("ART OF KING")
root.mainloop()
