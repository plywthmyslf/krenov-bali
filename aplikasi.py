import cv2, os, numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime
import paho.mqtt.publish as publish
import requests
import pathlib

# Konfigurasi
mqttHostname = "broker.hivemq.com"
mqttRootTopic = "artofkingmandaku"

eyeModel = "model/haarcascade_eye.xml"
frontalFaceModel = "model/haarcascade_frontalface_default.xml"
platModel = "model/haarcascade_russian_plate_number.xml"
wajahDir = 'datawajah'
latihDir = 'latihwajah'
webcamPlat = 5
webcamWajah = 2

def selesai1():
    intructions.config(text="Rekam Data Telah Selesai!")
def selesai2():
    intructions.config(text="Training Wajah Telah Selesai!")
def selesai3():
    intructions.config(text="Sukses Keluar Parkir")
def rekamDataWajah(nama):
    cam = cv2.VideoCapture(webcamWajah)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier(frontalFaceModel)
    eyeDetector = cv2.CascadeClassifier(eyeModel)
    ambilData = 1
    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile = str(nama) +'_'+ str(ambilData) +'.jpg'
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
    if(len(os.listdir(wajahDir))==0):
        return
    
    face_id_mapping = {}

    def getImageLabel(path):
        nonlocal face_id_mapping
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif'))]
        faceSamples = []
        faceIDs = []
        faceDetector = cv2.CascadeClassifier(frontalFaceModel)
        for imagePath in imagePaths:
            PILimg = Image.open(imagePath).convert('L')
            imgNum = np.array(PILimg, 'uint8')
            face_id_str = os.path.split(imagePath)[-1].split('_')[0]
            if face_id_str not in face_id_mapping:
                face_id_mapping[face_id_str] = len(face_id_mapping)
            faceID = face_id_mapping[face_id_str]
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
def hapusVisitor(id: str):
    wajahDir = pathlib.Path("datawajah")
    for file in wajahDir.iterdir():
        if file.name.startswith(f"{id}_"):
            try:
                file.unlink()
            except FileNotFoundError:
                print(f"{file.name} not found")

def absensiWajah(nama:str):
    cam = cv2.VideoCapture(webcamWajah)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier(frontalFaceModel)
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(latihDir + '/training.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    yourname = nama
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
                hapusVisitor(yourname)
                sekali=True

            elif confidence > 70:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(round(150 - confidence))

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('DETEKSI WAJAH', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
    selesai3()
    cam.release()
    cv2.destroyAllWindows()

# Procedure
def masuk():
    plat = scanPlat()
    rekamDataWajah(plat)
    
def keluar():
    plat = scanPlat()
    absensiWajah(plat)

def scanPlat():
    cam = cv2.VideoCapture(webcamPlat)
    cam.set(3, 640)
    cam.set(4, 420)
    min_area = 500
    plat=""
    while True:
        success, img = cam.read()

        plate_cascade = cv2.CascadeClassifier(platModel)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1,4)

        for(x, y, w , h) in plates:
            area = w*h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, "Number Plate", (x,y+5) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)

                img_roi = img[y:y+h, x:x+w]
                cv2.imshow("ROI", img_roi)
                plat = "k391"
                return plat
                
                
        cv2.imshow("Hasil Plat", img);
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
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
global intructions

# tombol untuk rekam data wajah
intructions = tk.Label(root, text="Welcome", font=("Roboto",15),fg="white",bg="black")
canvas.create_window(370, 300, window=intructions)
Rekam_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Rekam_text, font="Roboto", bg="#00ff00", fg="black", height=1, width=15,command=masuk)
Rekam_text.set("Masuk")
Rekam_btn.grid(column=0, row=7)

# # tombol untuk training wajah
# Rekam_text1 = tk.StringVar()
# # Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto", bg="#20bebe", fg="white", height=1, width=15,command=trainingWajah)
# Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto", bg="#20bebe", fg="white", height=1, width=15)
# Rekam_text1.set("Training")
# Rekam_btn1.grid(column=1, row=7)

# tombol absensi dengan wajah
Rekam_text2 = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Rekam_text2, font="Roboto", bg="#20bebe", fg="white", height=1, width=20, command=keluar)
Rekam_text2.set("Keluar")
Rekam_btn2.grid(column=2, row=7)

root.title("ART OF KING")
root.mainloop()
