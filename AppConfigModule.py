class AppConfig():

    # haarcascade path
    face_haarcascade_path = 'haarcascadeClassifier\haarcascade_frontalface_default.xml'
    eye_haarcascade_path = ''

    # detect face paramater
    scaleFactor = 1.2
    minNeighbors = 5
    minSize = (50, 50)

    # similarity compare
    compare_similarity_threshold = 0.9
    limit_similarity_width = 100
    limit_similarity_height = 100

    # save image
    save_image_format = 'jpg'
    save_image_path = r'C:\Users\Jim\PycharmProjects\DetectFace3.0\Image'