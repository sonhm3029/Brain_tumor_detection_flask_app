import os


UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER_TYPE = ['images', 'videos', 'texts', 'others', 'docs', 'audios']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def createUploadFolders():
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
        for typeFolder in UPLOAD_FOLDER_TYPE:
            os.mkdir(f'{UPLOAD_FOLDER}/{typeFolder}')
        print("Uploads folder was created!")


def allowed_file(filename):
    return '.' in filename and\
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
