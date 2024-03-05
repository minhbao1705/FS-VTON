import firebase_admin
from firebase_admin import credentials, storage
import os

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\vomin\Downloads\\fsvton-18ce5-firebase-adminsdk-cy17x-da5ea0d678.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'gs://fsvton-18ce5.appspot.com'
})

# Đường dẫn đến thư mục chứa các ảnh
folder_path = "C:\\Users\\vomin\\Desktop\\VITON_test"

# Tạo đối tượng bucket Firebase Storage
bucket = storage.bucket()

# Lặp qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    # Kiểm tra nếu file là file ảnh
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Đường dẫn đến file ảnh
        file_path = os.path.join(folder_path, filename)
        # Tải lên file ảnh lên Firebase Storage
        blob = bucket.blob(filename)
        blob.upload_from_filename(file_path)

print("Upload completed.")
