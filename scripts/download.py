import kagglehub

# Download latest version
path = kagglehub.dataset_download("hiumaidanh/dataset")

print("Path to dataset files:", path)

# cp -r /home/jovyan/.cache/kagglehub/datasets/hiumaidanh/dataset/versions/1/data /home/jovyan/ver_2