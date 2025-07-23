import os

path = "/mnt/xr_core_ai_asl_llm/"
# path = "/data/users/yangzho6/lingua/mount_pretrain_yangzho6/"

# List all files and directories
files_and_dirs = os.listdir(path)
print(files_and_dirs)
path = os.path.join(path, "tree/")
files_and_dirs = os.listdir(path)
print(files_and_dirs)
path = os.path.join(path, "pretrain_yangzho6/")
files_and_dirs = os.listdir(path)
print("under path {}".format(path))
print(files_and_dirs)
# Print the list
for item in files_and_dirs:
    print(item)
