import h5py


mat_file_path = 'results_ddpm1.mat'  

# 加载.mat文件
with h5py.File(mat_file_path, 'r') as file:
    def print_name(name):
        print(name)
    
    file.visit(print_name)
