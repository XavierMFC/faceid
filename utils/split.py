import splitfolders

input_folder = '/code/proj/dataset/lfw_data/lfw_images'
output_folder = '/code/proj/dataset/lfw_data/lfw_split' # 建议输出到新文件夹

# ratio=(训练集比例, 验证集比例)
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None)