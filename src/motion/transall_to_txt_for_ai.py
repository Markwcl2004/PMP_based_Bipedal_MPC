import os

def extract_specific_files_to_txt(folder_path, target_files, output_file_name="output.txt"):
    """
    找到指定文件夹及其子文件夹中特定列表中的文件，
    并将这些文件的内容整合输出到一个 txt 文件中。
    
    :param folder_path: 要处理的文件夹路径
    :param target_files: 要查找的目标文件列表
    :param output_file_name: 输出的 txt 文件名，默认为 output.txt
    """
    # 检查目标文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：目标文件夹 {folder_path} 不存在！")
        return

    print(f"目标文件夹路径：{folder_path}")

    # 遍历文件夹及其子文件夹
    specific_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file in target_files:
                file_path = os.path.join(root, file)
                specific_files.append(file_path)

    if not specific_files:
        print(f"未找到任何目标文件。")
        return

    # 输出文件路径
    output_file_path = os.path.join(folder_path, output_file_name)

    # 写入文件内容
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for file_path in specific_files:
            print(f"正在处理文件：{file_path}")
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"警告：文件 {file_path} 不存在，跳过。")
                    output_file.write(f"文件 {file_path} 不存在。\n\n")
                    continue

                # 检查文件是否可读
                if not os.access(file_path, os.R_OK):
                    print(f"警告：文件 {file_path} 无读取权限，跳过。")
                    output_file.write(f"文件 {file_path} 无读取权限。\n\n")
                    continue

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    # 写入文件路径和内容
                    output_file.write(f"文件路径：{file_path}\n")
                    output_file.write("-" * 40 + "\n")  # 添加分隔线
                    output_file.write(content)
                    output_file.write("\n\n")  # 添加空行分隔不同文件的内容
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{e}")
                output_file.write(f"无法读取文件内容：{e}\n\n")

    print(f"所有符合条件的文件内容已提取到 {output_file_path}。")

if __name__ == "__main__":
    # 指定目标文件夹路径为当前 Python 文件所在的目录
    folder_path = os.path.dirname(os.path.abspath(__file__))
    # target_files = [
    #     'ConvexMPC.cc',
    #     'ConvexMPC.h',
    #     'PMPOptimizer.cc',
    #     'PMPOptimizer.h'
    # ]

    target_files = [
        'TrajectoryStabilization.cc',
        'WholeBodyController.cc',
        'StateEstimationLKF.cc',
        'CycleTimer.cc',
        'GaitSchedule.cc',
        'ConvexMPC.cc',
        'FootholdOptimization.cc',
        'PMPOptimizer.cc',
        'TrajectorGeneration.cc'
        'MotionManager.cc',
        'main.cc'
    ]

    extract_specific_files_to_txt(folder_path, target_files)