使用自建数据集：
    cfg/{OBJECT}.data
    Dataset/[OBJECT]/train.txt
    Dataset/[OBJECT]/test.txt
每个物体需要包含：
    1） 包含图像的文件夹
    2） 包含标签的文件夹（ObjectDatasesetTools）
    3） 包含训练图像文件名的文本文件（train.txt）
    4)  包含测试图像文件名的文本文件（test.txt）
    5） 包含物体对象模型的ply文件（单位为m）
    6） （可选）包含分割mask的文件（ 如果要让训练在背景变化的图像中更加鲁棒，可以增加学习的泛化能力）

make sure：
    改变data configuration file 中的diam值，改为对象模型的直径
    改变相机参数：
            width = 640
            height = 480
            fx = 572.4114 
            fy = 573.5704
            u0 = 325.2611
            v0 = 242.0489
    改变py2/dataset.py数据增强参数  
            jitter = 0.2   #抖动 
            hue = 0.1      #色相
            saturation = 1.5    #饱和度 
            exposure = 1.5      #曝光参数
    改变cfg/yolo-pose.cfg 中训练数据
            max_batches = 80200
            steps=-1,80,160
            scales=0.1,0.1,0.1