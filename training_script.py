from rfdetr import RFDETRMedium

model = RFDETRMedium()
DATASET_PATH = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/training_sets/dataset_12"

model.train(
    dataset_dir=DATASET_PATH,
    epochs=50,
    batch_size=1,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_7_0",
    #resume="/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_6_1/checkpoint.pth"
)