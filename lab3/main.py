from src.task1_pipeline import Task1Pipeline
from src.task1_alternative_pipeline import Task1AlternativePipeline
from src.task2_pipeline import Task2Pipeline
from src.task3_pipeline import Task3Pipeline
from src.task4_pipeline import Task4Pipeline

def main():
    print("-------------Task 1------------------")
    first_task = Task1Pipeline(csv_path="lab3/data/global_emissions.csv")
    first_task.run()
    print("-------------Task 1 Alternative------------------")
    first_task = Task1AlternativePipeline(csv_path="lab3/data/global_emissions.csv")
    first_task.run()
    print("-------------Task 2------------------")
    second_task = Task2Pipeline(csv_path="lab3/data/global_emissions.csv")
    second_task.run()
    print("-------------Task 3------------------")
    third_task = Task3Pipeline(csv_path="lab3/data/global_emissions.csv")
    third_task.run()
    print("-------------Task 4------------------")
    forth_task = Task4Pipeline(csv_path="lab3/data/global_emissions.csv")
    forth_task.run()


if __name__ == "__main__":
    main()