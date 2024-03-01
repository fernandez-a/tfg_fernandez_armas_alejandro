from run import Runner

def main():
    num_classes = 3
    num_queries = 100
    null_class_coef = 0.5
    BATCH_SIZE = 8
    LR = 2e-5
    EPOCHS = 100

    runner = Runner(num_classes, num_queries, null_class_coef, BATCH_SIZE, LR, EPOCHS)
    runner.run()

if __name__ == "__main__":
    main()