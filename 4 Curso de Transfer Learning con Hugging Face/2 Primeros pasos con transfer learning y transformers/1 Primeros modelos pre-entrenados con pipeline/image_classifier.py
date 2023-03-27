from transformers import pipeline

if __name__ == '__main__':

    obj_classification = pipeline(task="image-classification")
    ans = obj_classification("coche.png")
    print(ans)
