import cv2


def get_data(n_classes, n_examples):

    video = cv2.VideoCapture(0)
    current_class = 1
    current_example = 1
    print("Collecting data for class", current_class)
    while video.isOpened():

        ret, frame = video.read()

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print(f"collecting example: {current_example}/{n_examples}")
            cv2.imwrite(f"../dataset/c{current_class}_e{current_example}.png", frame)
            current_example += 1

        if current_example > n_examples:
            current_example = 1
            current_class += 1
            print("Collecting data for class", current_class)

        cv2.putText(frame, f"C{current_class}/{n_classes} E{current_example}/{n_examples}", (20, 40),
                    color=(255, 0, 0), fontScale=1, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("Recording", frame)

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')) or current_class > n_classes:
            video.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    get_data(n_classes=5, n_examples=14)
