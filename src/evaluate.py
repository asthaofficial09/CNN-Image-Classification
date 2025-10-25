
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

(_, _), (x_test, y_test) = datasets.cifar10.load_data()
x_test = x_test / 255.0
model = load_model("models/cnn_cifar10_model.h5", compile= False)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.3f}")

y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues", annot=False)
plt.title("Confusion Matrix")
plt.show()
