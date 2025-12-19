import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)

X = mnist.data  
y = mnist.target.astype(int) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=0
)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


clf_l2 = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
clf_l2.fit(X_train_s, y_train)

# Affichage Matrice de Confusion
y_pred = clf_l2.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matrice de Confusion (L2 - Ridge)")
plt.show()

print("Affichage des coefficients L2")
coefs_l2 = clf_l2.coef_
scale = np.abs(coefs_l2).max()

plt.figure(figsize=(15, 6))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    img = coefs_l2[i].reshape(28, 28) 
    plt.imshow(img, cmap='RdBu', vmin=-scale, vmax=scale)
    plt.title(f'Classe {i}')
    plt.axis('off')
plt.suptitle('Coefficients L2 (Ridge) - "Cartes de chaleur"')
plt.tight_layout()
plt.show()

warnings.filterwarnings("ignore")

clf_l1 = LogisticRegression(
    solver='saga', 
    penalty='l1',   
    C=0.01,           
    tol=0.1,          
    max_iter=200      
)

clf_l1.fit(X_train_s, y_train)

coefs_l1 = clf_l1.coef_
scale_l1 = np.abs(coefs_l1).max()

plt.figure(figsize=(15, 6))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    img = coefs_l1[i].reshape(28, 28)
    plt.imshow(img, cmap='RdBu', vmin=-scale_l1, vmax=scale_l1)
    plt.title(f'Classe {i} (L1)')
    plt.axis('off')

plt.suptitle(f'Coefficients L1 (Lasso) avec C={clf_l1.C}')
plt.tight_layout()
plt.show()

# Comparaison quantitative
l2 = np.mean(np.abs(coefs_l2) < 1e-4) * 100
l1 = np.mean(np.abs(coefs_l1) < 1e-4) * 100

print(f"\n--- Comparaison de la Sparsity ---")
print(f"Pourcentage de coefficients nuls (L2 - Ridge) : {l2:.2f}%")
print(f"Pourcentage de coefficients nuls (L1 - Lasso) : {l1:.2f}%")

plt.hist(coefs_l2.flatten(), bins=100, range=(-0.5, 0.5), alpha=0.5, label='L2 (Ridge)', color='blue')
plt.hist(coefs_l1.flatten(), bins=100, range=(-0.5, 0.5), alpha=0.7, label='L1 (Lasso)', color='red')

plt.yscale('log') 
plt.title("Distribution des coefficients (Échelle Log)")
plt.xlabel("Valeur du coefficient")
plt.ylabel("Nombre de coefficients (Log)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()