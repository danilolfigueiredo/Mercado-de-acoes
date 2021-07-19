from sklearn import tree

lisa = 1
irregular = 0
maça = 1
laranja = 0

pomar = [[150,lisa],[130,lisa],[180,irregular],[160,irregular]]
         
resultado = [maça, maça, laranja, laranja]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(pomar, resultado)

while True:

    peso = input("peso: ")
    superficie = input("superficie (1 = lisa; 0 = irregular):")

    result = clf.predict([[peso,superficie]])

    if result == 1:
        print("maça")

    else:
        print("laranja")
