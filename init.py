from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# features (1 sim, 0 não)

# pelo longo?
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1,1,1,0,0,0]

model = LinearSVC()
model.fit(dados,classes)

animal_misterioso=[1,1,1]
print(model.predict([animal_misterioso]))
animal1=[1,1,1]
animal12=[1,1,0]
animal3=[0,1,1]

test_x=[animal1, animal12, animal3]
previsoes=model.predict(test_x)

test_y=[0, 1, 1]

corretos=(previsoes == test_y).sum()
total= len(test_x)

print(corretos)
print(total)
taxa_de_acerto = accuracy_score(test_y, previsoes)
print("taxa de acerto %.2f" % (taxa_de_acerto *100))

