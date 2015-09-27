# -*- coding: utf-8 -*-

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


from pybrain.structure.modules.svmunit        import SVMUnit
from pybrain.supervised.trainers.svmtrainer   import SVMTrainer



# Criando dataset no formato correto
dataset = SupervisedDataSet(2,1)
dataset.addSample((0,0), (0,))
dataset.addSample((0,1), (1,))
dataset.addSample((1,0), (1,))
dataset.addSample((1,1), (0,))

# Criando a SVM unit
svm = SVMUnit()
# Trainando a SVM
trainer = SVMTrainer( svm, dataset )


# Treinando a rede
err = 1.0
while err > 0.0001:
    err = trainer.train()
    #print err

# Imprimindo ativação da rede para todo o conjunto
print network.activateOnDataset(dataset)

# Imprimindo pesos da rede
for mod in network.modules:
    for conn in network.connections[mod]:
        print conn
        for cc in range(len(conn.params)):
            print conn.whichBuffers(cc), conn.params[cc]
            
# Ativando entradas individualmente
#print network.activate([0,0])
#print network.activate([0,1])
#print network.activate([1,0])
#print network.activate([1,1])
